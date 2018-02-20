package nlp.lm;

import java.io.*;
import java.util.*;

/** 
 * @author Ray Mooney
 * A simple backwardBigram language model that uses simple fixed-weight interpolation
 * with a unigram model for smoothing.
*/

public class BackwardBigramModel {

    /** Unigram model that maps a token to its unigram probability */
    public Map<String, DoubleValue> unigramMap = null; 

    /**  BackwardBigram model that maps a backwardBigram as a string "A\nB" to the
     *   P(B | A) */
    public Map<String, DoubleValue> backwardBigramMap = null;

    /** Total count of tokens in training data */
    public double tokenCount = 0;

    /** Interpolation weight for unigram model */
    public double lambda1 = 0.1;

    /** Interpolation weight for backwardBigram model */
    public double lambda2 = 0.9;

    /** Initialize model with empty hashmaps with initial
     *  unigram entries for setence start (<S>), sentence end (</S>)
     *  and unknown tokens */
    public BackwardBigramModel() {
	unigramMap = new HashMap<String, DoubleValue>();
	backwardBigramMap = new HashMap<String, DoubleValue>();
	unigramMap.put("<S>", new DoubleValue());
	unigramMap.put("</S>", new DoubleValue());
	unigramMap.put("<UNK>", new DoubleValue());
    }

    /** Train the model on a List of sentences represented as
     *  Lists of String tokens */
    public void train (List<List<String>> sentences) {
	// Accumulate unigram and backwardBigram counts in maps
	trainSentences(sentences);
	// Compure final unigram and backwardBigram probs from counts
	calculateProbs();
    }

    /** Accumulate unigram and backwardBigram counts for these sentences */
    public void trainSentences (List<List<String>> sentences) {
	for (List<String> sentence : sentences) {
	    trainSentence(sentence);
	}
    }

    /** Accumulate unigram and backwardBigram counts for this sentence */
    public void trainSentence (List<String> sentence) {
        Collections.reverse(sentence);
	// First count an initial start sentence token
	String prevToken = "</S>";
	DoubleValue unigramValue = unigramMap.get("</S>");
	unigramValue.increment();
	tokenCount++;
	// For each token in sentence, accumulate a unigram and backwardBigram count
	for (String token : sentence) {
	    unigramValue = unigramMap.get(token);
	    // If this is the first time token is seen then count it
	    // as an unkown token (<UNK>) to handle out-of-vocabulary 
	    // items in testing
	    if (unigramValue == null) {
		// Store token in unigram map with 0 count to indicate that
		// token has been seen but not counted
		unigramMap.put(token, new DoubleValue());
		token = "<UNK>";
		unigramValue = unigramMap.get(token);
	    }
	    unigramValue.increment();    // Count unigram
	    tokenCount++;               // Count token
	    // Make backwardBigram string 
	    String backwardBigram = backwardBigram(prevToken, token);
	    DoubleValue backwardBigramValue = backwardBigramMap.get(backwardBigram);
	    if (backwardBigramValue == null) {
		// If previously unseen backwardBigram, then
		// initialize it with a value
		backwardBigramValue = new DoubleValue();
		backwardBigramMap.put(backwardBigram, backwardBigramValue);
	    }
	    // Count backwardBigram..
	    backwardBigramValue.increment();
	    prevToken = token;
	}
	// Account for end of sentence unigram
	unigramValue = unigramMap.get("<S>");
	unigramValue.increment();
	tokenCount++;
	// Account for end of sentence backwardBigram
	String backwardBigram = backwardBigram(prevToken, "<S>");
	DoubleValue backwardBigramValue = backwardBigramMap.get(backwardBigram);
	if (backwardBigramValue == null) {
	    backwardBigramValue = new DoubleValue();
	    backwardBigramMap.put(backwardBigram, backwardBigramValue);
	}
	backwardBigramValue.increment();
    }

    /** Compute unigram and backwardBigram probabilities from unigram and backwardBigram counts */
    public void calculateProbs() {
	// Set backwardBigram values to conditional probability of second token given first
	for (Map.Entry<String, DoubleValue> entry : backwardBigramMap.entrySet()) {
	    // An entry in the HashMap maps a token to a DoubleValue
	    String backwardBigram = entry.getKey();
	    // The value for the token is in the value of the DoubleValue
	    DoubleValue value = entry.getValue();
	    double backwardBigramCount = value.getValue();
	    String token1 = backwardBigramToken1(backwardBigram); // Get first token of backwardBigram
	    // Prob is ratio of backwardBigram count to token1 unigram count
	    double condProb = backwardBigramCount / unigramMap.get(token1).getValue();
	    // Set map value to conditional probability 
	    value.setValue(condProb);
	}
	// Store unigrams with zero count to remove from map
	List<String> zeroTokens = new ArrayList<String>();
	// Set unigram values to unigram probability
	for (Map.Entry<String, DoubleValue> entry : unigramMap.entrySet()) {
	    // An entry in the HashMap maps a token to a DoubleValue
	    String token = entry.getKey();
	    // Uniggram count is the current map value
	    DoubleValue value = entry.getValue();
	    double count = value.getValue();
	    if (count == 0) 
		// If count is zero (due to first encounter as <UNK>)
		// then remove save it to remove from map
		zeroTokens.add(token);
	    else
		// Set map value to prob of unigram
		value.setValue(count / tokenCount);
	}
	// Remove zero count unigrams from map
	for (String token : zeroTokens) 
	    unigramMap.remove(token);
    }

    /** Return backwardBigram string as two tokens separated by a newline */
    public String backwardBigram (String prevToken, String token) {
	return prevToken + "\n" + token;
    }

    /** Return fist token of backwardBigram (substring before newline) */
    public String backwardBigramToken1 (String backwardBigram) {
	int newlinePos = backwardBigram.indexOf("\n");
	return backwardBigram.substring(0,newlinePos);
    }

    /** Return second token of backwardBigram (substring after newline) */
    public String backwardBigramToken2 (String backwardBigram) {
	int newlinePos = backwardBigram.indexOf("\n");
	return backwardBigram.substring(newlinePos + 1, backwardBigram.length());
    }

    /** Print model as lists of unigram and backwardBigram probabilities */
    public void print() {
	System.out.println("Unigram probs:");
	for (Map.Entry<String, DoubleValue> entry : unigramMap.entrySet()) {
	    // An entry in the HashMap maps a token to a DoubleValue
	    String token = entry.getKey();
	    // The value for the token is in the value of the DoubleValue
	    DoubleValue value = entry.getValue();
	    System.out.println(token + " : " + value.getValue());
	}
	System.out.println("\nBigram probs:");
	for (Map.Entry<String, DoubleValue> entry : backwardBigramMap.entrySet()) {
	    // An entry in the HashMap maps a token to a DoubleValue
	    String backwardBigram = entry.getKey();
	    // The value for the token is in the value of the DoubleValue
	    DoubleValue value = entry.getValue();
	    System.out.println(backwardBigramToken2(backwardBigram) + " given " + backwardBigramToken1(backwardBigram) + 
			       " : " + value.getValue());
	}
  }

    /** Use sentences as a test set to evaluate the model. Print out perplexity
     *  of the model for this test data */
    public void test (List<List<String>> sentences) {
	// Compute log probability of sentence to avoid underflow
	double totalLogProb = 0;
	// Keep count of total number of tokens predicted
	double totalNumTokens = 0;
	// Accumulate log prob of all test sentences
	for (List<String> sentence : sentences) {
	    // Num of tokens in sentence plus 1 for predicting </S>
	    totalNumTokens += sentence.size() + 1;
	    // Compute log prob of sentence
	    double sentenceLogProb = sentenceLogProb(sentence);
	    //	    System.out.println(sentenceLogProb + " : " + sentence);
	    // Add to total log prob (since add logs to multiply probs)
	    totalLogProb += sentenceLogProb;
	}
	// Given log prob compute perplexity
	double perplexity = Math.exp(-totalLogProb / totalNumTokens);
	System.out.println("Perplexity = " + perplexity );
    }
    
    /* Compute log probability of sentence given current model */
    public double sentenceLogProb (List<String> sentence) {
	// Set start-sentence as initial token
	String prevToken = "<S>";
	// Maintain total sentence prob as sum of individual token
	// log probs (since adding logs is same as multiplying probs)
	double sentenceLogProb = 0;
	// Check prediction of each token in sentence
	for (String token : sentence) {
	    // Retrieve unigram prob
	    DoubleValue unigramVal = unigramMap.get(token);
	    if (unigramVal == null) {
		// If token not in unigram model, treat as <UNK> token
		token = "<UNK>";
		unigramVal = unigramMap.get(token);
	    }
	    // Get backwardBigram prob
	    String backwardBigram = backwardBigram(prevToken, token);
	    DoubleValue backwardBigramVal = backwardBigramMap.get(backwardBigram);
	    // Compute log prob of token using interpolated prob of unigram and backwardBigram
	    double logProb = Math.log(interpolatedProb(unigramVal, backwardBigramVal));
	    // Add token log prob to sentence log prob
	    sentenceLogProb += logProb;
	    // update previous token and move to next token
	    prevToken = token;
	}
	// Check prediction of end of sentence token
	DoubleValue unigramVal = unigramMap.get("</S>");
	String backwardBigram = backwardBigram(prevToken, "</S>");
	DoubleValue backwardBigramVal = backwardBigramMap.get(backwardBigram);
	double logProb = Math.log(interpolatedProb(unigramVal, backwardBigramVal));
	// Update sentence log prob based on prediction of </S>
	sentenceLogProb += logProb;
	return sentenceLogProb;
    }

    /** Like test1 but excludes predicting end-of-sentence when computing perplexity */
    public void test2 (List<List<String>> sentences) {
	double totalLogProb = 0;
	double totalNumTokens = 0;
	for (List<String> sentence : sentences) {
	    totalNumTokens += sentence.size();
	    double sentenceLogProb = sentenceLogProb2(sentence);
	    //	    System.out.println(sentenceLogProb + " : " + sentence);
	    totalLogProb += sentenceLogProb;
	}
	double perplexity = Math.exp(-totalLogProb / totalNumTokens);
	System.out.println("Word Perplexity = " + perplexity );
    }
    
    /** Like sentenceLogProb but excludes predicting end-of-sentence when computing prob */
    public double sentenceLogProb2 (List<String> sentence) {
	String prevToken = "<S>";
	double sentenceLogProb = 0;
	for (String token : sentence) {
	    DoubleValue unigramVal = unigramMap.get(token);
	    if (unigramVal == null) {
		token = "<UNK>";
		unigramVal = unigramMap.get(token);
	    }
	    String backwardBigram = backwardBigram(prevToken, token);
	    DoubleValue backwardBigramVal = backwardBigramMap.get(backwardBigram);
	    double logProb = Math.log(interpolatedProb(unigramVal, backwardBigramVal));
	    sentenceLogProb += logProb;
	    prevToken = token;
	}
	return sentenceLogProb;
    }

    /** Returns vector of probabilities of predicting each token in the sentence
     *  including the end of sentence */
    public double[] sentenceTokenProbs (List<String> sentence) {
	// Set start-sentence as initial token
	String prevToken = "<S>";
	// Vector for storing token prediction probs
	double[] tokenProbs = new double[sentence.size() + 1];
	// Token counter
	int i = 0;
	// Compute prob of predicting each token in sentence
	for (String token : sentence) {
	    DoubleValue unigramVal = unigramMap.get(token);
	    if (unigramVal == null) {
		token = "<UNK>";
		unigramVal = unigramMap.get(token);
	    }
	    String backwardBigram = backwardBigram(prevToken, token);
	    DoubleValue backwardBigramVal = backwardBigramMap.get(backwardBigram);
	    // Store prediction prob for i'th token
	    tokenProbs[i] = interpolatedProb(unigramVal, backwardBigramVal);
	    prevToken = token;
	    i++;
	}
	// Check prediction of end of sentence
	DoubleValue unigramVal = unigramMap.get("</S>");
	String backwardBigram = backwardBigram(prevToken, "</S>");
	DoubleValue backwardBigramVal = backwardBigramMap.get(backwardBigram);
	// Store end of sentence prediction prob
	tokenProbs[i] = interpolatedProb(unigramVal, backwardBigramVal);
	return tokenProbs;
    }

    /** Interpolate backwardBigram prob using backwardBigram and unigram model predictions */	 
    public double interpolatedProb(DoubleValue unigramVal, DoubleValue backwardBigramVal) {
	double backwardBigramProb = 0;
	// In backwardBigram unknown then its prob is zero
	if (backwardBigramVal != null)
	    backwardBigramProb = backwardBigramVal.getValue();
	// Linearly combine weighted unigram and backwardBigram probs
	return lambda1 * unigramVal.getValue() + lambda2 * backwardBigramProb;
    }

    public static int wordCount (List<List<String>> sentences) {
	int wordCount = 0;
	for (List<String> sentence : sentences) {
	    wordCount += sentence.size();
	}
	return wordCount;
    }

    /** Train and test a backwardBigram model.
     *  Command format: "nlp.lm.BigramModel [DIR]* [TestFrac]" where DIR 
     *  is the name of a file or directory whose LDC POS Tagged files should be 
     *  used for input data; and TestFrac is the fraction of the sentences
     *  in this data that should be used for testing, the rest for training.
     *  0 < TestFrac < 1
     *  Uses the last fraction of the data for testing and the first part
     *  for training.
     */
    public static void main(String[] args) throws IOException {
//        args = new String[2];
//        args[0] = "test/atis3.pos";
//        args[1] = "0.1";
	// All but last arg is a file/directory of LDC tagged input data
	File[] files = new File[args.length - 1];
	for (int i = 0; i < files.length; i++) 
	    files[i] = new File(args[i]);
	// Last arg is the TestFrac
	double testFraction = Double.valueOf(args[args.length -1]);
	// Get list of sentences from the LDC POS tagged input files
	List<List<String>> sentences = 	POSTaggedFile.convertToTokenLists(files);
	int numSentences = sentences.size();
	// Compute number of test sentences based on TestFrac
	int numTest = (int)Math.round(numSentences * testFraction);
	// Take test sentences from end of data
	List<List<String>> testSentences = sentences.subList(numSentences - numTest, numSentences);
	// Take training sentences from start of data
	List<List<String>> trainSentences = sentences.subList(0, numSentences - numTest);
	System.out.println("# Train Sentences = " + trainSentences.size() + 
			   " (# words = " + wordCount(trainSentences) + 
			   ") \n# Test Sentences = " + testSentences.size() +
			   " (# words = " + wordCount(testSentences) + ")");
	// Create a backwardBigram model and train it.
	BackwardBigramModel model = new BackwardBigramModel();
	System.out.println("Training...");
        Collections.reverse(trainSentences);
	model.train(trainSentences);
	// Test on training data using test and test2
	model.test(trainSentences);
	model.test2(trainSentences);
	System.out.println("Testing...");
	// Test on test data using test and test2
        Collections.reverse(testSentences);
	model.test(testSentences);
	model.test2(testSentences);
    }

}
