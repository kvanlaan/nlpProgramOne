package nlp.lm;

import java.io.*;
import java.util.*;

public class BidirectionalBigramModel {

    /**
     * Unigram model that maps a token to its unigram probability
     */
    public BackwardBigramModel backwardBigramModel = null;

    /**
     * Unigram model that maps a token to its unigram probability
     */
    public BigramModel bigramModel = null;
    /**
     * Unigram model that maps a token to its unigram probability
     */
    public Map<String, DoubleValue> backwardUnigramMap = null;

    /**
     * Unigram model that maps a token to its unigram probability
     */
    public Map<String, DoubleValue> unigramMap = null;

    /**
     * Bigram model that maps a bigram as a string "A\nB" to the P(B | A)
     */
    public Map<String, DoubleValue> bigramMap = null;

    /**
     * BackwardBigram model that maps a backwardBigram as a string "A\nB" to the
     * P(B | A)
     */
    public Map<String, DoubleValue> backwardBigramMap = null;

    /**
     * Total count of tokens in training data
     */
    public double tokenCount = 0;

    /**
     * Interpolation weight for unigram model
     */
    public double lambda1 = 0.1;

    /**
     * Interpolation weight for bigram model
     */
    public double lambda2 = 0.9;

    /**
     * Initialize model with empty hashmaps with initial unigram entries for
     * setence start (<S>), sentence end (</S>) and unknown tokens
     */
    public BidirectionalBigramModel() {
        unigramMap = new HashMap<String, DoubleValue>();
        bigramMap = new HashMap<String, DoubleValue>();

        unigramMap.put("<S>", new DoubleValue());
        unigramMap.put("</S>", new DoubleValue());
        unigramMap.put("<UNK>", new DoubleValue());

        backwardUnigramMap = new HashMap<String, DoubleValue>();
        backwardBigramMap = new HashMap<String, DoubleValue>();

        backwardUnigramMap.put("<S>", new DoubleValue());
        backwardUnigramMap.put("</S>", new DoubleValue());
        backwardUnigramMap.put("<UNK>", new DoubleValue());
    }

    /**
     * Return bigram string as two tokens separated by a newline
     */
    public String bigram(String prevToken, String token) {
        return prevToken + "\n" + token;
    }

    /**
     * Return fist token of bigram (substring before newline)
     */
    public String bigramToken1(String bigram) {
        int newlinePos = bigram.indexOf("\n");
        return bigram.substring(0, newlinePos);
    }

    /**
     * Return second token of bigram (substring after newline)
     */
    public String bigramToken2(String bigram) {
        int newlinePos = bigram.indexOf("\n");
        return bigram.substring(newlinePos + 1, bigram.length());
    }

    /**
     * Train the model on a List of sentences represented as Lists of String
     * tokens
     */
    public void train(List<List<String>> sentences) {
        bigramModel = new BigramModel();
        backwardBigramModel = new BackwardBigramModel();
        // Accumulate unigram and backwardBigram counts in maps
        bigramModel.train(sentences);
        backwardBigramModel.train(sentences);
        // Compute final unigram and bidirectionalBigram probs from counts
        unigramMap = bigramModel.unigramMap;
        backwardBigramMap = backwardBigramModel.backwardBigramMap;
        bigramMap = bigramModel.bigramMap;
    }

    /**
     * Print model as lists of unigram and backwardBigram probabilities
     */
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
            System.out.println(bigramToken2(backwardBigram) + " given " + bigramToken1(backwardBigram)
                    + " : " + value.getValue());
        }
    }

    /** Like test1 but excludes predicting end-of-sentence when computing perplexity */
    public void test (List<List<String>> sentences) {
	double totalLogProb = 0;
	double totalNumTokens = 0;
	for (List<String> sentence : sentences) {
	    totalNumTokens += sentence.size();
	    double sentenceLogProb = sentenceLogProb(sentence);
	    //	    System.out.println(sentenceLogProb + " : " + sentence);
	    totalLogProb += sentenceLogProb;
	}
	double perplexity = Math.exp(-totalLogProb / totalNumTokens);
	System.out.println("Word Perplexity = " + perplexity );
    }

    /**
     * Like sentenceLogProb but excludes predicting end-of-sentence when
     * computing prob
     */
    public double sentenceLogProb(List<String> sentence) {
        String prevToken = "<S>";
        double sentenceLogProb = 0;
        int index = 0;
        for (String token : sentence) {
            DoubleValue unigramVal = unigramMap.get(token);
            if (unigramVal == null) {
                token = "<UNK>";
                unigramVal = unigramMap.get(token);
            }
            String bigram = bigram(prevToken, token);
            DoubleValue bigramVal = bigramMap.get(bigram);
            String nextToken;
            if ((index + 1) < sentence.size()) {
                nextToken = sentence.get(index + 1);
            } else {
                nextToken = "</S>";
            }
            String backwardBigram = bigram(nextToken, token);
            DoubleValue backwardBigramVal = backwardBigramMap.get(backwardBigram);
            double logProb = Math.log(interpolatedProb(unigramVal, bigramVal, backwardBigramVal));
            sentenceLogProb += logProb;
            prevToken = token;
            index++;
        }
        return sentenceLogProb;
    }

    /**
     * Interpolate bigram prob using bigram and unigram model predictions
     */
    public double interpolatedProb(DoubleValue unigramVal, DoubleValue bigramVal, DoubleValue backwardBigramVal) {
        double bigramProb = 0;
        // In bigram unknown then its prob is zero
        if (bigramVal != null) {
            bigramProb = bigramVal.getValue();
        }
        double backwardBigramProb = 0;
        if (backwardBigramVal != null) {
            backwardBigramProb = backwardBigramVal.getValue();
        }
        // Linearly combine weighted unigram and bigram probs
        return lambda1 * unigramVal.getValue() + ((lambda2/2) * (bigramProb + backwardBigramProb));
    }

    public static int wordCount(List<List<String>> sentences) {
        int wordCount = 0;
        for (List<String> sentence : sentences) {
            wordCount += sentence.size();
        }
        return wordCount;
    }

    /**
     * Train and test a backwardBigram model. Command format:
     * "nlp.lm.BigramModel [DIR]* [TestFrac]" where DIR is the name of a file or
     * directory whose LDC POS Tagged files should be used for input data; and
     * TestFrac is the fraction of the sentences in this data that should be
     * used for testing, the rest for training. 0 < TestFrac < 1 Uses the last
     * fraction of the data for testing and the first part for training.
     */
    public static void main(String[] args) throws IOException {
        // All but last arg is a file/directory of LDC tagged input data
        File[] files = new File[args.length - 1];
        for (int i = 0; i < files.length; i++) {
            files[i] = new File(args[i]);
        }
        // Last arg is the TestFrac
        double testFraction = Double.valueOf(args[args.length - 1]);
        // Get list of sentences from the LDC POS tagged input files
        List<List<String>> sentences = POSTaggedFile.convertToTokenLists(files);
        int numSentences = sentences.size();
        // Compute number of test sentences based on TestFrac
        int numTest = (int) Math.round(numSentences * testFraction);
        // Take test sentences from end of data
        List<List<String>> testSentences = sentences.subList(numSentences - numTest, numSentences);
        // Take training sentences from start of data
        List<List<String>> trainSentences = sentences.subList(0, numSentences - numTest);
        System.out.println("# Train Sentences = " + trainSentences.size()
                + " (# words = " + wordCount(trainSentences)
                + ") \n# Test Sentences = " + testSentences.size()
                + " (# words = " + wordCount(testSentences) + ")");
        // Create a backwardBigram model and train it.
        BidirectionalBigramModel model = new BidirectionalBigramModel();
        System.out.println("Training...");
        model.train(trainSentences);
        // Test on training data using test and test2
        model.test(trainSentences);
        System.out.println("Testing...");
        // Test on test data using test and test2
        model.test(testSentences);
    }
}
