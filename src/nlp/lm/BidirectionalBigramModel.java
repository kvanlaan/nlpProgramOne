package nlp.lm;

import java.io.*;
import java.util.*;

public class BidirectionalBigramModel {

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
        // Accumulate unigram and backwardBigram counts in maps
        trainSentences(sentences);
        tokenCount = 0;
        trainSentencesBackward(sentences);
        // Compute final unigram and bidirectionalBigram probs from counts
        calculateProbs();
        calculateProbsBackward();
        calculateProbsBidirectional();
    }

    /**
     * Accumulate unigram and bigram counts for these sentences
     */
    public void trainSentences(List<List<String>> sentences) {
        for (List<String> sentence : sentences) {
            trainSentence(sentence);
        }
    }

    /**
     * Accumulate unigram and backwardBigram counts for these sentences
     */
    public void trainSentencesBackward(List<List<String>> sentences) {
        Collections.reverse(sentences);
        for (List<String> sentence : sentences) {
            trainSentenceBackward(sentence);
        }
    }

    /**
     * Accumulate unigram and backwardBigram counts for this sentence
     */
    public void trainSentenceBackward(List<String> sentence) {
        Collections.reverse(sentence);
        // First count an initial start sentence token
        String prevToken = "</S>";
        DoubleValue unigramValue = backwardUnigramMap.get("</S>");
        unigramValue.increment();
        tokenCount++;
        // For each token in sentence, accumulate a unigram and backwardBigram count
        for (String token : sentence) {
            unigramValue = backwardUnigramMap.get(token);
            // If this is the first time token is seen then count it
            // as an unkown token (<UNK>) to handle out-of-vocabulary 
            // items in testing
            if (unigramValue == null) {
                // Store token in unigram map with 0 count to indicate that
                // token has been seen but not counted
                backwardUnigramMap.put(token, new DoubleValue());
                token = "<UNK>";
                unigramValue = backwardUnigramMap.get(token);
            }
            unigramValue.increment();    // Count unigram
            tokenCount++;               // Count token
            // Make backwardBigram string 
            String backwardBigram = bigram(prevToken, token);
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
        unigramValue = backwardUnigramMap.get("<S>");
        unigramValue.increment();
        tokenCount++;
        // Account for end of sentence backwardBigram
        String backwardBigram = bigram(prevToken, "<S>");
        DoubleValue backwardBigramValue = backwardUnigramMap.get(backwardBigram);
        if (backwardBigramValue == null) {
            backwardBigramValue = new DoubleValue();
            backwardBigramMap.put(backwardBigram, backwardBigramValue);
        }
        backwardBigramValue.increment();
    }

    /**
     * Accumulate unigram and bigram counts for this sentence
     */
    public void trainSentence(List<String> sentence) {
        // First count an initial start sentence token
        String prevToken = "<S>";
        DoubleValue unigramValue = unigramMap.get("<S>");
        unigramValue.increment();
        tokenCount++;
        // For each token in sentence, accumulate a unigram and bigram count
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
            // Make bigram string 
            String bigram = bigram(prevToken, token);
            DoubleValue bigramValue = bigramMap.get(bigram);
            if (bigramValue == null) {
                // If previously unseen bigram, then
                // initialize it with a value
                bigramValue = new DoubleValue();
                bigramMap.put(bigram, bigramValue);
            }
            // Count bigram..
            bigramValue.increment();
            prevToken = token;
        }
        // Account for end of sentence unigram
        unigramValue = unigramMap.get("</S>");
        unigramValue.increment();
        tokenCount++;
        // Account for end of sentence bigram
        String bigram = bigram(prevToken, "</S>");
        DoubleValue bigramValue = bigramMap.get(bigram);
        if (bigramValue == null) {
            bigramValue = new DoubleValue();
            bigramMap.put(bigram, bigramValue);
        }
        bigramValue.increment();
    }

    /**
     * Compute unigram and bigram probabilities from unigram and bigram counts
     */
    public void calculateProbsBackward() {
        // Set bigram values to conditional probability of second token given first
        for (Map.Entry<String, DoubleValue> entry : backwardBigramMap.entrySet()) {
            // An entry in the HashMap maps a token to a DoubleValue
            String backwardBigram = entry.getKey();
            // The value for the token is in the value of the DoubleValue
            DoubleValue value = entry.getValue();
            double backwardBigramCount = value.getValue();
            String token1 = bigramToken1(backwardBigram); // Get first token of backwardBigram
            // Prob is ratio of backwardBigram count to token1 unigram count
            double condProb = backwardBigramCount / backwardUnigramMap.get(token1).getValue();
            // Set map value to conditional probability 
            value.setValue(condProb);
        }
        // Store unigrams with zero count to remove from map
        List<String> zeroTokens = new ArrayList<String>();
        // Set unigram values to unigram probability
        for (Map.Entry<String, DoubleValue> entry : backwardUnigramMap.entrySet()) {
            // An entry in the HashMap maps a token to a DoubleValue
            String token = entry.getKey();
            // Uniggram count is the current map value
            DoubleValue value = entry.getValue();
            double count = value.getValue();
            if (count == 0) // If count is zero (due to first encounter as <UNK>)
            // then remove save it to remove from map
            {
                zeroTokens.add(token);
            } else // Set map value to prob of unigram
            {
                value.setValue(count / tokenCount);
            }
        }
        // Remove zero count unigrams from map
        for (String token : zeroTokens) {
            backwardUnigramMap.remove(token);
        }
    }

    /**
     * Compute unigram and bigram probabilities from unigram and bigram counts
     */
    public void calculateProbs() {
        // Set bigram values to conditional probability of second token given first
        for (Map.Entry<String, DoubleValue> entry : bigramMap.entrySet()) {
            // An entry in the HashMap maps a token to a DoubleValue
            String bigram = entry.getKey();
            // The value for the token is in the value of the DoubleValue
            DoubleValue value = entry.getValue();
            double bigramCount = value.getValue();
            String token1 = bigramToken1(bigram); // Get first token of bigram
            // Prob is ratio of bigram count to token1 unigram count
            double condProb = bigramCount / unigramMap.get(token1).getValue();
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
            if (count == 0) // If count is zero (due to first encounter as <UNK>)
            // then remove save it to remove from map
            {
                zeroTokens.add(token);
            } else // Set map value to prob of unigram
            {
                value.setValue(count / tokenCount);
            }
        }
        // Remove zero count unigrams from map
        for (String token : zeroTokens) {
            unigramMap.remove(token);
        }
    }

    public void calculateProbsBidirectional() {
        for (Map.Entry<String, DoubleValue> entry : unigramMap.entrySet()) {
            // An entry in the HashMap maps a token to a DoubleValue
            String token = entry.getKey();
            // Uniggram count is the current map value
            DoubleValue value = entry.getValue();
            double count = value.getValue();
            // Uniggram count is the current map value
            DoubleValue backwardValue = backwardUnigramMap.get(token);
            double backwardCount = backwardValue.getValue();
            double interpolatedValue = (((count/tokenCount) + (backwardCount/tokenCount)) / 2);
            value.setValue(interpolatedValue);
        }
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

    /**
     * Like test1 but excludes predicting end-of-sentence when computing
     * perplexity
     */
    public void test(List<List<String>> sentences) {
        double totalLogProb = 0;
        double totalNumTokens = 0;
        for (List<String> sentence : sentences) {
            totalNumTokens += sentence.size();
            double sentenceLogProb = sentenceLogProb2(sentence);
            //	    System.out.println(sentenceLogProb + " : " + sentence);
            totalLogProb += sentenceLogProb;
        }
        double perplexity = Math.exp(-totalLogProb / totalNumTokens);


        Collections.reverse(sentences);
        totalLogProb = 0;
        totalNumTokens = 0;
        for (List<String> sentence : sentences) {
            totalNumTokens += sentence.size();
            double sentenceLogProb = sentenceLogProb2(sentence);
            //	    System.out.println(sentenceLogProb + " : " + sentence);
            totalLogProb += sentenceLogProb;
        }
        double backwardPerplexity = Math.exp(-totalLogProb / totalNumTokens);

        System.out.println("Word Perplexity = " + ((perplexity + backwardPerplexity)/2));
    }

    /**
     * Like sentenceLogProb but excludes predicting end-of-sentence when
     * computing prob
     */
    public double sentenceLogProb2(List<String> sentence) {
        String prevToken = "<S>";
        double sentenceLogProb = 0;
        for (String token : sentence) {
            DoubleValue unigramVal = unigramMap.get(token);
            if (unigramVal == null) {
                token = "<UNK>";
                unigramVal = unigramMap.get(token);
            }
            String bigram = bigram(prevToken, token);
            DoubleValue bigramVal = bigramMap.get(bigram);
            double logProb = Math.log(interpolatedProb(unigramVal, bigramVal));
            sentenceLogProb += logProb;
            prevToken = token;
        }
        return sentenceLogProb;
    }

    /**
     * Returns vector of probabilities of predicting each token in the sentence
     * including the end of sentence
     */
    public double[] sentenceTokenProbs(List<String> sentence) {
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
            String backwardBigram = bigram(prevToken, token);
            DoubleValue backwardBigramVal = backwardBigramMap.get(backwardBigram);
            // Store prediction prob for i'th token
            tokenProbs[i] = interpolatedProb(unigramVal, backwardBigramVal);
            prevToken = token;
            i++;
        }
        // Check prediction of end of sentence
        DoubleValue unigramVal = unigramMap.get("</S>");
        String backwardBigram = bigram(prevToken, "</S>");
        DoubleValue backwardBigramVal = backwardBigramMap.get(backwardBigram);
        // Store end of sentence prediction prob
        tokenProbs[i] = interpolatedProb(unigramVal, backwardBigramVal);
        return tokenProbs;
    }

    /**
     * Interpolate bigram prob using bigram and unigram model predictions
     */
    public double interpolatedProb(DoubleValue unigramVal, DoubleValue bigramVal) {
        double bigramProb = 0;
        // In bigram unknown then its prob is zero
        if (bigramVal != null) {
            bigramProb = bigramVal.getValue();
        }
        // Linearly combine weighted unigram and bigram probs
        return lambda1 * unigramVal.getValue() + lambda2 * bigramProb;
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
