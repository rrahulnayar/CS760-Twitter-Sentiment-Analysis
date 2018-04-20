import sys
import os
from subprocess import Popen, PIPE
import numpy as np
import pickle as pkl

# part of this script is taken from theano lstm example
# https://raw.githubusercontent.com/kyunghyuncho/DeepLearningTutorials/master/code/imdb_preprocess.py

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_cmd = ['./tokenizer.perl', '-l', 'en', '-q', '-']
def tokenize(sentences):
    print ('Tokenizing..')
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(str.encode(text))
    #print (type(tok_text))
    tok_text = tok_text.decode("utf-8")
    toks = tok_text.split('\n')[:-1]
    print ('Done')

    return toks

def readFiles(filename):
    # cant use pandas for given csv
    # tweet text string has 'commas' within and causes it to be viewed as separate column
    fp = open(filename, 'r')
    lines = fp.readlines()
    sentiment = []
    tweet = []

    # start from 1 to skip header row
    for i in range(1, len(lines)):
        words = lines[i].split(',')
        sentiment.append(int(words[1]))
        text = words[2].strip('\n')
        if (len(words) > 3):
            # stitch text back with comma separation
            for j in range(2+1, len(words)):
                text = text + ',' + words[j].strip('\n')
        tweet.append(text)

    return (tweet, sentiment)

def build_dict(inputFile):
    #read input files
    # trainDF = pd.read_csv(trainFile, index_col='ItemID')
    # textLines = trainDF['SentimentText']
    # tweetLines = textLines.tolist()

    tweetLines, _ = readFiles(inputFile)
    tokens = tokenize(tweetLines)
    print ('Building dictionary..')
    wordcount = dict()
    for ss in tokens:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = list(wordcount.values())
    keys = list(wordcount.keys())

    sorted_idx = np.argsort(counts)[::-1]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print (np.sum(counts), ' total words ', len(keys), ' unique words')

    return worddict

def dictEncodeInput(filename, dictionary):
    # dataDF = pd.read_csv(filename, index_col='ItemID')
    # textLines = dataDF['SentimentText']
    # tweetLines = textLines.tolist()

    tweetLines, labels = readFiles(filename)
    tokens = tokenize(tweetLines)

    seqs = [None] * len(tokens)
    for idx, ss in enumerate(tokens):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs, labels


# ItemID	Sentiment	SentimentText

if __name__ == '__main__':

    trainFile = './data/train_utf8.csv'
    testFile = './data/test_utf8.csv'

    dictionary = build_dict(trainFile)

    trainInput, trainLabel = dictEncodeInput(trainFile, dictionary)
    fpTrain = open('./data/train.pkl', 'wb')
    pkl.dump((trainInput, trainLabel), fpTrain, pkl.HIGHEST_PROTOCOL)
    fpTrain.close()

    # testInput, testLabel = dictEncodeInput(testFile, dictionary)
    # fpTest = open('./data/test.pkl', 'wb')
    # pkl.dump((testInput, testLabel), fpTest, pkl.HIGHEST_PROTOCOL)
    # fpTest.close()

    fpDictionary = open('./data/dictionary.pkl', 'wb')
    pkl.dump(dictionary, fpDictionary, pkl.HIGHEST_PROTOCOL)
    fpDictionary.close()
