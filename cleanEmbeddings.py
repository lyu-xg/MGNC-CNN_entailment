from os.path import expanduser, exists
import numpy as np
from numpy import array
from collections import defaultdict
import re
from data_helpers import clean_str
import string
import pickle

def saveas(filename,content):
    with open(filename,'w') as outfile:
        outfile.write(repr(content))
    return 1

home = expanduser("~")
W2VDir = home + '/DATA/GoogleNews-vectors-negative300.bin'
GloVeDir = home + '/DATA/glove.6B/glove.6B.300d.txt'
DBwordDir = home + '/DATA/deps.words'
data_folder = ["./SNLI/train.txt","./SNLI/valid.txt","./SNLI/test.txt"]

def cleanSentence(sentence):
    sentence = clean_str(sentence.translate(None,string.punctuation))
    sentence = sentence.lower().split()
    res = []
    for i in range(112): # the max length of sentences in the given corpus
        if i<len(sentence):
            res.append(sentence[i])
        else:
            res.append('<pad>')
    return res

def readDataFile(filename,vocab):
    revs = []
    with open(filename, "rb") as infile:
        for idx,line in enumerate(infile):
            if not idx%1000: print('reading entry no.{} in {}.'.format(idx,filename))
            data = line.split("\t")
            label = data[0]
            label = label.strip()
            sentence = data[1]
            sentence = sentence.strip()
            entry  = {"label":label,
                      "text":cleanSentence(sentence),
                      "idx":idx}
            revs.append(entry)
            for w in entry['text']:
                vocab[w] += 1
    return revs,vocab

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)


def getVectors(path,vocab):
    word_vecs = {}
    with open(path,'r') as infile:
        for line in infile:
            line = line.split()
            if line[0] in vocab:
                word_vecs[line[0]] = line[1:]
    add_unknown_words(word_vecs, vocab)
    return word_vecs

def getDBword(vocab):
    return getVectors(DBwordDir,vocab)

def getGloVe(vocab):
    return getVectors(GloVeDir,vocab)

def getW2V(vocab):
    """
    Loads 300x1 word vecs from Google word2vecs
    Some lines copied from Todor Davchev's work on sentence classification
    """
    word_vecs = {}
    with open(W2VDir, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    add_unknown_words(word_vecs, vocab)
    return word_vecs


def build_data():
    train, _, test = data_folder
    vocab = defaultdict(int)
    trainData, vocab = readDataFile(train,vocab)
    testData,  vocab = readDataFile(test,vocab)
    vocabList = makeVocabIndex(vocab)


    result = [(trainData,testData),vocab,vocabList]
    pickle.dump(result,open('all_data.p','wb'))
    return result

def load_data():
    print('reading parsed data from cache.')
    return pickle.load(open('all_data.p','rb'))

def makeVocabIndex(vocab):
    return list(vocab.keys())

def buildEmbeddingMatrix(vocabList,embeddingTable):
    res = np.zeros(shape=(len(vocabList),len(embeddingTable['hello'])))
    for i,w in enumerate(vocabList):
        res[i] = embeddingTable[w]
    return res

def buildDataMatrix(vocabList,data):
    indexLookup = {w:i for i,w in enumerate(vocabList)}
    res = []
    for entry in data:
        res.append(map(lambda w:indexLookup[w],entry['text']))
    return np.array(res)

def buildLableMatrix(data):
    res = []
    for entry in data:
        temp = [0,0,0]
        l = int(entry['label'])
        temp[l] = 1
        res.append(temp)
    return res


def dumpMatrixes(vocabList,trainData,testData):
    pickle.dump(buildDataMatrix(vocabList,trainData),open('train.p','wb'))
    pickle.dump(buildDataMatrix(vocabList,testData),open('test.p','wb'))
    pickle.dump(buildLableMatrix(trainData),open('trainLabel.p','wb'))
    pickle.dump(buildLableMatrix(testData),open('testLabel.p','wb'))
    #saveas('w2v.pyon',matrix)
    #saveas('trainData.pyon',buildDataMatrix(vocabList,trainData))
    #saveas('trainLabel.pyon',buildLableMatrix(trainData))
    #saveas('testData.pyon',buildDataMatrix(vocabList,testData))
    #saveas('testLabel.pyon',buildLableMatrix(testData))


def main():
    (trainData,testData),vocab,vocabList = build_data() if not exists('all_data.p') else load_data()
    print('got {} entries of train data.'.format(len(trainData)))
    print('got {} entries of test data.'.format(len(testData)))
    print('got {} vocab.'.format(len(vocab)))
    print('just to make sure, vocab is a {}'.format(type(vocab)))
    pickle.dump(buildEmbeddingMatrix(vocabList,getW2V(vocab)),open('w2v.p','wb'))
    pickle.dump(buildEmbeddingMatrix(vocabList,getDBword(vocab)),open('dbword.p','wb'))
    pickle.dump(buildEmbeddingMatrix(vocabList,getGloVe(vocab)),open('glove.p','wb'))
    dumpMatrixes(vocabList,trainData,testData)


if __name__ == "__main__":
    main()

# Legacy

def fineMaxLen(trainData,testData):
    cur = float('-inf')
    for e in trainData:
        if len(e['text'])>cur:
            cur = len(e['text'])
    for e in testData:
        if len(e['text'])>cur:
            cur = len(e['text'])
    return cur
