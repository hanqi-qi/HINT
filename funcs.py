import logging
import argparse
import copy
import numpy as np
import torch
from torch import nn
from torch import optim
from collections import Counter
from itertools import chain
#from ipdb import launch_ipdb_on_exception
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib
import pickle
import json
from gensim.models import KeyedVectors
import dill
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

def load_process(dataset,type,data):
    filename = filename = '/home/hanqiyan/latent_topic/SINE/input_data/{}/{}_{}_list.pkl'.format(dataset,type,data)
    with open(filename, 'rb') as f:
        result = pickle.load(f)
    return result


class input_example:
    def __init__(self):
        self.id = None
        self.word = None
    def feed(self,id,word):
        self.id = id
        self.word = word

def convert_words2ids(words, vocab, unk, max_value, sos=None, eos=None,use_stem=False):
    """ convert string sequence into word id sequence
        Args:
            words (list): word sequence
            vocab (dict): word-id mapping
            unk (int): id of unkown word "<unk>"
            sos (int): id of start-of-sentence symbol "<sos>"
            eos (int): id of end-of-sentence symbol "eos"
        Returns:
            numpy array of word ids sequence
    """
    # id_list = [ vocab[w] if w in vocab else unk for w in words ] #ori_code
    #make the input date more clear and formal
    id_list = []
    word_list = []
    for w in words:
        w = w.strip('\'"?!,.():;')
        w = w.lower()
        if use_stem:
            porter_stemmer = PorterStemmer()
            w = str(porter_stemmer.stem(w))
        if w in vocab and len(w) > 0:
            id_list.append(vocab[w])
            word_list.append(w)
        # else:
        #     id_list.append(unk)

    # if sos is not None:
    #     id_list.insert(0, sos)
    # if eos is not None:
    #     id_list.append(2)
    #     word_list.append(eos)
    # sen_len = len(id_list)
    # if eos is not None:
    #     for pad_i in range(max_value-sen_len): #pad to max_value with '<eos>'
    #         id_list.append(eos)
    #         word_list.append("<eos>")

    #only return the sentence longer than 3 words
    if len(id_list) > 3:
        result = input_example()
        result.feed(id_list,word_list)
        return result

def getVectors(embed_dim,wordvocab):
    vectors = []
    id2word = {}
    hit = 0
    for word in wordvocab.keys():
        id2word[wordvocab[word]]=word
    # id2word = wordvocab
    if id2word != None:
        word2vec = KeyedVectors.load_word2vec_format('/mnt/Data3/hanqiyan/latent_topic/lin_absa/yelp_code/GoogleNews-vectors-negative300.bin', binary=True)
        for i in range(len(id2word)):
            word = id2word[i]
            if word in word2vec.vocab:
                vectors.append(word2vec[word])
                hit +=1
            else:
                vectors.append(np.random.uniform(-0.01, 0.01, embed_dim))
    else:
        for i in range(len(wordvocab)):
            vectors.append(np.random.uniform(-0.01, 0.01, embed_dim))
    hit_rate = float(hit)/len(id2word)
    print(("The hit rate is {}".format(hit_rate)))
    return np.array(vectors),hit_rate

def get_vocabulary(textfile, initial_vocab={'<unk>':0,'<sssss>':1,'<eos>':2}, vocabsize=0,dataset='yelp',use_stem=False):
    """ acquire vocabulary from dialog text corpus
        Args:
            textfile (str): filename of a dialog corpus
            initial_vocab (dict): initial word-id mapping
            vocabsize (int): upper bound of vocabulary size (0 means no limitation)
        Return:
            dict of word-id mapping
    """
    vocab = copy.copy(initial_vocab)
    word_count = Counter()
    for line in open(textfile,'r').readlines():
        if dataset =="yelp":
            _,_,label,text = line.strip().split("\t\t")
        elif dataset == 'imdb':
            # label = imbd_dic[json.loads(line)['sentiment']]
            text = json.loads(line)['text']
        elif dataset =='guardian_news':
            text, label = line.strip().split("\t")
        for w in text.split(): # skip speaker indicator
            w = w.strip('\'"?!,.():;')
            w = w.lower()
            if use_stem:
                porter_stemmer = PorterStemmer()
                w = str(porter_stemmer.stem(w))
            word_count[w] += 1

    # if vocabulary size is specified, most common words are selected
    if vocabsize > 0:
        for w in word_count.most_common(vocabsize):
            if w[0] not in vocab:
                vocab[w[0]] = len(vocab)
                if len(vocab) >= vocabsize:
                    break
    else: # all observed words are stored
        for w in word_count:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab

def load_embedding(word_id_dict,embedding_file_name="./data/glove.840B.300d.txt",embedding_size=300):
    '''less word overlap for imdb'''
    embedding_length = len(word_id_dict)
    embedding_matrix = np.random.uniform(-1e-2,1e-2,size=(embedding_length,embedding_size))
    embedding_matrix[0] = 0
    hit = 0
    with open(embedding_file_name,"r") as f:
        for line in f:
            splited_line = line.strip().split(" ")
            word,embeddings = splited_line[0],splited_line[1:]
            if word in word_id_dict:
                word_index = word_id_dict[word]
                embedding_array = np.fromstring("\n".join(embeddings),dtype=np.float32,sep="\n")
                embedding_matrix[word_index] = embedding_array
                hit += 1
    hit_rate = float(hit)/embedding_length
    logging.info("\nThe hit rate is :{0:8d}".format(hit_rate))
    # print(("The hit rate is {}".format(hit_rate)))
    return embedding_matrix

def convert_words2tfidf(input_example, vocab, unk, max_value, sos=None, eos=None,doc_id=None,doc_tfidf=None):
    """ convert string sequence into tf-idf id sequence
        Args:
            words (list): word sequence
            vocab (dict): word-id mapping
            unk (int): id of unkown word "<unk>"
            sos (int): id of start-of-sentence symbol "<sos>"
            eos (int): id of end-of-sentence symbol "eos"
        Returns:
            numpy array of word ids sequence
    """
    # id_list = [ vocab[w] if w in vocab else unk for w in words ] #ori_code
    #make the input date more clear and formal
    # id_list = []
    
    docs = []
    words = []
    for sample in input_example:
        docs.append(" ".join(sample.word))
        words = words + [word for word in sample.word]

    vocab_list = list(set(words))
    cv = CountVectorizer(vocabulary=vocab_list)
    tfidf_transformer = TfidfTransformer(norm="l1")
    word_count_vector = cv.fit_transform(docs)#transform sentence to bow
    X = tfidf_transformer.fit_transform(word_count_vector)
    feature_names = vocab_list

    #map the tf-idf value to word
    tfidf_list = []
    for sen_id,sample in enumerate(input_example):
        sen_tfidf = []
        for word in sample.word:
            idx = feature_names.index(word)
            #some word tfidf is zeros, if so, make it non-zero by adding 1e-4
            if X[sen_id,idx] == 0.0:
                sen_tfidf.append(1e-4)
            else:
                sen_tfidf.append(X[sen_id,idx])
        tfidf_list.append(sen_tfidf)
    return tfidf_list
def batch_from_list(textlist, labellist):
    batch = B()
    batch.text = textlist[0]
    batch.label = labellist[0]
    for txt, la in zip(textlist[1:], labellist[1:]):
        batch.text = torch.cat((batch.text, txt), 0)
        # you may need to change the type of "la" to torch.tensor for different datasets, sorry for the inconvenience
        batch.label = torch.cat((batch.label, la), 0) # for SST and IMDB dataset, you do not need to change "la" type
    batch.text = batch.text.to("cuda:0")
    batch.label = batch.label.to("cuda:0")
    return batch

def load(textfile, vocab, max_value, max_utterance,dataset,type=None):
    """Only padding the input sentence, sentence number in each document can be different"""
    document_list = []
    document_tfidf_list = []
    label_list = []
    imbd_dic = {"neg":"1","pos":"2"}
    # def filter_key(sent):
    #     unk_count = sent.count(vocab['<unk>'])
    #     return unk_count/len(sent) < 0.3
    # def filter_tfidf(sent):
    #     unk_count = sent.count(0)
    #     return unk_count/len(sent) < 0.3
    with open(textfile, "r") as f:
        line_list = f.readlines()
        line_len = len(line_list)
        # random_index = np.random.permutation(line_len)
        # line_list = [line_list[index] for index in random_index]
        # progressbar = ProgressBar(maxval=len(line_list))
        word_list_buffer = []
        # for line in progressbar(line_list):
        for doc_id,line in enumerate(line_list):
            if dataset=='yelp':
                _,_,label,text = line.strip().split("\t\t") #2 pos, 1neg
                sents = text.strip().split("<sssss>")
            elif dataset=='imdb':
                label = imbd_dic[json.loads(line)['sentiment']]
                text = json.loads(line)['text']
                sents = [x for i in text.strip().split('.') for x in i.split('<br /><br />')]
            elif dataset=='guardian_news':
                text,label = line.strip().split("\t")
                label = int(label)+1
                sents = text.strip().split(".")  
            sent_list = [sent.strip().split(" ") for sent in sents]
            sent_list = list(filter(lambda x:x!=["."],sent_list))

            tokenids = []
            for sent in sent_list:
                sen_sample = convert_words2ids(sent,vocab,max_value=max_value,unk=vocab['<unk>'],use_stem=False,eos=vocab['<eos>'])
                if not sen_sample is None:
                    tokenids.append(sen_sample)
            if len(tokenids) == 0:
                continue
            else:
                tfidf_list = convert_words2tfidf(tokenids, vocab, unk=None, max_value=max_value, sos=None, eos=None, doc_id=doc_id)
                
                # sent_id_list = list(filter(filter_key,sent_id_list)) #
                # tfidf_list = list(filter(filter_tfidf,tfidf_list))

                sent_id_list = []
                sent_tfidf_list = []

                #post-processing
                for sample,tfidf_sen in zip(tokenids,tfidf_list):
                    sent_id_list.append(sample.id)
                    sent_tfidf_list.append(tfidf_sen)
                    assert len(sample.id) == len(tfidf_sen)

                document_list.append(sent_id_list)
                document_tfidf_list.append(sent_tfidf_list)
                label_list.append(int(label))

    def sort_key(document_with_label):
        document = document_with_label[0]
        first_key = len(document)  # The first key is the number of utterance of input
        second_key = np.max([len(utterance) for utterance in document]) # The third key is the max number of word in input
        third_key = np.mean([len(utterance) for utterance in document]) # The third key is the max number of word in input
        return first_key,second_key,third_key

    document_with_label_list = list(zip(*[document_list,label_list,document_tfidf_list]))
    document_with_label_list = sorted(document_with_label_list,key=sort_key)
    document_list,label_list,document_tfidf_list = list(zip(*document_with_label_list))

    #save processed data to disk
    # for data in ["data","tfidf","label"]:
    filename = '/home/hanqiyan/latent_topic/SINE/input_data/{}/{}_data_list.pkl'.format(dataset,type)
    with open(filename, 'wb') as f:
        pickle.dump(document_list, f)
    filename = '/home/hanqiyan/latent_topic/SINE/input_data/{}/{}_tfidf_list.pkl'.format(dataset,type)
    with open(filename, 'wb') as f:
        pickle.dump(document_tfidf_list, f)
    filename = '/home/hanqiyan/latent_topic/SINE/input_data/{}/{}_label_list.pkl'.format(dataset,type)
    with open(filename, 'wb') as f:
        pickle.dump(label_list, f)

    return document_list,label_list,document_tfidf_list