import logging
import argparse
import copy
import numpy as np
import torch
from torch import nn
from torch import optim
from collections import Counter
from itertools import chain
from progressbar import ProgressBar
#from ipdb import launch_ipdb_on_exception
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib
import json
from gensim.models import KeyedVectors
import dill
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import ParameterGrid


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
    for w in words:
        w = w.strip('\'"?!,.():;')
        w = w.lower()
        if use_stem:
            porter_stemmer = PorterStemmer()
            w = str(porter_stemmer.stem(w))
        if w in vocab:
            id_list.append(vocab[w])
        else:
            id_list.append(unk)

    if sos is not None:
        id_list.insert(0, sos)
    if eos is not None:
        id_list.append(eos)
    return id_list[:max_value]

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

def get_vocabulary(textfile, initial_vocab={'<unk>':0,'<sssss>':1}, vocabsize=0,dataset='yelp',use_stem=False):
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
        elif dataset =='news':
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

def convert_words2tfidf(words, vocab, unk, max_value, sos=None, eos=None,doc_id=None,doc_tfidf=None):
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
    tfidf_list = []
    for w in words:
        w = w.strip('\'"?!,.():;')
        w = w.lower()
        if w in vocab:#add the tf-idf list, the same length as id_list
            # id_list.append(vocab[w])
            if w in doc_tfidf[doc_id]:
                tfidf_list.append(doc_tfidf[doc_id][w])
            else:
                tfidf_list.append(1e-4)
        else:
            tfidf_list.append(0) #
    if sos is not None:
        tfidf_list.insert(1e-4, sos)
    if eos is not None:
        tfidf_list.append(1e-4)
    #normalized tfidf_list
    arr = np.array(tfidf_list)
    norm_arr=list(arr/(arr.sum(axis=0)+1e-5)) #
    return tfidf_list[:max_value]

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

def load(textfile, vocab, max_value, max_utterance,dataset,doc_tfidf=None):
    """ Load a dialog text corpus as word Id sequences
        Args:
            textfile (str): filename of a dialog corpus
            vocab (dict): word-id mapping
        Return:
            list of dialogue : dialogue is (input_id_list,output_id_list)
    """
    document_list = []
    document_tfidf_list = []
    label_list = []
    imbd_dic = {"neg":"1","pos":"2"}
    def filter_key(sent):
        unk_count = sent.count(vocab['<unk>'])
        return unk_count/len(sent) < 0.3
    def filter_tfidf(sent):
        unk_count = sent.count(0)
        return unk_count/len(sent) < 0.3
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
            sent_list = [sent.strip().split(" ") for sent in sents]
            sent_list = list(filter(lambda x:x!=["."],sent_list))
            sent_id_list = [convert_words2ids(sent,vocab,max_value=max_value,unk=vocab['<unk>'],use_stem=False) for sent in sent_list]

            tfidf_list = [convert_words2tfidf(sent, vocab, unk=None, max_value=max_value, sos=None, eos=None, doc_id=doc_id,doc_tfidf=doc_tfidf) for sent in sent_list]
            
            sent_id_list = list(filter(filter_key,sent_id_list)) #
            tfidf_list = list(filter(filter_tfidf,tfidf_list))
            new_sent_id_list = []
            new_tfidf_list = []

            previous_sent = []
            previous_tfidf_sent = []
            for sent,tfidf_sent in zip(sent_id_list,tfidf_list):
                if len(previous_sent) != 0:
                    new_sent = previous_sent + sent
                    new_tfidf_sent = previous_tfidf_sent +tfidf_sent
                else:
                    new_sent = sent
                    new_tfidf_sent = tfidf_sent
                if len(new_sent) < 3:
                    previous_sent = new_sent
                    previous_tfidf_sent = new_tfidf_sent
                else:
                    new_sent_id_list.append(new_sent)
                    new_tfidf_list.append(new_tfidf_sent)
                    previous_sent = []
                    previous_tfidf_sent = []
            if len(previous_sent) > 0:
                new_sent_id_list.append(previous_sent)
                new_tfidf_list.append(previous_tfidf_sent)
            if len(new_sent_id_list) > 0: 
                document_list.append(new_sent_id_list[:max_utterance])
                document_tfidf_list.append(new_tfidf_list[:max_utterance])
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
    return document_list,label_list,document_tfidf_list