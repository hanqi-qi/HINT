import logging
import argparse
import copy
import numpy as np
import torch
import random
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
from pathlib import Path
from gensim.models import KeyedVectors
import dill
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords


def load_process(dataset,type,data):
    filename = filename = './input_data/{}/{}_{}_list.pkl'.format(dataset,type,data)
    with open(filename, 'rb') as f:
        result = pickle.load(f)
    return result


def sort_key(document_with_label):
    document = document_with_label[0]
    first_key = len(document)  # The first key is the number of utterance of input
    second_key = np.max([len(utterance) for utterance in document]) # The third key is the max number of word in input
    third_key = np.mean([len(utterance) for utterance in document]) # The third key is the max number of word in input
    return first_key,second_key,third_key

class input_example:
    def __init__(self):
        self.pos_idx = None
        self.word = None
        self.voc_idx = None
    def feed(self,id,word,vid):
        self.id = id
        self.word = word
        self.vid = vid

class old_input_example:
    def __init__(self):
        self.pos_idx = None
        self.word = None
        # self.voc_idx = None
    def feed(self,id,word):
        self.id = id
        self.word = word
        # self.vid = vid
        
def convert_words2ids_old(words, vocab, unk, max_value, sos=None, eos=None,use_stem=False,stopwords=None):
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
    # resultx = []
    for item in words:

        w = item
        w = w.strip('\'"?!,.():;')
        w = w.lower()
        if use_stem:
            porter_stemmer = PorterStemmer()
            w = str(porter_stemmer.stem(w))
        if w in vocab and len(w) > 0:
            id_list.append(vocab[w])
            word_list.append(w)

    #only return the sentence longer than 3 words


    if len(id_list) > 0:
        result = old_input_example()
        result.feed(id_list,word_list)
        return result

def convert_words2ids(words, vocab, unk, max_value, sos=None, eos=None,use_stem=False,stopwords=None):
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
    resultx = []
    for item in words:
        if isinstance(item,input_example):
            w = item.word
        else:
            w = item
        w = w.strip('\'"?!,.():;')
        w = w.lower()
        if use_stem:
            porter_stemmer = PorterStemmer()
            w = str(porter_stemmer.stem(w))
        if w in vocab and len(w) > 0 and w not in stopwords:
            if isinstance(item,input_example):
                result = input_example()
                result.word = w
                result.pos_idx = item.pos_idx
                result.voc_idx = vocab[w]

                resultx.append(result)
            else:
                id_list.append(vocab[w])
                word_list.append(w)

    #only return the sentence longer than 3 words
    if isinstance(item,input_example):
        if len(resultx) > 1:
            return resultx
    else:
        if len(id_list) > 1:
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
        word2vec = KeyedVectors.load_word2vec_format('/mnt/sda/media/Data2/hanqi/sine/GoogleNews-vectors-negative300.bin', binary=True)
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

def get_vocabulary(textfile, initial_vocab={'<unk>':0,'<sssss>':1,'<eos>':2}, vocabsize=0,dataset='yelp',use_stem=False,stopwords=None):
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
        elif dataset == "eraser_movie":
            filename = json.loads(line)['annotation_id']
            text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/docs/"+filename,"r").readlines()#one doc
            text = " ".join(text)
        for w in text.split(): # skip speaker indicator
            w = w.strip('\'"?!,.():;')
            w = w.lower()
            if use_stem:
                porter_stemmer = PorterStemmer()
                w = str(porter_stemmer.stem(w))
            if not w in stopwords:
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
    logging.info("\nThe hit rate is :{}".format(hit_rate))
    # print(("The hit rate is {}".format(hit_rate)))
    return embedding_matrix

def convert_words2tfidf_old(input_example, vocab, unk, max_value, sos=None, eos=None,doc_id=None,doc_tfidf=None):
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
        # for word_sample in sample:
        for word in sample.word:
            idx = feature_names.index(word)
            #some word tfidf is zeros, if so, make it non-zero by adding 1e-4
            if X[sen_id,idx] == 0.0:
                sen_tfidf.append(1e-4)
            else:
                sen_tfidf.append(X[sen_id,idx])
        tfidf_list.append(sen_tfidf)
    return tfidf_list

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
        #need to process the class
        sent = []
        for word_sample in sample:
            sent.append(word_sample.word)
            words.append(word_sample.word)
        docs.append(" ".join(sent))
        # else:
            # docs.append(" ".join(sample.word))

        # words = words + [word for word in sample.word]

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
        for word_sample in sample:
        # for word in sample.word:
            idx = feature_names.index(word_sample.word)
            #some word tfidf is zeros, if so, make it non-zero by adding 1e-4
            if X[sen_id,idx] == 0.0:
                sen_tfidf.append(1e-4)
            else:
                sen_tfidf.append(X[sen_id,idx])
        tfidf_list.append(sen_tfidf)
    return tfidf_list

# class B:
#     text = torch.zeros(1).to(args.device)
#     label = torch.zeros(1).to(args.device)
    
# def batch_from_list(textlist, labellist):
#     batch = B()
#     batch.text = textlist[0]
#     batch.label = labellist[0]
#     for txt, la in zip(textlist[1:], labellist[1:]):
#         batch.text = torch.cat((batch.text, txt), 0)
#         # you may need to change the type of "la" to torch.tensor for different datasets, sorry for the inconvenience
#         batch.label = torch.cat((batch.label, la), 0) # for SST and IMDB dataset, you do not need to change "la" type
#     batch.text = batch.text.to("cuda:0")
#     batch.label = batch.label.to("cuda:0")
#     return batch
def load_bin_dataset(metric,num_bin,vocab,max_value,max_utterance,stopwords,model,dataset,variants):
    document_list = []
    document_tfidf_list = []
    label_file = open("/home/hanq1yanwarwick/SINE/input_data/{}/variants/{}/original_set/label.txt".format(dataset,variants)).readlines()
    label_list=[]
    idx_list =  []
    if metric ==  "original":
        file_dir = "/home/hanq1yanwarwick/SINE/input_data/{}/variants/{}/original_set/".format(dataset,variants)
        target_dir = file_dir
    else:
        file_dir = "/home/hanq1yanwarwick/SINE/input_data/{}/variants/{}/{}/".format(dataset,variants,metric)
        bin_dir = "bin"+str(num_bin)
        target_dir = file_dir+"/"+bin_dir+"/" 

    for file_num in range(1,200):
        text = open(target_dir+str(file_num)+".txt","r").readlines()
        idx = file_num
        label = int(label_file[file_num-1])+1
        sent_list = [sent.strip().split(" ") for sent in text]
        sent_list = list(filter(lambda x:x!=["."],sent_list))
        
        tokenids = []
        for sent in sent_list:
            sen_sample = convert_words2ids_old(sent,vocab,max_value=max_value,unk=vocab['<unk>'],use_stem=False,eos=vocab['<eos>'],stopwords=stopwords)
            if not sen_sample is None:
                tokenids.append(sen_sample)
        if len(tokenids) == 0:
            continue
        else:
            tfidf_list = convert_words2tfidf_old(tokenids, vocab, unk=None, max_value=max_value, sos=None, eos=None, doc_id=file_num)
            
            # sent_id_list = list(filter(filter_key,sent_id_list)) #
            # tfidf_list = list(filter(filter_tfidf,tfidf_list))

            sent_id_list = []
            sent_tfidf_list = []

            #post-processing
            for sample,tfidf_sen in zip(tokenids,tfidf_list):
                sent_id_list.append(sample.id)
                sent_tfidf_list.append(tfidf_sen)
                assert len(sample.id) == len(tfidf_sen)

            document_list.append(sent_id_list[:max_utterance])
            document_tfidf_list.append(sent_tfidf_list[:max_utterance])
            label_list.append(int(label))
            idx_list.append(int(idx))


    document_with_label_list = list(zip(*[document_list,label_list,document_tfidf_list,idx_list]))
    document_with_label_list = sorted(document_with_label_list,key=sort_key)
    document_list,label_list,document_tfidf_list,idx_list = list(zip(*document_with_label_list))

    return  document_list,label_list,document_tfidf_list,idx_list,document_list

def load(textfile, vocab, max_value, max_utterance,dataset,type=None,stopwords = None):
    """Only padding the input sentence, sentence number in each document can be different"""
    document_list = []
    document_tfidf_list = []
    label_list = []
    idx_list = []
    imbd_dic = {"neg":"1","pos":"2"}
    eraser_dict  =  {"NEG":"1","POS":"2"}
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
                #yhq replace "\t" to <h>
                _,_,label,text = line.strip().split("<h>") #2 pos, 1neg
                sents = text.strip().split("<sssss>")
            elif dataset=='imdb':
                label = imbd_dic[json.loads(line)['sentiment']]
                text = json.loads(line)['text']
                idx = doc_id
                #use input_example
                # doc_list =  []
                # for sen_id in range(len(text).split('<br /><br />')):
                #     sen_list = []
                #     for idx, wid in enumerate(text[sen_id].split()):
                #         resultx = input_example()
                #         resultx.word = wid
                #         resultx.pos_idx = sen_len + idx
                #         sen_list.append(resultx)
                #     sen_len = sen_len+len(text[sen_id].split())
                #     doc_list.append(sen_list)
                sents = [x for i in text.strip().split('.') for x in i.split('<br /><br />')]
            elif dataset=='guardian_news':
                text,label = line.strip().split("\t")
                label = int(label)+1
                sents = text.strip().split(".")
            elif dataset=="eraser_movie":
                label = eraser_dict[json.loads(line)['classification']] 
                filename = json.loads(line)['annotation_id']
                idx = filename
                # if type == "rational_free":
                #     path = Path("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/rational_free_docs/"+filename)
                #     if path.is_file():
                #         text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/rational_free_docs/"+filename,"r").readlines()
                #         sents = text
                #     else:
                #         text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/docs/"+filename,"r").readlines()
                #         sents = [x.strip() for x in text if len(x.split())>1] 
                # elif type== "original":
                text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/docs/"+filename,"r").readlines()
                sents = [x.strip() for x in text if len(x.split())>1] 
            # sent_list = [sent.strip().split(" ") for sent in sents]
            # sent_list = list(filter(lambda x:x!=["."],sent_list))

            tokenids = []
            for sent in sents:
                sen_sample = convert_words2ids_old(sent.split(),vocab,max_value=max_value,unk=vocab['<unk>'],use_stem=False,eos=vocab['<eos>'],stopwords=stopwords)
                if not sen_sample is None:
                    tokenids.append(sen_sample)
            if len(tokenids) == 0:
                continue
            else:
                tfidf_list = convert_words2tfidf_old(tokenids, vocab, unk=None, max_value=max_value, sos=None, eos=None, doc_id=doc_id)
                
                # sent_id_list = list(filter(filter_key,sent_id_list)) #
                # tfidf_list = list(filter(filter_tfidf,tfidf_list))

                sent_id_list = []
                sent_tfidf_list = []

                #post-processing
                for sample,tfidf_sen in zip(tokenids,tfidf_list):
                    sent_id_list.append(sample.id)
                    sent_tfidf_list.append(tfidf_sen)
                    assert len(sample.id) == len(tfidf_sen)

                document_list.append(sent_id_list[:max_utterance])
                document_tfidf_list.append(sent_tfidf_list[:max_utterance])
                label_list.append(int(label))
                idx_list.append(idx)

    def sort_key(document_with_label):
        document = document_with_label[0]
        first_key = len(document)  # The first key is the number of utterance of input
        second_key = np.max([len(utterance) for utterance in document]) # The third key is the max number of word in input
        third_key = np.mean([len(utterance) for utterance in document]) # The third key is the max number of word in input
        return first_key,second_key,third_key

    document_with_label_list = list(zip(*[document_list,label_list,document_tfidf_list,idx_list]))
    document_with_label_list = sorted(document_with_label_list,key=sort_key)
    document_list,label_list,document_tfidf_list,idx_list = list(zip(*document_with_label_list))

    #save processed data to disk
    # for data in ["data","tfidf","label"]:
    # filename = '/home/hanqiyan/latent_topic/SINE/input_data/{}/{}_data_list.pkl'.format(dataset,type)
    # with open(filename, 'wb') as f:
    #     pickle.dump(document_list, f)
    # filename = '/home/hanqiyan/latent_topic/SINE/input_data/{}/{}_tfidf_list.pkl'.format(dataset,type)
    # with open(filename, 'wb') as f:
    #     pickle.dump(document_tfidf_list, f)
    # filename = '/home/hanqiyan/latent_topic/SINE/input_data/{}/{}_label_list.pkl'.format(dataset,type)
    # with open(filename, 'wb') as f:
    #     pickle.dump(label_list, f)

    return document_list,label_list,document_tfidf_list,idx_list,document_list

def load_rats(rat_file,vocab,max_value=60,max_utterance=50,dataset=None,type="original",stopwords=None):    
    eraser_dict  =  {"NEG":"1","POS":"2"}
    document_list = []
    document_tfidf_list = []
    label_list = []
    doc_idx_list = []
    word_pos_list = []
    # label_file = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/SINE/original_set/label.txt").readlines()
    # label_list=[]
    with open(rat_file, "r") as f:
        line_list = f.readlines()
        # docs_wRational = []
        for doc_id,line in enumerate(line_list):
            #  label = int(label_file[file_num-1])+1
            label = eraser_dict[json.loads(line)['classification']] 
            filename = json.loads(line)['annotation_id']
            # idx = filenasme
            text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/docs/"+filename,"r").readlines()
            sen_len = 0
            #couple the idx and word
            doc_list =  []
            for sen_id in range(len(text)):
                sen_list = []
                for idx, wid in enumerate(text[sen_id].split()):
                    resultx = input_example()
                    resultx.word = wid
                    resultx.pos_idx = sen_len + idx
                    sen_list.append(resultx)
                sen_len = sen_len+len(text[sen_id].split())
                doc_list.append(sen_list)

            tokenids = []
            for sent in doc_list:
                sen_sample = convert_words2ids(sent,vocab,max_value=max_value,unk=vocab['<unk>'],use_stem=False,eos=vocab['<eos>'],stopwords=stopwords)
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
                sent_wordpos_list = []

                #post-processing
                for sample,tfidf_sen in zip(tokenids,tfidf_list):
                    sent_id_list_tmp,sent_wordpos_list_tmp = [],[]
                    for word_sample in sample:
                        sent_id_list_tmp.append(word_sample.voc_idx)
                        sent_wordpos_list_tmp.append(word_sample.pos_idx)
                    assert len(sent_id_list_tmp) == len(sent_wordpos_list_tmp)
                    sent_tfidf_list.append(tfidf_sen)
                    sent_id_list.append(sent_id_list_tmp)
                    sent_wordpos_list.append(sent_wordpos_list_tmp)
                    assert len(sent_tfidf_list) == len(sent_id_list) == len(sent_wordpos_list)

                document_list.append(sent_id_list[:max_utterance])
                document_tfidf_list.append(sent_tfidf_list[:max_utterance])
                label_list.append(int(label))
                doc_idx_list.append(filename)
                word_pos_list.append(sent_wordpos_list[:max_utterance])

        document_with_label_list = list(zip(*[document_list,label_list,document_tfidf_list,doc_idx_list,word_pos_list]))
        document_with_label_list = sorted(document_with_label_list,key=sort_key)
        document_list,label_list,document_tfidf_list,doc_idx_list,word_pos_list = list(zip(*document_with_label_list))

    return document_list,label_list,document_tfidf_list,doc_idx_list,word_pos_list


def load_general_dataset(textfile, vocab, max_value, max_utterance,dataset,type=None,stopwords = None):
    """Only padding the input sentence, sentence number in each document can be different"""
    document_list = []
    document_tfidf_list = []
    label_list = []
    idx_list = []
    doc_idx_list = []
    word_pos_list = []
    imbd_dic = {"neg":"1","pos":"2"}
    eraser_dict  =  {"NEG":"1","POS":"2"}
    # def filter_key(sent):
    #     unk_count = sent.count(vocab['<unk>'])
    #     return unk_count/len(sent) < 0.3
    # def filter_tfidf(sent):
    #     unk_count = sent.count(0)
    #     return unk_count/len(sent) < 0.3
    with open(textfile, "r") as f:
        line_list = f.readlines()[:3000]
        #random select 200 samples
        random.shuffle(line_list)
        line_len = len(line_list)
        # random_index = np.random.permutation(line_len)
        # line_list = [line_list[index] for index in random_index]
        # progressbar = ProgressBar(maxval=len(line_list))
        word_list_buffer = []
        # for line in progressbar(line_list):
        for doc_id,line in enumerate(line_list):
            if dataset=='yelp':
                filename = doc_id
                #yhq replace "\t" to <h>
                label = line.strip().split("\t")[-3]
                text = line.strip().split("\t")[-1] #2 pos, 1neg
                sents_list = text.strip().split("<sssss>")
                sen_len =  0
                doc_list =  []
                for sen_id in range(len(sents_list)):
                    sent = " ".join(sents_list[sen_id].strip().split('<br /><br />')).strip()
                    sen_list = []
                    for idx, wid in enumerate(sent.split()):
                        resultx = input_example()
                        resultx.word = wid
                        resultx.pos_idx = sen_len + idx
                        sen_list.append(resultx)
                    sen_len = sen_len+len(sent.split())
                    doc_list.append(sen_list)
            elif dataset=='imdb':
                label = imbd_dic[json.loads(line)['sentiment']]
                text = json.loads(line)['text']
                filename = doc_id
                #use input_example
                doc_list =  []
                sents_list = text.strip().split(".")
                sen_len =  0
                for sen_id in range(len(sents_list)):
                    sent = " ".join(sents_list[sen_id].strip().split('<br /><br />')).strip()
                    sen_list = []
                    for idx, wid in enumerate(sent.split()):
                        resultx = input_example()
                        resultx.word = wid
                        resultx.pos_idx = sen_len + idx
                        sen_list.append(resultx)
                    sen_len = sen_len+len(sent.split())
                    doc_list.append(sen_list)
                # sents = [x for i in text.strip().split('.') for x in i.split('<br /><br />')]
            elif dataset=='guardian_news':
                text,label = line.strip().split("\t")
                label = int(label)+1
                sents_list = text.strip().split(".")
                sen_len =  0
                filename = doc_id
                doc_list =  []
                for sen_id in range(len(sents_list)):
                    sent = " ".join(sents_list[sen_id].strip().split('<br /><br />')).strip()
                    sen_list = []
                    for idx, wid in enumerate(sent.split()):
                        resultx = input_example()
                        resultx.word = wid
                        resultx.pos_idx = sen_len + idx
                        sen_list.append(resultx)
                    sen_len = sen_len+len(sent.split())
                    doc_list.append(sen_list)
            elif dataset=="eraser_movie":
                label = eraser_dict[json.loads(line)['classification']] 
                filename = json.loads(line)['annotation_id']
                idx = filename
                if type == "rational_free":
                    path = Path("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/rational_free_docs/"+filename)
                    if path.is_file():
                        text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/rational_free_docs/"+filename,"r").readlines()
                        sents = text
                    else:
                        text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/docs/"+filename,"r").readlines()
                        sents = [x.strip() for x in text if len(x.split())>1] 
                elif type== "original":
                    text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/docs/"+filename,"r").readlines()
                    sents = [x.strip() for x in text if len(x.split())>1] 
            # sent_list = [sent.strip().split(" ") for sent in sents]
            # sent_list = list(filter(lambda x:x!=["."],sent_list))

            tokenids = []
            for sent in doc_list:
                if len(sent)>0:
                    sen_sample = convert_words2ids(sent,vocab,max_value=max_value,unk=vocab['<unk>'],use_stem=False,eos=vocab['<eos>'],stopwords=stopwords)
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
                sent_wordpos_list = []

                #post-processing
                for sample,tfidf_sen in zip(tokenids,tfidf_list):
                    sent_id_list_tmp,sent_wordpos_list_tmp = [],[]
                    for word_sample in sample:
                        sent_id_list_tmp.append(word_sample.voc_idx)
                        sent_wordpos_list_tmp.append(word_sample.pos_idx)
                    assert len(sent_id_list_tmp) == len(sent_wordpos_list_tmp)
                    sent_tfidf_list.append(tfidf_sen)
                    sent_id_list.append(sent_id_list_tmp)
                    sent_wordpos_list.append(sent_wordpos_list_tmp)
                    assert len(sent_tfidf_list) == len(sent_id_list) == len(sent_wordpos_list)

                document_list.append(sent_id_list[:max_utterance])
                document_tfidf_list.append(sent_tfidf_list[:max_utterance])
                label_list.append(int(label))
                doc_idx_list.append(filename)
                word_pos_list.append(sent_wordpos_list[:max_utterance])

        document_with_label_list = list(zip(*[document_list,label_list,document_tfidf_list,doc_idx_list,word_pos_list]))
        document_with_label_list = sorted(document_with_label_list,key=sort_key)
        document_list,label_list,document_tfidf_list,doc_idx_list,word_pos_list = list(zip(*document_with_label_list))

    return document_list,label_list,document_tfidf_list,doc_idx_list,word_pos_list

def load_general_dataset_explain(textfile, vocab, max_value, max_utterance,dataset,type=None,stopwords = None):
    """Only padding the input sentence, sentence number in each document can be different"""
    document_list = []
    document_tfidf_list = []
    label_list = []
    idx_list = []
    doc_idx_list = []
    word_pos_list = []
    imbd_dic = {"neg":"1","pos":"2"}
    eraser_dict  =  {"NEG":"1","POS":"2"}
    # def filter_key(sent):
    #     unk_count = sent.count(vocab['<unk>'])
    #     return unk_count/len(sent) < 0.3
    # def filter_tfidf(sent):
    #     unk_count = sent.count(0)
    #     return unk_count/len(sent) < 0.3
    with open(textfile, "r") as f:
        line_list = f.readlines()
        #random select 200 samples
        random.shuffle(line_list)
        line_len = len(line_list)
        # random_index = np.random.permutation(line_len)
        # line_list = [line_list[index] for index in random_index]
        # progressbar = ProgressBar(maxval=len(line_list))
        word_list_buffer = []
        # for line in progressbar(line_list):
        for doc_id,line in enumerate(line_list[:500]):
            if dataset=='yelp':
                filename = doc_id
                #yhq replace "\t" to <h>
                label = line.strip().split("\t")[-3]
                text = line.strip().split("\t")[-1] #2 pos, 1neg
                sents_list = text.strip().split("<sssss>")
                sen_len =  0
                doc_list =  []
                for sen_id in range(len(sents_list)):
                    sent = " ".join(sents_list[sen_id].strip().split('<br /><br />')).strip()
                    sen_list = []
                    for idx, wid in enumerate(sent.split()):
                        resultx = input_example()
                        resultx.word = wid
                        resultx.pos_idx = sen_len + idx
                        sen_list.append(resultx)
                    sen_len = sen_len+len(sent.split())
                    doc_list.append(sen_list)
            elif dataset=='imdb':
                label = imbd_dic[json.loads(line)['sentiment']]
                text = json.loads(line)['text']
                filename = doc_id
                #use input_example
                doc_list =  []
                sents_list = text.strip().split(".")
                sen_len =  0
                for sen_id in range(len(sents_list)):
                    sent = " ".join(sents_list[sen_id].strip().split('<br /><br />')).strip()
                    sen_list = []
                    for idx, wid in enumerate(sent.split()):
                        resultx = input_example()
                        resultx.word = wid
                        resultx.pos_idx = sen_len + idx
                        sen_list.append(resultx)
                    sen_len = sen_len+len(sent.split())
                    doc_list.append(sen_list)
                # sents = [x for i in text.strip().split('.') for x in i.split('<br /><br />')]
            elif dataset=='guardian_news':
                text,label = line.strip().split("\t")
                label = int(label)+1
                sents_list = text.strip().split(".")
                sen_len =  0
                filename = doc_id
                doc_list =  []
                for sen_id in range(len(sents_list)):
                    sent = " ".join(sents_list[sen_id].strip().split('<br /><br />')).strip()
                    sen_list = []
                    for idx, wid in enumerate(sent.split()):
                        resultx = input_example()
                        resultx.word = wid
                        resultx.pos_idx = sen_len + idx
                        sen_list.append(resultx)
                    sen_len = sen_len+len(sent.split())
                    doc_list.append(sen_list)
            elif dataset=="eraser_movie":
                label = eraser_dict[json.loads(line)['classification']] 
                filename = json.loads(line)['annotation_id']
                idx = filename
                if type == "rational_free":
                    path = Path("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/rational_free_docs/"+filename)
                    if path.is_file():
                        text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/rational_free_docs/"+filename,"r").readlines()
                        sents = text
                    else:
                        text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/docs/"+filename,"r").readlines()
                        sents = [x.strip() for x in text if len(x.split())>1] 
                elif type== "original":
                    text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/docs/"+filename,"r").readlines()
                    sents = [x.strip() for x in text if len(x.split())>1] 
            # sent_list = [sent.strip().split(" ") for sent in sents]
            # sent_list = list(filter(lambda x:x!=["."],sent_list))

            tokenids = []
            for sent in doc_list:
                if len(sent)>0:
                    sen_sample = convert_words2ids(sent,vocab,max_value=max_value,unk=vocab['<unk>'],use_stem=False,eos=vocab['<eos>'],stopwords=stopwords)
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
                sent_wordpos_list = []

                #post-processing
                for sample,tfidf_sen in zip(tokenids,tfidf_list):
                    sent_id_list_tmp,sent_wordpos_list_tmp = [],[]
                    for word_sample in sample:
                        sent_id_list_tmp.append(word_sample.voc_idx)
                        sent_wordpos_list_tmp.append(word_sample.pos_idx)
                    assert len(sent_id_list_tmp) == len(sent_wordpos_list_tmp)
                    sent_tfidf_list.append(tfidf_sen)
                    sent_id_list.append(sent_id_list_tmp)
                    sent_wordpos_list.append(sent_wordpos_list_tmp)
                    assert len(sent_tfidf_list) == len(sent_id_list) == len(sent_wordpos_list)

                document_list.append(sent_id_list[:max_utterance])
                document_tfidf_list.append(sent_tfidf_list[:max_utterance])
                label_list.append(int(label))
                doc_idx_list.append(filename)
                word_pos_list.append(sent_wordpos_list[:max_utterance])

        document_with_label_list = list(zip(*[document_list,label_list,document_tfidf_list,doc_idx_list,word_pos_list]))
        document_with_label_list = sorted(document_with_label_list,key=sort_key)
        document_list,label_list,document_tfidf_list,doc_idx_list,word_pos_list = list(zip(*document_with_label_list))

    return document_list,label_list,document_tfidf_list,doc_idx_list,word_pos_list

