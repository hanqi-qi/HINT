# CNN Attentional Document Classification
# Author : Jiachen DU
# Last Modified : 9th, May. 2019
# Copyright Reserved, 2018, 2019


import logging
import copy
import numpy as np
import torch
from torch import nn
from torch import optim
from collections import Counter
from itertools import chain
from progressbar import ProgressBar
from matplotlib import pyplot as plt
import pickle
import json
import os
import random
import re
import pandas as pd
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import sys
sys.path.insert(0,"/home/hanq1yanwarwick/SINE/")

from funcs import load_rats,get_vocabulary,load,load_bin_dataset
from dataloader import DataIter
#from ipdb import launch_ipdb_on_exception
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
latex_special_token = ["!@#$%^&*()"]


# def sort_key(document_with_label):
#     document = document_with_label[0]
#     first_key = len(document)  # The first key is the number of utterance of input
#     second_key = np.max([len(utterance) for utterance in document]) # The third key is the max number of word in input
#     third_key = np.mean([len(utterance) for utterance in document]) # The third key is the max number of word in input
#     return first_key,second_key,third_key

# class input_example:
#     def __init__(self):
#         self.pos_idx = None
#         self.word = None
#         self.voc_idx = None


# class old_input_example:
#     def __init__(self):
#         self.pos_idx = None
#         self.word = None
#         # self.voc_idx = None
#     def feed(self,id,word):
#         self.id = id
#         self.word = word

# def convert_words2ids_old(words, vocab, unk, max_value, sos=None, eos=None,use_stem=False,stopwords=None):
#     """ convert string sequence into word id sequence
#         Args:
#             words (list): word sequence
#             vocab (dict): word-id mapping
#             unk (int): id of unkown word "<unk>"
#             sos (int): id of start-of-sentence symbol "<sos>"
#             eos (int): id of end-of-sentence symbol "eos"
#         Returns:
#             numpy array of word ids sequence
#     """
#     # id_list = [ vocab[w] if w in vocab else unk for w in words ] #ori_code
#     #make the input date more clear and formal
#     id_list = []
#     word_list = []
#     # resultx = []
#     for item in words:

#         w = item
#         w = w.strip('\'"?!,.():;')
#         w = w.lower()
#         if use_stem:
#             porter_stemmer = PorterStemmer()
#             w = str(porter_stemmer.stem(w))
#         if w in vocab and len(w) > 0:
#             id_list.append(vocab[w])
#             word_list.append(w)
#     if len(id_list) > 0:
#         result = old_input_example()
#         result.feed(id_list,word_list)
#         return result

def gen_hard_rats(doc,doc_word_att,level,doc_wpos_id):
    """write out the test_decode.json as original ERASER project:
    1) use level to decide how many percentage important words will be selected as hard rats.
    2) len(doc) == len(doc_att) == len(truth_rat) truth_rat is the same length as doc, element is 1/0 represents rats or not"""
    rat_list = []
    for sen_id in range(len(doc)):
        assert len(doc[sen_id]) == len(doc_word_att[sen_id]) == len(doc_wpos_id[sen_id])
        deleted_num = max(int(len(doc_word_att[sen_id])*level),1)
        important_idx = list(np.array(doc_word_att[sen_id]).argsort()[-deleted_num:][::-1])
        # threshold = np.percentile(np.array(doc_word_att[sen_id]),100*(1-level))
        for i in range(len(doc_word_att[sen_id])):
            if i in important_idx:
            # if doc_word_att[sen_id][i]>threshold:
                print(doc_wpos_id[sen_id][i],doc[sen_id][i])
                rat_list.append({'start_token': int(doc_wpos_id[sen_id][i]), 'end_token': int(doc_wpos_id[sen_id][i])+1})
    return rat_list


# def load_rats(rat_file,vocab,max_value=60,max_utterance=50,dataset=None,type="original",stopwords=None):    
#     eraser_dict  =  {"NEG":"1","POS":"2"}
#     document_list = []
#     document_tfidf_list = []
#     label_list = []
#     doc_idx_list = []
#     word_pos_list = []
#     with open(rat_file, "r") as f:
#         line_list = f.readlines()
#         # docs_wRational = []
#         for doc_id,line in enumerate(line_list):
#             label = eraser_dict[json.loads(line)['classification']] 
#             filename = json.loads(line)['annotation_id']
#             # idx = filenasme
#             text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/docs/"+filename,"r").readlines()
#             sen_len = 0
#             #couple the idx and word
#             doc_list =  []
#             for sen_id in range(len(text)):
#                 sen_list = []
#                 for idx, wid in enumerate(text[sen_id].split()):
#                     resultx = input_example()
#                     resultx.word = wid
#                     resultx.pos_idx = sen_len + idx
#                     sen_list.append(resultx)
#                 sen_len = sen_len+len(text[sen_id].split())
#                 doc_list.append(sen_list)

#             tokenids = []
#             for sent in doc_list:
#                 sen_sample = convert_words2ids(sent,vocab,max_value=max_value,unk=vocab['<unk>'],use_stem=False,eos=vocab['<eos>'],stopwords=stopwords)
#                 if not sen_sample is None:
#                     tokenids.append(sen_sample)
#             if len(tokenids) == 0:
#                 continue
#             else:
#                 # tfidf_list = convert_words2tfidf(tokenids, vocab, unk=None, max_value=max_value, sos=None, eos=None, doc_id=doc_id)
                
#                 # sent_id_list = list(filter(filter_key,sent_id_list)) #
#                 # tfidf_list = list(filter(filter_tfidf,tfidf_list))

#                 sent_id_list = []
#                 sent_tfidf_list = []
#                 sent_wordpos_list = []

#                 #post-processing
#                 for sample in tokenids:
#                     sent_id_list_tmp,sent_wordpos_list_tmp = [],[]
#                     for word_sample in sample:
#                         sent_id_list_tmp.append(word_sample.voc_idx)
#                         sent_wordpos_list_tmp.append(word_sample.pos_idx)
#                     assert len(sent_id_list_tmp) == len(sent_wordpos_list_tmp)
#                     # sent_tfidf_list.append(tfidf_sen)
#                     sent_id_list.append(sent_id_list_tmp)
#                     sent_wordpos_list.append(sent_wordpos_list_tmp)
#                     assert len(sent_id_list) == len(sent_wordpos_list)

#                 document_list.append(sent_id_list[:max_utterance])
#                 # document_tfidf_list.append(sent_tfidf_list[:max_utterance])
#                 label_list.append(int(label))
#                 doc_idx_list.append(filename)
#                 word_pos_list.append(sent_wordpos_list[:max_utterance])

#         document_with_label_list = list(zip(*[document_list,label_list,doc_idx_list,word_pos_list]))
#         document_with_label_list = sorted(document_with_label_list,key=sort_key)
#         document_list,label_list,doc_idx_list,word_pos_list = list(zip(*document_with_label_list))

#     return document_list,label_list,doc_idx_list,word_pos_list

def label2symbol(label):
    if label >0:
        symbol = "$+$1"
    else:
        symbol = "$-$1"
    return symbol

def generate(doc_sen, doc_word_att, doc_sen_att,latex_file, sen_color='green',word_color='red',pre_label=None,target=None):
    stopwords = set(STOPWORDS)
    stopwords.update("i, my, me, he, his,him, she, her, they")
    assert(len(doc_sen) == len(doc_sen_att))
    doc_id = re.findall(r'\d+',latex_file)
    # if rescale_value:
    #     attention_list = rescale(attention_list)
    #     word_num = len(text_list)
    #     text_list = clean_word(text_list)
    with open(latex_file,'w') as f:
        f.write(r'''\documentclass{article}
\usepackage{tikz,lipsum,lmodern}
\usepackage[most]{tcolorbox}
\usepackage{caption}
\usepackage{subcaption}
\begin{document}
\begin{tcolorbox}[colback=yellow!5!white,colframe=yellow!50!black,colbacktitle=yellow!75!black,''')
        f.write('title=Document '+(doc_id[0]))
        f.write(r''',fonttitle=\bfseries]'''+'\n')
        string = ""
        senatt_arr = [100*i for i in doc_sen_att]
        for sen_id in range(len(doc_sen)):
            string += r"\colorbox{%s!%s}{S"%(sen_color,senatt_arr[sen_id])+str(sen_id)+"} "
            att_arr = rescale(doc_word_att[sen_id])
            # att_arr = [300*i for i in doc_word_att[sen_id]]
            for word_id in range(len(doc_sen[sen_id])):
                if doc_sen[sen_id][word_id] in latex_special_token[0]:
                    word = "\\"+doc_sen[sen_id][word_id].strip()
                else:
                    word = doc_sen[sen_id][word_id].strip()
                if att_arr[word_id]>5:
                    string += r"\colorbox{%s!%s}{"%(word_color, att_arr[word_id])+ word+"} "
                else:
                    string += word+ " "
            string += r"\\"+"\n"
        string += r"\tcblower"+"\n"
        string += r"Predict Label: \textsf{"+str(label2symbol(pre_label))+r"} \\"+"\n"
        string += r"GroundTruth Label: \textsf{"+str(label2symbol(target))+"}"+"\n"
        string += r"\end{tcolorbox}"+"\n"
        f.write(string+'\n')
        f.write("\n"+r'''\end{document}''')


def rescale(input_list):
	the_array = np.asarray(input_list)
	the_max = np.max(the_array)
	the_min = np.min(the_array)
	rescale = (the_array - the_min)/(the_max-the_min)*100
	return rescale.tolist()

# def convert_words2ids(words, vocab, unk, max_value, sos=None, eos=None,use_stem=False,stopwords=None):
#     """ convert string sequence into word id sequence
#         Args:
#             words (list): word sequence
#             vocab (dict): word-id mapping
#             unk (int): id of unkown word "<unk>"
#             sos (int): id of start-of-sentence symbol "<sos>"
#             eos (int): id of end-of-sentence symbol "eos"
#         Returns:
#             numpy array of word ids sequence
#     """
#     # id_list = [ vocab[w] if w in vocab else unk for w in words ] #ori_code
#     #make the input date more clear and formal
#     id_list = []
#     word_list = []
#     resultx = []
#     for item in words:
#         if isinstance(item,input_example):
#             w = item.word
#         else:
#             w = item
#         w = w.strip('\'"?!,.():;')
#         w = w.lower()
#         if use_stem:
#             porter_stemmer = PorterStemmer()
#             w = str(porter_stemmer.stem(w))
#         if w in vocab and len(w) > 0 and w not in stopwords:
#             if isinstance(item,input_example):
#                 result = input_example()
#                 result.word = w
#                 result.pos_idx = item.pos_idx
#                 result.voc_idx = vocab[w]

#                 resultx.append(result)
#             else:
#                 id_list.append(vocab[w])
#                 word_list.append(w)

#     #only return the sentence longer than 3 words
#     if isinstance(item,input_example):
#         if len(resultx) > 0:
#             return resultx
#     else:
#         if len(id_list) > 0:
#             result = input_example()
#             result.feed(id_list,word_list)
#             return result

# def get_vocabulary(textfile, initial_vocab={'<unk>':0,'<sssss>':1,'<eos>':2}, vocabsize=0,dataset='yelp',use_stem=False,stopwords=stopwords):
#     """ acquire vocabulary from dialog text corpus
#         Args:
#             textfile (str): filename of a dialog corpus
#             initial_vocab (dict): initial word-id mapping
#             vocabsize (int): upper bound of vocabulary size (0 means no limitation)
#         Return:
#             dict of word-id mapping
#     """
#     vocab = copy.copy(initial_vocab)
#     word_count = Counter()
#     for line in open(textfile,'r').readlines():
#         if dataset =="yelp":
#             _,_,label,text = line.strip().split("\t\t")
#         elif dataset == 'imdb':
#             # label = imbd_dic[json.loads(line)['sentiment']]
#             text = json.loads(line)['text']
#         elif dataset =='guardian_news':
#             text, label = line.strip().split("\t")
#         elif dataset == "eraser_movie":
#             filename = json.loads(line)['annotation_id']
#             text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/docs/"+filename,"r").readlines()#one doc
#             text = " ".join(text)
#         for w in text.split(): # skip speaker indicator
#             w = w.strip('\'"?!,.():;')
#             w = w.lower()
#             if use_stem:
#                 porter_stemmer = PorterStemmer()
#                 w = str(porter_stemmer.stem(w))
#             if w not in stopwords:
#                 word_count[w] += 1

#     # if vocabulary size is specified, most common words are selected
#     if vocabsize > 0:
#         for w in word_count.most_common(vocabsize):
#             if w[0] not in vocab:
#                 vocab[w[0]] = len(vocab)
#                 if len(vocab) >= vocabsize:
#                     break
#     else: # all observed words are stored
#         for w in word_count:
#             if w not in vocab:
#                 vocab[w] = len(vocab)
#     return vocab


def load_embedding(word_id_dict,embedding_file_name="../glove.840B.300d.txt",embedding_size=300):
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
    print(("The hit rate is {}".format(hit_rate)))
    return embedding_matrix
 

# def load_bin_dataset(metric,num_bin,vocab,max_value,max_utterance,stopwords):
#     document_list = []
#     document_tfidf_list = []
#     label_file = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/HAN/original_set/label.txt").readlines()
#     label_list=[]
#     idx_list =  []
#     if metric ==  "original":
#         file_dir = "/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/HAN/original_set/"
#         target_dir = file_dir
#     else:
#         file_dir =  "/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/HAN/{}".format(metric)
#         bin_dir = "bin"+str(num_bin)
#         target_dir = file_dir+"/"+bin_dir+"/" 

#     def filter_key(sent):
#         unk_count = sent.count(vocab['<unk>'])
#         return unk_count/len(sent) < 0.3
#     for file_num in range(1,200):
#         text = open(target_dir+str(file_num)+".txt","r").readlines()
#         idx = file_num
#         label = int(label_file[file_num-1])+1
#         sent_list = [sent.strip().split(" ") for sent in text]
#         sent_list = list(filter(lambda x:x!=["."],sent_list))
#         tokenids =  []
#         for sent in sent_list:
#             sen_sample = convert_words2ids_old(sent,vocab,unk=vocab['<unk>'],max_value=max_value,eos=vocab['<eos>'])
#             if not sen_sample is None:
#                 tokenids.append(sen_sample.id)
#         if len(tokenids) == 0:
#             continue
#         else:

#             sent_id_list = []
#             # sent_tfidf_list = []

#             #post-processing
#             for sample in tokenids:
#                 sent_id_list.append(sample)

#             document_list.append(sent_id_list[:max_utterance])
#             label_list.append(int(label))
#             idx_list.append(idx)

#     document_with_label_list = list(zip(*[document_list,label_list,idx_list]))
#     document_with_label_list = sorted(document_with_label_list,key=sort_key)
#     document_list,label_list,idx_list = list(zip(*document_with_label_list))

#     return  document_list,label_list,idx_list,document_list

# def load(textfile, vocab, max_value, max_utterance,dataset=None):
#     """ Load a dialog text corpus as word Id sequences
#         Args:
#             textfile (str): filename of a dialog corpus
#             vocab (dict): word-id mapping
#         Return:
#             list of dialogue : dialogue is (input_id_list,output_id_list)
#     """
#     document_list = []
#     label_list = []
#     idx_list = []
#     def filter_key(sent):
#         unk_count = sent.count(vocab['<unk>'])
#         return unk_count/len(sent) < 0.3
#     with open(textfile, "r") as f:
#         line_list = f.readlines()
#         line_len = len(line_list)
#         random_index = np.random.permutation(line_len)
#         line_list = [line_list[index] for index in random_index]
#         progressbar = ProgressBar(maxval=len(line_list))
#         word_list_buffer = []
#         imbd_dic = {"neg":"1","pos":"2"}
#         eraser_dict  =  {"NEG":"1","POS":"2"}
#         for doc_id,line in enumerate(line_list):
#             if dataset == "yelp":
#                 _,_,label,text = line.strip().split("<h>")
#                 sent_list = text.strip().split("<sssss>")
#             elif dataset == "imdb":
#                 label = imbd_dic[json.loads(line)['sentiment']]
#                 text = json.loads(line)['text']
#                 sent_list = [x for i in text.strip().split('.') for x in i.split('<br /><br />')]
#             elif dataset == "eraser_movie":
#                 label = eraser_dict[json.loads(line)['classification']]
#                 filename = json.loads(line)['annotation_id']
#                 idx = doc_id
#                 text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/docs/"+filename,"r").readlines()
#                 sent_list = [x.strip() for x in text if len(x.split())>1] 
            
#             sent_list = [sent.strip().split(" ") for sent in sent_list]
#             sent_list = list(filter(lambda x:x!=["."],sent_list))
#             tokenids =  []
#             for sent in sent_list:
#                 if len(sent)>0:
#                     sen_sample = convert_words2ids_old(sent,vocab,unk=vocab['<unk>'],max_value=max_value,eos=vocab['<eos>'])
#                     if not sen_sample is None:
#                         tokenids.append(sen_sample.id)
#             if len(tokenids) == 0:
#                 continue
#             else:
#                 sent_id_list = []
#                 # sent_tfidf_list = []
#                 #post-processing
#                 for sample in tokenids:
#                     sent_id_list.append(sample)

#                 document_list.append(sent_id_list[:max_utterance])
#                 label_list.append(int(label))
#                 idx_list.append(idx)
#     document_with_label_list = list(zip(*[document_list,label_list,idx_list]))
#     document_with_label_list = sorted(document_with_label_list,key=sort_key)
#     document_list,label_list,idx_list = list(zip(*document_with_label_list))
#     return document_list,label_list,idx_list,document_list

import copy
import numpy as np
from sklearn.feature_extraction.text import _document_frequency


class HierachicalClassifier(nn.Module):
    def __init__(self, num_word, emb_size, word_rnn_size, word_rnn_num_layer, word_rnn_dropout, word_rnn_bidirectional,word_attention_size, 
                context_rnn_size, context_rnn_num_layer, context_rnn_dropout, context_rnn_bidirectional, context_attention_size, mlp_size, num_label, pretrained_embedding=None):
        self.emb_size = emb_size
        self.word_rnn_size = word_rnn_size
        self.word_rnn_num_layer = word_rnn_num_layer
        self.word_rnn_bidirectional = word_rnn_bidirectional
        self.context_rnn_size = context_rnn_size
        self.context_rnn_num_layer = context_rnn_num_layer
        self.context_rnn_bidirectional = context_rnn_bidirectional
        self.num_label = num_label
        super(HierachicalClassifier, self).__init__()
        self.embedding = nn.Embedding(num_word, emb_size)
        self.word_rnn = nn.GRU(input_size = emb_size, hidden_size = word_rnn_size, dropout = word_rnn_dropout,
                num_layers = word_rnn_num_layer, bidirectional = word_rnn_bidirectional)
        word_rnn_output_size = word_rnn_size * 2 if word_rnn_bidirectional else word_rnn_size
        self.word_conv_attention_linear = nn.Linear(word_rnn_output_size, word_rnn_output_size, bias=False)
        self.word_conv_attention_linear2 = nn.Linear(word_rnn_output_size, 1, bias=False)
        self.context_rnn = nn.GRU(input_size = word_rnn_output_size, hidden_size = context_rnn_size, dropout = context_rnn_dropout,
                num_layers = context_rnn_num_layer,bidirectional=context_rnn_bidirectional)
        context_rnn_output_size = context_rnn_size * 2 if context_rnn_bidirectional else context_rnn_size
        self.context_conv_attention_linear = nn.Linear(context_rnn_output_size, 1, bias=False)
        self.classifier = nn.Sequential(nn.Linear(context_rnn_output_size, mlp_size),
                                        nn.LeakyReLU(),
                                        nn.Linear(mlp_size, num_label),
                                        nn.Softmax())
        if pretrained_embedding is not None:
            self.embedding.weight.data = self.embedding.weight.data.new(pretrained_embedding)

    def init_rnn_hidden(self, batch_size, level):
        param_data = next(self.parameters()).data
        if level == "word":
            bidirectional_multipier = 2 if self.word_rnn_bidirectional else 1
            layer_size = self.word_rnn_num_layer * bidirectional_multipier
            word_rnn_init_hidden = param_data.new(layer_size, batch_size, self.word_rnn_size).zero_()
            return word_rnn_init_hidden
        elif level == "context":
            bidirectional_multipier = 2 if self.context_rnn_bidirectional else 1
            layer_size = self.context_rnn_num_layer * bidirectional_multipier
            context_rnn_init_hidden = param_data.new(layer_size, batch_size, self.context_rnn_size).zero_()
            return context_rnn_init_hidden
        else:
            raise Exception("level must be 'word' or 'context'")

    def groups_attwords(self,doc,doc_word_att,doc_id,target):
        """remove important words and create new documents for completeness&sufficient metrics"""
        doc_len = len(doc)
        bins = [0.01,0.05,0.1,0.2,0.5]
        print("Generate Completeness&Sufficient dataset for EraserMovie")
        ofile_dir = "/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/HAN/original_set/"
        if not os.path.isdir(ofile_dir):
            os.mkdir(ofile_dir)
        label_file = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/HAN/original_set/label.txt","a+")
        for k, level in enumerate(bins):
            print("The level is %.4f"%level)
            sfile_dir = "/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/HAN/sset/bin"+str(k)+"/"
            cfile_dir = "/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/HAN/cset/bin"+str(k)+"/"
            rcfile_dir = "/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/HAN/rcset/bin"+str(k)+"/"
            rsfile_dir = "/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/HAN/rsset/bin"+str(k)+"/"
            if not os.path.isdir(sfile_dir):
                print("Save New Files for Intepretability Metrics")
                os.mkdir(sfile_dir)
                os.mkdir(cfile_dir)
                os.mkdir(rcfile_dir)
                os.mkdir(rsfile_dir)
            s_file = open(sfile_dir+str(doc_id)+".txt","w")
            c_file = open(cfile_dir+str(doc_id)+".txt","w")
            rc_file = open(rcfile_dir+str(doc_id)+".txt","w")
            rs_file = open(rsfile_dir+str(doc_id)+".txt","w")
            if k == 0:
                o_file = open(ofile_dir+str(doc_id)+".txt","w")
            for sen_id in range(doc_len):
                #use topK words, K is the same for each model
                deleted_num = max(int(len(doc_word_att[sen_id])*level),1)
                important_idx = list(np.array(doc_word_att[sen_id]).argsort()[-deleted_num:][::-1])
                # threshold = np.percentile(np.array(doc_word_att[sen_id]),100*(1-level))
                #calculate the  deleted number
                # deleted_idx = np.where(np.array(doc_word_att[sen_id])>threshold)
                # deleted_num = deleted_idx[0].shape[0]
                random_ids = random.choices(range(len(doc[sen_id])), k=deleted_num)
                for i in range(len(doc_word_att[sen_id])):
                    if k==0:
                        o_file.write(doc[sen_id][i]+" ")
                    if i in important_idx:
                    # if doc_word_att[sen_id][i]>threshold:
                        s_file.write(doc[sen_id][i]+" ")
                    elif i not in important_idx:
                        c_file.write(doc[sen_id][i]+" ")
                    if i in random_ids:
                        rs_file.write(doc[sen_id][i]+" ")
                    elif i not in random_ids:
                        rc_file.write(doc[sen_id][i]+" ")
                s_file.write("\n")
                c_file.write("\n")
                rs_file.write("\n")
                rc_file.write("\n")
                if k==0:
                    o_file.write("\n")
            s_file.close()
            c_file.close()
            o_file.close()
            rs_file.close()
            rc_file.close()
        label_file.writelines(str(target)+"\n")
        print("")

    def generate_wordEx(self,sent,alpha,id2word,wpos):
        """sent: wordids [sen_len,bs]
            alpha: word context attention [sen_len,bs,1]
            omega: word topic attention [sen_len,bs,1]"""

        text = sent.detach().cpu().numpy()
        alpha = alpha.squeeze(-1).detach().cpu().numpy()
        wpos = wpos.detach().cpu().numpy()
        doc_list, doc_word_att, doc_tokenids,doc_wordpos_ids = [], [],[],[]
        for bs in range(text.shape[1]):
            if len((text[:,bs]==2).nonzero()[0]) > 0 :
                doc_len = (text[:,bs]==2).nonzero()[0][0]
            else:
                doc_len = len(text[:,bs])
            doc,doc_tokenid,doc_wordpos =  [],[],[]
            sen_word_att = []
            for watt, wid,wpid in zip(alpha[:,bs],text[:,bs],wpos[:,bs]):
                if wid>2:
                    doc.append(id2word[wid])
                    sen_word_att.append(watt)
                    doc_wordpos.append(wpid)
            doc_list.append(doc)
            doc_word_att.append(sen_word_att)
            doc_wordpos_ids.append(doc_wordpos)
        return doc_list,doc_word_att,doc_wordpos_ids
    
    def symbol(self,input):
        if input>0:
            symbol = r"$\bigstar$"
        else:
            symbol = r"$\blacksquare$"
        return symbol

    def genEx(self,doc_words,doc_att,doc_wordpos_ids,logit,sen_att,output_file,doc_id,target,doc_name_list):
        """
        all the input are list of N elements for N sentences in a document
        doc_words:
        doc_att:
        doc_btt: """
        sen_att = sen_att.transpose(1,0)
        assert len(doc_att)==len(doc_words)==sen_att.shape[0] #doc_len
        pre_label = torch.argmax(logit,-1).detach().cpu().numpy()
        bsize = len(doc_words[0])
        target = target.detach().cpu().numpy()
        anots =  []
        for bs in range(bsize):
            doc_id += 1
            print(doc_id)
            doc, doc_word_att, doc_sen_att,sen_labels,doc_wpos_id = [],[],[],[],[]
            for doc_bs, att_bs, psen_att,doc_wpid_bs in zip(doc_words,doc_att,sen_att,doc_wordpos_ids):
                assert len(doc_bs[bs]) == len(att_bs[bs])
                doc.append(doc_bs[bs])
                doc_word_att.append(att_bs[bs])#will filter some unimportant word att
                doc_sen_att.append(psen_att[bs].tolist())
                doc_wpos_id.append(doc_wpid_bs[bs])

            level = 0.4 #control the percentage of important words
            print(doc_name_list[bs])
            rat_list = gen_hard_rats(doc,doc_word_att,level,doc_wpos_id)
            anot = {"annotation_id":doc_name_list[bs],"rationales":[{"docid":doc_name_list[bs],"hard_rationale_predictions":rat_list}]}
            anots.append(anot)
            # self.groups_attwords(doc,doc_word_att,doc_id,target[bs])
            # generate(doc, doc_word_att, doc_sen_att,"./latex_files/HAN_YELP_doc{}_combine.tex".format(doc_id), sen_color='green',word_color='red',pre_label=pre_label[bs],target=target[bs])
        return doc_id,anots

    #     """
    #     all the input are list of N elements for N sentences in a document
    #     doc_words:
    #     doc_att:
    #     doc_btt: """
    #     assert len(doc_att)==len(doc_words)
    #     pre_label = torch.argmax(logit,-1).detach().cpu().numpy()
    #     f = open(output_file,"a+")
    #     bsize = len(doc_words[0])
    #     #save a sentence attention heatmaps
    #     #select the most important/representative sentences from data
    #     target = target.detach().cpu().numpy()
    #     for bs in range(bsize):
    #         doc_str = ""
    #         doc_id += 1
    #         data = co_topic_weight[bs,:]
    #         # mask = np.triu(np.ones_like(data, dtype=bool))
    #         # new_mask = (mask==False)
    #         # new_data = new_mask*data
    #         senids = np.argmax(data)
    #         f.writelines("DocID{}:\n".format(doc_id))
    #         print(doc_id)
    #         sen_id = 0
    #         for doc_bs, att_bs in zip(doc_words,doc_att):
    #             if len(doc_bs)>bs:
    #                 f.writelines("SID"+str(sen_id)+": "+" ".join(doc_bs[bs])+"("+"Context: "+",".join(att_bs[bs])+"/"+")"+"\n")
    #                 if sen_id == senids:
    #                     symbol = r"$\triangleright$"
    #                     doc_str += symbol+" S{}: ".format(sen_id)+" ".join(doc_bs[bs])+" ("+" ".join(att_bs[bs])+") "+"\n"
    #                 elif not sen_id == senids:
    #                     symbol =  r"$\odot$"
    #                     doc_str += symbol+" S{}: ".format(sen_id)+" ".join(doc_bs[bs])+" ("+" ".join(att_bs[bs])+") "+"\n"
    #                 sen_id+=1

    #         f.writelines("Predict Document Label: %d\n"%pre_label[bs])
    #         f.writelines("GT Document Label: %d\n"%target[bs])
    #         f.writelines("\n")
    #         symbol = self.symbol(pre_label[bs])
    #         doc_str += "Predict Document Label: "+symbol+"\n"
    #         symbol = self.symbol(target[bs])
    #         doc_str += "GT Document Label: "+symbol+"\n"
    #         fig, axs = plt.subplots(2, 1,gridspec_kw={
    #                     'height_ratios': [1,2]})
    #         plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5, wspace=0.4)
    #         # sns.heatmap(data, linewidth=0.3,annot=True,cmap="YlGn",ax=axs[0],mask=mask)
    #         axs[0].axis('off')
    #         axs[1].text(0.01,0,doc_str,multialignment="left",wrap=True)
    #         axs[1].axis('off')
    #         #insert the wordcloud pic, should be the most predominent one. 
    #         # im = plt.imread(get_sample_data("/mnt/sda/media/Data2/hanqi/sine/sen_att/107.png"))
    #         plt.savefig("/mnt/sda/media/Data2/hanqi/sine/han_ex/human_evaluation/HAN_yelp_{}".format(doc_id))
    #         plt.close()
    #     return doc_id

    def forward(self, input_list,wordpos_list, length_list,id2word=None,output_file=None,doc_id=0,target=None,metric=None,doc_name=None,flag=None):
        """ 
        Arguments: 
        input_list (list) : list of quote utterances, the item is Variable of FloatTensor (word_length * batch_size)
                                 the length of list is number of utterance
        length_list (list): list of length utterances
        Returns:
        word_rnn_output (Variable of FloatTensor): (word_length_of_last_utterance * batch_size)
        context_rnn_ouput (Variable of FloatTensor): (num_utterance * batch_size)
        """
        num_utterance = len(input_list)
        _, batch_size = input_list[0].size()
        # word-level rnn
        word_rnn_hidden = self.init_rnn_hidden(batch_size, level="word")
        word_rnn_output_list = []
        doc_words,doc_att,doc_wordpos_ids = [],[],[]
        for utterance_index in range(num_utterance):
            word_rnn_input = self.embedding(input_list[utterance_index])
            word_rnn_output, word_rnn_hidden = self.word_rnn(word_rnn_input, word_rnn_hidden)
            word_attention_weight = self.word_conv_attention_linear(word_rnn_output) 
            word_attention_weight = self.word_conv_attention_linear2(word_attention_weight)
            word_attention_weight = nn.functional.relu(word_attention_weight)
            #hanqi:word-level importance
            word_attention_weight = nn.functional.softmax(word_attention_weight, dim=0)     
            word_rnn_last_output = torch.mul(word_rnn_output,word_attention_weight).sum(dim=0)
            word_rnn_output_list.append(word_rnn_last_output)
            word_rnn_hidden = word_rnn_hidden.detach()
            if flag == "gen_ex":
                #get alpha, omega
                sen_list,awords_list,sen_wordpos_list=self.generate_wordEx(input_list[utterance_index],word_attention_weight,id2word,wordpos_list[utterance_index])
                doc_words.append(sen_list)
                doc_att.append(awords_list)
                doc_wordpos_ids.append(sen_wordpos_list)
        # context-level rnn
        context_rnn_hidden = self.init_rnn_hidden(batch_size, level="context")
        context_rnn_input = torch.stack(word_rnn_output_list, dim=0)
        context_rnn_output,context_rnn_hidden = self.context_rnn(context_rnn_input, context_rnn_hidden)
        context_attention_weight = self.context_conv_attention_linear(context_rnn_output)
        context_attention_weight = nn.functional.relu(context_attention_weight)
        context_attention_weight = nn.functional.softmax(context_attention_weight, dim=0)
        context_rnn_last_output = torch.mul(context_rnn_output,context_attention_weight).sum(dim=0)
        classifier_input = context_rnn_last_output
        classifier_input_array = np.array(classifier_input.cpu().data)
        logit = self.classifier(classifier_input)
        #hanqi:sentence-level importance
        attention_weight_array = np.array(context_attention_weight.data.cpu().squeeze(-1)).transpose(1,0)
        if flag == "gen_ex":
            # genEx(self,doc_words,doc_att,sen_att,logit,co_topic_weight,output_file,doc_id,target):
            doc_id,anots=self.genEx(doc_words,doc_att,doc_wordpos_ids,logit,attention_weight_array,output_file,doc_id,target,doc_name)
        return logit,attention_weight_array,classifier_input_array,doc_id,anots

def evaluate(model,loss_function,batch_generator,cuda=None,id2word=None,output_file=None,metric=None,k_bin=None,flag=None):
    model.eval()
    total_loss = 0
    total_sample = 0
    batch_i = 0
    true_label_list = []
    predicted_label_list = []
    total_hit = 0
    total_sample = 0
    batch_i = 0
    true_label_list = []
    test_sample_idx = []
    predicted_label_list = []
    neg_logits,pos_logits = [] ,[]
    doc_id = 0
    annotations = []
    with torch.no_grad():
        for batch in batch_generator:
            data, target,tfidf, idx, wordpos,length, original_index = batch[0], batch[1], batch[2], batch[3],batch[4],batch[5],batch[6]
            if cuda is None:
                data_var_list = [torch.LongTensor(chunk) for chunk in data]
                target_var = torch.LongTensor(target)
                length_var = torch.LongTensor(length)
                wordpos_var_list = [torch.tensor(chunk) for chunk in wordpos]
            else:
                data_var_list = [torch.LongTensor(chunk).cuda(cuda) for chunk in data]
                target_var = torch.LongTensor(target).cuda(cuda)
                length_var = torch.LongTensor(length)
                wordpos_var_list = [torch.tensor(chunk).cuda(cuda) for chunk in wordpos]

            predicted_target,attention_weight,classifier_input,doc_id,anot = model(data_var_list, wordpos_var_list,length,id2word,output_file,doc_id,target_var,doc_name=idx,flag=flag)
            batch_i += 1
            annotations.extend(anot)
            # att_list.append(senatt_arr.cpu().detach().numpy())
            # loss = loss_function(predicted_target, target_var)
            _,predicted_label = torch.max(predicted_target,dim=1)
            neg_logits.extend(list(np.array(predicted_target[:,0].cpu())))
            pos_logits.extend(list(np.array(predicted_target[:,1].cpu())))
            total_hit += torch.sum(predicted_label.data == target_var.data)
            # total_loss += loss.item()
            total_sample += data[0].shape[1]
            true_label_list.append(target)
            test_sample_idx.extend(idx)
            predicted_label_array_batch = np.array(predicted_label.data.cpu())
            predicted_label_list.append(predicted_label_array_batch)

    true_label_array = np.asarray(list(chain(*true_label_list)),dtype=np.int32)
    predicted_label_array = np.asarray(list(chain(*predicted_label_list)),dtype=np.int32)
    acc = float(total_hit)/float(total_sample)
    with open('/home/hanq1yanwarwick/SINE/explain_metric/HAN_test_decoder_04bin_imdbweight.json', 'w') as fout:
        json.dump(annotations , fout)
    print(acc)
    # if metric == "original":
    #     pre_results = pd.DataFrame({"idx":test_sample_idx,"base_pre":predicted_label_array,"base_label":true_label_array,"base_neg_logits":neg_logits,"base_pos_logits":pos_logits})
    # else:
    #     pre_results = pd.DataFrame({"idx":test_sample_idx,"pre":predicted_label_array,"label":true_label_array,"neg_logits":neg_logits,"pos_logits":pos_logits})
    # pre_results.to_csv("/home/hanq1yanwarwick/SINE/eraser_metric/output_logits/HAN/prediction_results_{}_{}bin_selfexplain.csv".format(metric,k_bin),index_label="idx",index=False)
    # print("Predict Probalibility file of %s for %d bins is saved"%(metric,k_bin))
    # original_index_array = np.asarray(list(chain(*original_index_list)),dtype=np.int32)
    # classifier_input_array = np.concatenate(classifier_input_list,axis=0)
 
def main(k_bin):
    logging.basicConfig(level=logging.INFO,format="%(asctime)s\t%(message)s")
    """HardCode for Arguments"""
    dataset = "eraser_movie"
    metric = "hard_rats"
    k_bin = k_bin
    cuda = 0
    """HardCode for Arguments"""
    if dataset == "yelp":
        train_data_file = "/home/hanq1yanwarwick/SINE/input_data/yelp/medical_train.txt"
        test_data_file = "/home/hanq1yanwarwick/SINE/baselises/yelp_samples.txt"
        num_label = 2
    elif dataset == "imdb":
        train_data_file = "/home/hanq1yanwarwick/SINE/input_data/imdb/train.jsonlist"
        test_data_file = "/home/hanq1yanwarwick/SINE/baselises/imdb_samples.jsonlist"
        num_label = 2
    elif dataset == "eraser_movie":
        train_data_file = '/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/train.jsonl'
        test_data_file = '/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/test.jsonl'
        num_label = 2

    vocab = get_vocabulary("/home/hanq1yanwarwick/SINE/input_data/imdb/train.jsonlist",vocabsize=15000,dataset="imdb",use_stem=False,stopwords=stopwords)
    #pretrained_embedding = None
    pretrained_embedding = load_embedding(vocab,"/mnt/sda/media/Data2/hanqi/sine/glove.840B.300d.txt",embedding_size=300)
    # train_data,train_label = load(train_data_file,vocab,max_value=60,max_utterance=10)
    if metric == "hard_rats":
        test_data,test_label,test_tfidf,test_idx,test_wordposidx = load_rats(test_data_file,vocab,max_value=60,max_utterance=50,dataset=dataset,type="original",stopwords=stopwords)
    if metric == "acc" or metric == "gen_bin_dataset":
        test_data,test_label,test_tfidf,test_idx,test_wordposidx = load_rats(test_data_file,vocab,max_value=60,max_utterance=50,dataset=dataset,type="original",stopwords=stopwords)
        label_file = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/HAN/original_set/label.txt","w")
        label_file.close()
    elif metric == "cset" or metric == "sset" or metric == "original" or metric=="rsset" or metric == "rcset":
        test_data,test_label,test_tfidf,test_idx,test_wordposidx = load_bin_dataset(metric,k_bin,vocab,max_value=60,max_utterance=50,stopwords=stopwords,model="HAN")

    test_batch = DataIter(test_data,test_label,test_tfidf,test_idx,test_wordposidx,16,2,shuffle=False)#ori_code
    model = HierachicalClassifier(num_word=15000, emb_size=300, word_rnn_size=300, word_rnn_num_layer=1, word_rnn_dropout = 0.4, word_rnn_bidirectional=True,
            word_attention_size =150, context_rnn_size=150, context_rnn_dropout = 0.3, context_rnn_bidirectional=True,
            context_attention_size=200, mlp_size = 200, num_label = num_label, context_rnn_num_layer=1, pretrained_embedding=pretrained_embedding)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    loss_function = nn.CrossEntropyLoss()
    print("Loading the pretrained model weight")
    weight_file = "BestModel_IMDB_0109.pt"     
    loaded_dict = torch.load(weight_file)
    model.load_state_dict(loaded_dict)
    model.cuda(cuda)
    #evaluate includes the explanation generation
    id2word = {vocab[key]:key for key in vocab.keys()}
    output_file = "/mnt/sda/media/Data2/hanqi/sine/han_ex/HAN_explanations_{}.txt".format(dataset)
    f = open(output_file,"w")
    f.close()
    if metric == "gen_bin_dataset" or metric == "hard_rats":
        flag = "gen_ex"
    else:
        flag = "gen_noex"
    evaluate(model,loss_function,test_batch,cuda=0,id2word=id2word,output_file=output_file,metric=metric,k_bin=k_bin,flag=flag)

if __name__ == "__main__":
    #with launch_ipdb_on_exception():
    for k_bin in [999]:
        main(k_bin)
