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
import pickle
import json
#from ipdb import launch_ipdb_on_exception
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

import sys
sys.path.insert(0,"/home/hanq1yanwarwick/SINE/")

from funcs import load_rats,get_vocabulary,load,load_bin_dataset,load_general_dataset
from dataloader import DataIter

# def convert_words2ids(words, vocab, unk, max_value, sos=None, eos=None):
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
#     id_list = []
#     word_list = []
#     for w in words:
#         w = w.strip('\'"?!,.():;')
#         w = w.lower()
#         if w in vocab and len(w) > 0:
#             id_list.append(vocab[w])
#             word_list.append(w)

#     if len(id_list) > 0:
#         return id_list

# def get_vocabulary(textfile, initial_vocab={'<unk>':0,'<sssss>':1,'<eos>':2}, vocabsize=0,dataset='imdb',use_stem=False,stopwords=None):
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
#             # if w not in stopwords:
#             word_count[w] += 1

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
 
# def load(textfile, vocab, max_value, max_utterance,dataset):
#     """ Load a dialog text corpus as word Id sequences
#         Args:
#             textfile (str): filename of a dialog corpus
#             vocab (dict): word-id mapping
#         Return:
#             list of dialogue : dialogue is (input_id_list,output_id_list)
#     """
#     document_list = []
#     label_list = []
#     def filter_key(sent):
#         unk_count = sent.count(vocab['<unk>'])
#         return unk_count/len(sent) < 0.3
#     with open(textfile, "r") as f:
#         line_list = f.readlines()
#         eraser_dict  =  {"NEG":"1","POS":"2"}
#         line_len = len(line_list)
#         random_index = np.random.permutation(line_len)
#         line_list = [line_list[index] for index in random_index]
#         progressbar = ProgressBar(maxval=len(line_list))
#         word_list_buffer = []
#         for line in progressbar(line_list):
#             if dataset == "yelp":
#                 _,_,label,text = line.strip().split("\t\t")
#                 sent_list = text.strip().split("<sssss>")
#             elif dataset == "eraser_movie":
#                 label = eraser_dict[json.loads(line)['classification']]
#                 filename = json.loads(line)['annotation_id']
#                 text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/docs/"+filename,"r").readlines()
#                 sent_list = [x.strip() for x in text if len(x.split())>1] 
#             sent_list = [sent.strip().split(" ") for sent in sent_list]
#             sent_list = list(filter(lambda x:x!=["."],sent_list))
#             tokenids =  []
#             for sent in sent_list:
#                 sen_sample = convert_words2ids(sent,vocab,unk=vocab['<unk>'],max_value=max_value,eos=vocab['<eos>'])
#                 if not sen_sample is None:
#                     tokenids.append(sen_sample)
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
#                 # idx_list.append(idx)

#     def sort_key(document_with_label):
#         document = document_with_label[0]
#         first_key = len(document)  # The first key is the number of utterance of input
#         second_key = np.max([len(utterance) for utterance in document]) # The third key is the max number of word in input
#         third_key = np.mean([len(utterance) for utterance in document]) # The third key is the max number of word in input
#         return first_key,second_key,third_key
#     document_with_label_list = list(zip(*[document_list,label_list]))
#     document_with_label_list = sorted(document_with_label_list,key=sort_key)
#     document_list,label_list = list(zip(*document_with_label_list))
    # return document_list,label_list

# class DataIter(object):
#     def __init__(self, document_list, label_list, batch_size, padded_value):
#         self.document_list = document_list
#         self.label_list = label_list
#         self.batch_size = batch_size
#         self.padded_value = padded_value
#         self.batch_starting_point_list = self._batch_starting_point_list()

#     def _batch_starting_point_list(self):
#         self.num_sample = len(self.document_list)
#         batch_starting_list =  []
#         for i in range(0,self.num_sample,self.batch_size):
#             batch_starting_list.append(i)
#         return batch_starting_list
    # def _batch_starting_point_list(self):
        # num_turn_list = [len(document) for document in self.document_list]
        # batch_starting_list = []
        # previous_turn_index=-1
        # previous_num_turn=-1
        # for index,num_turn in enumerate(num_turn_list):
        #     if num_turn != previous_num_turn:
        #         if index != 0:
        #             assert num_turn > previous_num_turn
        #             num_batch = (index-previous_turn_index) // self.batch_size
        #             for i in range(num_batch):
        #                 batch_starting_list.append(previous_turn_index + i*self.batch_size)
        #         previous_turn_index = index
        #         previous_num_turn = num_turn
        # if previous_num_turn != len(self.document_list):
        #     num_batch = (index - previous_turn_index) // self.batch_size
        #     for i in range(num_batch):
        #         batch_starting_list.append(previous_turn_index + i * self.batch_size)
        # return batch_starting_list

    # def sample_document(self,index):
    #     return self.document_list[index]

    # def __iter__(self):
    #     self.current_batch_starting_point_list = copy.copy(self.batch_starting_point_list)
    #     # self.current_batch_starting_point_list = np.random.permutation(self.current_batch_starting_point_list) 
    #     self.batch_index = 0
    #     return self

    # def __next__(self):
    #     if self.batch_index >= len(self.current_batch_starting_point_list):
    #         raise StopIteration
    #     batch_starting = self.current_batch_starting_point_list[self.batch_index]
    #     batch_end = batch_starting + self.batch_size
    #     raw_batch = self.document_list[batch_starting:batch_end]
    #     label_batch = self.label_list[batch_starting:batch_end]
    #     transeposed_batch = map(list, zip(*raw_batch)) 
    #     padded_batch = []
    #     length_batch = []
    #     for transeposed_doc in transeposed_batch:
    #         length_list = [len(sent) for sent in transeposed_doc]
    #         max_length = max(length_list)
    #         new_doc = [sent+[self.padded_value]*(max_length-len(sent)) for sent in transeposed_doc]
    #         padded_batch.append(np.asarray(new_doc, dtype=np.int32).transpose(1,0))
    #         length_batch.append(length_list)
    #     padded_length = np.asarray(length_batch)
    #     padded_label = np.asarray(label_batch, dtype=np.int32) -1
    #     original_index =  np.arange(batch_starting,batch_end)
    #     self.batch_index += 1
    #     return padded_batch, padded_label, padded_length ,original_index

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

    def forward(self, input_list, length_list):
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
        for utterance_index in range(num_utterance):
            word_rnn_input = self.embedding(input_list[utterance_index])
            word_rnn_output, word_rnn_hidden = self.word_rnn(word_rnn_input, word_rnn_hidden)
            word_attention_weight = self.word_conv_attention_linear(word_rnn_output) 
            word_attention_weight = self.word_conv_attention_linear2(word_attention_weight)
            word_attention_weight = nn.functional.relu(word_attention_weight)
            word_attention_weight = nn.functional.softmax(word_attention_weight, dim=0)     
            word_rnn_last_output = torch.mul(word_rnn_output,word_attention_weight).sum(dim=0)
            word_rnn_output_list.append(word_rnn_last_output)
            word_rnn_hidden = word_rnn_hidden.detach()  
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
        attention_weight_array = np.array(context_attention_weight.data.cpu().squeeze(-1)).transpose(1,0)
        return logit,attention_weight_array,classifier_input_array

def evaluate(model,loss_function,batch_generator,cuda=None):
    model.eval()
    total_loss = 0
    total_hit = 0
    total_sample = 0
    batch_i = 0
    true_label_list = []
    predicted_label_list = []
    attention_weight_list = []
    classifier_input_list = []
    original_index_list = []
    with torch.no_grad():
        for batch in batch_generator:
            data, target,tfidf,idx,pos, length, original_index = batch[0], batch[1], batch[2], batch[3],batch[4],batch[5],batch[6]
            if cuda is None:
                data_var_list = [torch.LongTensor(chunk) for chunk in data]
                target_var = torch.LongTensor(target)
                length_var = torch.LongTensor(length)
            else:
                data_var_list = [torch.LongTensor(chunk).cuda(cuda) for chunk in data]
                target_var = torch.LongTensor(target).cuda(cuda)
                length_var = torch.LongTensor(length)
            predicted_target,attention_weight,classifier_input = model(data_var_list, length)
            loss = loss_function(predicted_target, target_var)
            _,predicted_label = torch.max(predicted_target,dim=1)
            total_hit += torch.sum(predicted_label.data == target_var.data)
            total_loss += loss.item()
            total_sample += data[0].shape[1]
            true_label_list.append(target)
            predicted_label_array_batch = np.array(predicted_label.data.cpu())
            predicted_label_list.append(predicted_label_array_batch)
            attention_weight_list.append(attention_weight)
            classifier_input_list.append(classifier_input)
            original_index_list.append(original_index)
            batch_i += 1

    # true_label_array = np.asarray(list(chain(*true_label_list)),dtype=np.int32)
    # predicted_label_array = np.asarray(list(chain(*predicted_label_list)),dtype=np.int32)
    # original_index_array = np.asarray(list(chain(*original_index_list)),dtype=np.int32)
    # classifier_input_array = np.concatenate(classifier_input_list,axis=0) 
    acc = float(total_hit)/float(total_sample)
    print("batch_i: %d number of evaluate sampels: %d"%(batch_i,total_sample))
    # returned_document_list = [batch_generator.sample_document(index) for index in original_index_array]
    model.train()
    return total_loss/(batch_i+1),acc

def train_model(model,optimizer,loss_function,num_epoch,train_batch_generator,test_batch_generator,vocab,cuda=None):
    logging.info("Start Tranining")
    if cuda != None:
        model.cuda(cuda)
    reverse_vocab = {vocab[key]:key for key in vocab.keys()}
    best_model_loss,best_test_acc = 1e7,0 
    temp_batch_index = 0
    for epoch_i in range(num_epoch):
        logging.info("Epoch {}".format(epoch_i))
        temp_batch_index = 0
        for train_batch in train_batch_generator:
            temp_batch_index += 1
            train_data,train_target,tfidf,idx,wpos,length_data = train_batch[0],train_batch[1],train_batch[2],train_batch[3],train_batch[4],train_batch[5]
            if cuda is None:
                train_data_var_list = [torch.LongTensor(chunk) for chunk in train_data]
                train_target_var = torch.LongTensor(train_target)
                length_var = torch.LongTensor(length_data)
            else:
                train_data_var_list = [torch.LongTensor(chunk).cuda(cuda) for chunk in train_data]
                train_target_var = torch.LongTensor(train_target).cuda(cuda)
                length_var = torch.LongTensor(length_data)
            predicted_train_target,_,_ = model(train_data_var_list,length_var)
            optimizer.zero_grad()
            loss = loss_function(predicted_train_target,train_target_var)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            # if temp_batch_index % 100 == 0:
        # train_loss,train_acc = 0.0,0.0 #print results per epoch
        print("num_train_sample%d"%(temp_batch_index*16))
        test_loss,test_acc = evaluate(model,loss_function,test_batch_generator,cuda)
        predicted_label = np.argmax(predicted_train_target.detach().cpu().numpy(),axis=1)
        train_hits = np.count_nonzero(predicted_label == train_target_var.data.detach().cpu().numpy())
        train_acc = train_hits/len(predicted_label)
        logging.info("\nBatch :{0:8d}\ntrain_loss:{1:0.6f}\ttrain_acc:{2:0.6f}\ntest_loss:{3:0.6f}\ttest_acc:{4:0.6f}".format(temp_batch_index, loss.item(),train_acc,test_loss,test_acc))

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            filename =  "BestModel_IMDB_0109.pt"
            torch.save(model.state_dict(), filename)
            print("Save best model with test_acc %.3f"%(best_test_acc))
        # reverse_vocab = {vocab[key]:key for key in vocab.keys()}
        # logging.info("True : {} \t Predicted : {}".format(true_label_array[0], predicted_label_array[0]))
        # logging.info(attention_weight[0][0])
        sent_str_list = []
        # filename =  "besthan_imdb.pt"
        # torch.save(model.state_dict(), filename)
        # print("Save best model")
                #for sent in document_list[0]:
                #    sent_str_list.append(" ".join([reverse_vocab[word] for word in sent]))
                #logging.info("\n"+"\n".join(sent_str_list))
                # sent_length_array = np.asarray([len(document) for document in document_list],dtype=np.int32)
                # np.save("data/sent_length_{}".format(temp_batch_index),sent_length_array)
                # np.save("data/classifier_input_embedding_{}".format(temp_batch_index),classifier_input_array)
                # np.save("data/true_label_{}".format(temp_batch_index),true_label_array)
                # np.save("data/predicted_label_{}".format(temp_batch_index), predicted_label_array)

def error_analysis(batch_generator, wrong_index, predicted_label_array, true_label_array):
    wrong_document_list = [batch_generator.sample_document(index) for index in wrong_index]
    wrong_length_counter = Counter()
    total_length_counter = batch_generator.length_count()
    for doc in wrong_document_list:
        wrong_length_counter[len(doc)] += 1
    for length in sorted(wrong_length_counter.keys()):
        print("Length : {0} \t ACC: {1:6f} \t total_num : {2:6d} \t wrong_num: {3:6d}".format(length, 1-wrong_length_counter[length]/total_length_counter[length],
                                                                                        total_length_counter[length], wrong_length_counter[length]))
    
    fusion_array = np.zeros((5,5))
    assert predicted_label_array.shape == true_label_array.shape
    for predicted_label, true_label in zip(predicted_label_array, true_label_array):
        fusion_array[predicted_label, true_label] += 1
    fusion_array = fusion_array / np.sum(fusion_array, axis=1, keepdims=True)
    print("\t{0:6d}\t\t{1:6d}\t\t{2:6d}\t\t{3:6d}\t\t{4:6d}".format(1,2,3,4,5))
    for true_label,row in enumerate(fusion_array):
        print(true_label+1,end="\t")
        for predicted_label in row:
            print("{0:6f}".format(predicted_label),end="\t")
        print()
def load_process(dataset,type,data):
    filename = '/home/hanq1yanwarwick/SINE/input_data/{}/{}_{}_list.pkl'.format(dataset,type,data)
    with open(filename, 'rb') as f:
        result = pickle.load(f)
    return result   

def main():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s\t%(message)s")
    dataset = "imdb"
    if dataset == "yelp":
        train_data_file = "/home/hanq1yanwarwick/SINE/input_data/yelp/medical_train.txt"
        test_data_file = "/home/hanq1yanwarwick/SINE/input_data/yelp/medical_test.txt"
    elif dataset == "imdb":
        train_data_file = "/home/hanq1yanwarwick/SINE/input_data/imdb/train.jsonlist"
        test_data_file = "/home/hanq1yanwarwick/SINE/input_data/imdb/test.jsonlist"
    elif dataset == "eraser_movie":
        train_data_file = '/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/train.jsonl'
        test_data_file = '/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/test.jsonl'
    # test_data_file = "medical_test.txt"
    # textfile, initial_vocab={'<unk>':0,'<sssss>':1,'<eos>':2}, vocabsize=0,dataset='imdb',use_stem=False
    vocab = get_vocabulary(train_data_file,vocabsize=15000,dataset=dataset,use_stem=False,stopwords=stopwords)
    #pretrained_embedding = None
    pretrained_embedding = load_embedding(vocab,"/mnt/sda/media/Data2/hanqi/sine/glove.840B.300d.txt",embedding_size=300)
    if dataset == "eraser_movie":
        train_data,train_label,train_tfidf,train_idx,train_wordposidx = load_rats(train_data_file,vocab,max_value=60,max_utterance=50,dataset=None,type="original",stopwords=stopwords)
        test_data,test_label,test_tfidf,test_idx,test_wordposidx = load_rats(test_data_file,vocab,max_value=60,max_utterance=50,dataset=dataset,type="original",stopwords=stopwords)
    else:
        train_data,train_label,train_tfidf,train_idx,train_wordposidx = load_general_dataset(train_data_file,vocab,max_value=60,max_utterance=50,dataset=dataset,type="original",stopwords=stopwords)
        test_data,test_label,test_tfidf,test_idx,test_wordposidx = load_general_dataset(test_data_file,vocab,max_value=60,max_utterance=50,dataset=dataset,type="original",stopwords=stopwords)
    # train_data = load_process(dataset,"train","data")
    # train_label = load_process(dataset,"train","label")
    # test_data = load_process(dataset,"test","data")
    # test_label = load_process(dataset,"test","label")
    train_batch = DataIter(train_data,train_label,train_tfidf,train_idx,train_wordposidx,16,2)
    test_batch = DataIter(test_data,test_label,test_tfidf,test_idx,test_wordposidx,16,2)
    model = HierachicalClassifier(num_word=15000, emb_size=300, word_rnn_size=300, word_rnn_num_layer=1, word_rnn_dropout = 0.4, word_rnn_bidirectional=True,
            word_attention_size =150, context_rnn_size=150, context_rnn_dropout = 0.3, context_rnn_bidirectional=True,
            context_attention_size=200, mlp_size = 200, num_label = 2, context_rnn_num_layer=1, pretrained_embedding=pretrained_embedding)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    loss_function = nn.CrossEntropyLoss()
    train_model(model,optimizer,loss_function,30,train_batch,test_batch,vocab,cuda=0)

if __name__ == "__main__":
    #with launch_ipdb_on_exception():
    main()
