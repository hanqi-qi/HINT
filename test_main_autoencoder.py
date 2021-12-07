# CNN Attentional Document Classification
# Author : Jiachen DU
# Last Modified : 9th, May. 2019
# Copyright Reserved, 2018, 2019

#modify the load function for different datasets, also calculate the document lengthe
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

parser = argparse.ArgumentParser(description='MASK_LSTM text classificer')
parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('-num_epoch', type=int, default=30, help='epochs')
parser.add_argument('-batch_size', type=int, default=32, help='batch size')
parser.add_argument('-d_t', type=int, default=50, help='number of topics in code')
parser.add_argument('-cuda', type=int, default=1, help='gpu id')
parser.add_argument('-pretrain_emb', type=str, default=1, help='use glove or googlenews word embedding')
args = parser.parse_args()


# class B:
#     text = torch.zeros(1).to("cuda:0")
#     label = torch.zeros(1).to("cuda:0")

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
        word2vec = KeyedVectors.load_word2vec_format('/mnt/sda/media/Data2/hanqi/sine/GoogleNews-vectors-negative300.bin', binary=True)
        for i in range(len(id2word)):
            word = id2word[i]
            if word in word2vec.key_to_index:
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
 
def load(textfile, vocab, max_value, max_utterance,dataset='yelp'):
    """ Load a dialog text corpus as word Id sequences
        Args:
            textfile (str): filename of a dialog corpus
            vocab (dict): word-id mapping
        Return:
            list of dialogue : dialogue is (input_id_list,output_id_list)
    """
    document_list = []
    label_list = []
    imbd_dic = {"neg":"1","pos":"2"}
    def filter_key(sent):
        unk_count = sent.count(vocab['<unk>'])
        return unk_count/len(sent) < 0.3
    with open(textfile, "r") as f:
        line_list = f.readlines()
        line_len = len(line_list)
        random_index = np.random.permutation(line_len)
        line_list = [line_list[index] for index in random_index]
        # progressbar = ProgressBar(maxval=len(line_list))
        word_list_buffer = []
        # for line in progressbar(line_list):
        for line in line_list:
            if dataset=='yelp':
                _,_,label,text = line.strip().split("\t\t") #2 pos, 1neg
                sents = text.strip().split("<sssss>")
            elif dataset=='imdb':
                label = imbd_dic[json.loads(line)['sentiment']]
                text = json.loads(line)['text']
                sents = [x for i in text.strip().split('.') for x in i.split('<br /><br />')]
            elif dataset=='news':
                text,label = line.strip().split("\t")
                label = int(label)+1
                sents = text.strip().split(".")               
            sent_list = [sent.strip().split(" ") for sent in sents]
            sent_list = list(filter(lambda x:x!=["."],sent_list))
            sent_id_list = [convert_words2ids(sent,vocab,max_value=max_value,unk=vocab['<unk>'],use_stem=False) for sent in sent_list]
            sent_id_list = list(filter(filter_key,sent_id_list))
            new_sent_id_list = []
            previous_sent = []
            for sent in sent_id_list:
                if len(previous_sent) != 0:
                    new_sent = previous_sent + sent
                else:
                    new_sent = sent
                if len(new_sent) < 3:
                    previous_sent = new_sent
                else:
                    new_sent_id_list.append(new_sent)
                    previous_sent = []
            if len(previous_sent) > 0:
                new_sent_id_list.append(previous_sent)
            if len(new_sent_id_list) > 0: 
                document_list.append(new_sent_id_list[:max_utterance])
                label_list.append(int(label))
    def sort_key(document_with_label):
        document = document_with_label[0]
        first_key = len(document)  # The first key is the number of utterance of input
        second_key = np.max([len(utterance) for utterance in document]) # The third key is the max number of word in input
        third_key = np.mean([len(utterance) for utterance in document]) # The third key is the max number of word in input
        return first_key,second_key,third_key
    document_with_label_list = list(zip(*[document_list,label_list]))
    document_with_label_list = sorted(document_with_label_list,key=sort_key)
    document_list,label_list = list(zip(*document_with_label_list))
    return document_list,label_list

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

class DataIter(object):
    def __init__(self, document_list, label_list, batch_size, padded_value):
        self.document_list = document_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.padded_value = padded_value
        self.batch_starting_point_list = self._batch_starting_point_list()

    def _batch_starting_point_list(self):
        num_turn_list = [len(document) for document in self.document_list]
        batch_starting_list = []
        previous_turn_index=-1
        previous_num_turn=-1
        for index,num_turn in enumerate(num_turn_list):
            if num_turn != previous_num_turn:
                if index != 0:
                    assert num_turn == previous_num_turn + 1
                    num_batch = (index-previous_turn_index) // self.batch_size
                    for i in range(num_batch):
                        batch_starting_list.append(previous_turn_index + i*self.batch_size)
                previous_turn_index = index
                previous_num_turn = num_turn
        if previous_num_turn != len(self.document_list):
            num_batch = (index - previous_turn_index) // self.batch_size
            for i in range(num_batch):
                batch_starting_list.append(previous_turn_index + i * self.batch_size)
        return batch_starting_list

    def sample_document(self,index):
        return self.document_list[index]

    def __iter__(self):
        self.current_batch_starting_point_list = copy.copy(self.batch_starting_point_list)
        self.current_batch_starting_point_list = np.random.permutation(self.current_batch_starting_point_list) 
        self.batch_index = 0
        return self

    def __next__(self):
        if self.batch_index >= len(self.current_batch_starting_point_list):
            raise StopIteration
        batch_starting = self.current_batch_starting_point_list[self.batch_index]
        batch_end = batch_starting + self.batch_size
        raw_batch = self.document_list[batch_starting:batch_end]
        label_batch = self.label_list[batch_starting:batch_end]
        transeposed_batch = map(list, zip(*raw_batch)) 
        padded_batch = []
        length_batch = []
        for transeposed_doc in transeposed_batch:
            length_list = [len(sent) for sent in transeposed_doc]
            max_length = max(length_list)
            new_doc = [sent+[self.padded_value]*(max_length-len(sent)) for sent in transeposed_doc]
            padded_batch.append(np.asarray(new_doc, dtype=np.int32).transpose(1,0))
            length_batch.append(length_list)
        padded_length = np.asarray(length_batch)
        padded_label = np.asarray(label_batch, dtype=np.int32) -1
        original_index =  np.arange(batch_starting,batch_end)
        self.batch_index += 1
        return padded_batch, padded_label, padded_length ,original_index

class HierachicalClassifier(nn.Module):
    def __init__(self, d_t, num_word, emb_size, word_rnn_size, word_rnn_num_layer, word_rnn_dropout, word_rnn_bidirectional,word_attention_size, 
                context_rnn_size, context_rnn_num_layer, context_rnn_dropout, context_rnn_bidirectional, context_attention_size, mlp_size, num_label, pretrained_embedding=None):
        self.emb_size = emb_size
        self.d_t = d_t
        self.word_rnn_size = word_rnn_size
        self.word_rnn_num_layer = word_rnn_num_layer
        self.word_rnn_bidirectional = word_rnn_bidirectional
        self.context_rnn_size = context_rnn_size
        self.context_rnn_num_layer = context_rnn_num_layer
        self.context_rnn_bidirectional = context_rnn_bidirectional
        self.num_label = num_label
        super(HierachicalClassifier, self).__init__()
        self.embedding = nn.Embedding(num_word, emb_size)
        self.word_dropout = nn.Dropout(word_rnn_dropout)
        self.word_rnn = nn.GRU(input_size = emb_size, hidden_size = word_rnn_size,
                num_layers = word_rnn_num_layer, bidirectional = word_rnn_bidirectional)
        word_rnn_output_size = word_rnn_size * 2 if word_rnn_bidirectional else word_rnn_size
        self.word_conv_attention_layer = nn.Conv1d(emb_size, word_attention_size, 3, padding=2, stride=1)
        self.word_conv_attention_linear = nn.Linear(word_attention_size, 1, bias=False)
        self.word_aspect_attention_linear = nn.Linear(word_rnn_output_size, self.d_t, bias=False)
        self.word_aspect_attention_linear2 = nn.Linear(self.d_t, 1, bias=False)
        self.topic_encoder = nn.Linear(emb_size, self.d_t, bias=False)
        self.topic_decoder = nn.Linear(self.d_t, emb_size, bias=False)
        self.context_dropout = nn.Dropout(context_rnn_dropout)
        self.context_rnn = nn.GRU(input_size = word_rnn_output_size, hidden_size = context_rnn_size,
                num_layers = context_rnn_num_layer,bidirectional=context_rnn_bidirectional)
        context_rnn_output_size = context_rnn_size * 2 if context_rnn_bidirectional else context_rnn_size
        self.context_conv_attention_layer = nn.Conv1d(word_rnn_output_size, context_attention_size, kernel_size=1, stride=1)
        self.context_conv_attention_linear = nn.Linear(context_attention_size, self.d_t, bias=False)
        self.context_topic_attention_linear = nn.Linear(self.d_t, self.d_t, bias = True)
        
        self.classifier = nn.Sequential(nn.Linear(context_rnn_output_size, mlp_size),
                                        nn.LeakyReLU(),
                                        nn.Linear(mlp_size, num_label),
                                        nn.Tanh())
        self.device = args.cuda
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
        num_utterance = len(input_list) #ori code: len(input_list)=doc_len, input_list[i]=
        _, batch_size = input_list[0].size() #old code
        # print(batch_size)
        # batch_size = len(input_list)
        # word-level rnn
        word_rnn_hidden = self.init_rnn_hidden(batch_size, level="word")
        word_rnn_output_list = []
        word_aspect_output_list = []
        aspect_loss = torch.zeros(batch_size).cuda(self.device)
        for utterance_index in range(num_utterance):
            word_rnn_input = self.embedding(input_list[utterance_index])
            word_rnn_input = self.word_dropout(word_rnn_input)
            word_rnn_output, word_rnn_hidden = self.word_rnn(word_rnn_input, word_rnn_hidden)
            word_attention_weight = self.word_conv_attention_layer(word_rnn_input.permute(1,2,0))
            word_attention_weight = word_attention_weight[:,:,1:-1]
            word_attention_weight = word_attention_weight.permute(2, 0 ,1)
            word_attention_weight = self.word_conv_attention_linear(word_attention_weight)
            word_attention_weight = nn.functional.relu(word_attention_weight)
            word_attention_weight = nn.functional.softmax(word_attention_weight, dim=0)
            word_rnn_last_output = torch.mul(word_rnn_output,word_attention_weight).sum(dim=0) #sentiment representation of the sentence,s
            word_rnn_output_list.append(word_rnn_last_output)
            word_rnn_hidden = word_rnn_hidden.detach()
            
            
            word_aspect_weight = self.word_aspect_attention_linear(word_rnn_output) 
            word_aspect_weight = self.word_aspect_attention_linear2(word_aspect_weight)
            word_aspect_weight = nn.functional.relu(word_aspect_weight)
            word_aspect_weight = nn.functional.softmax(word_aspect_weight, dim=0)
            word_aspect_output = torch.mul(word_rnn_input,word_aspect_weight).sum(dim=0)#z
            word_aspect = self.topic_encoder(word_aspect_output)#h
            recons_word = self.topic_decoder(word_aspect)#z'
            r = nn.functional.normalize(recons_word)
            z = nn.functional.normalize(word_aspect_output)
            n = nn.functional.normalize(word_rnn_last_output)
            #print(r.size(),z.size(),n.size())
            y = torch.ones(batch_size).cuda(self.device) - torch.sum(r*z, 1) + torch.sum(r*n, 1)
            word_aspect_output_list.append(word_aspect)
            aspect_loss += nn.functional.relu(y)
            word_rnn_hidden = word_rnn_hidden.detach()  
        
        
        # context-level rnn
        #seq_len should be doc_len
        context_rnn_hidden = self.init_rnn_hidden(batch_size, level="context")
        context_rnn_input = torch.stack(word_rnn_output_list, dim=0)
        context_rnn_input = self.context_dropout(context_rnn_input)
        context_rnn_output,context_rnn_hidden = self.context_rnn(context_rnn_input, context_rnn_hidden)#[seq_len,bs,dim]

        context_attention_weight = self.context_conv_attention_layer(context_rnn_input.permute(1,2,0))#[bs,dim,seq_len]
        context_attention_weight = context_attention_weight.permute(2, 0, 1)#[seq_len,bs,dim]
        context_attention_weight = nn.functional.relu(context_attention_weight)
        context_attention_weight = self.context_conv_attention_linear(context_attention_weight)#transform the snetence representation's feature dim to be the same as the aspect-level representation's feature, d_t [seq_len,bs,d_t]

        #context_attention_weight = nn.functional.relu(context_attention_weight)
        #context_attention_weight = nn.functional.softmax(context_attention_weight, dim=0)
        
        word_aspect = torch.stack(word_aspect_output_list, dim=0)#[seq_len, bs,d_t]
        context_topic_weight = self.context_topic_attention_linear(word_aspect) #[seq_len,bs,d_t]
        #print(context_topic_weight.size())
        #print(context_attention_weight.size())
        #print(context_rnn_output.size())
        context_topic_weight = context_topic_weight.permute(1,2,0)#[bs,d_t,seq_len]

        context_attention_weight = context_attention_weight.permute(1,0,2) #[bs,seq_len,d_t]
        self_attention = torch.div(nn.functional.softmax(torch.bmm(context_attention_weight, context_topic_weight),2),1.5)#[bs,seq_len,d_t]*[bs,d_t,seq_len] ->[bs,seq_len,seq_len]
        context_rnn_output = context_rnn_output.permute(1,0,2)#[bs,seq_len,dim]
        context_rnn_transform = 0.5 * torch.bmm(self_attention,context_rnn_output) + 0.5 * context_rnn_output #[bs,seq_len,seq_len]*[bs,seq_len,dim]+[bs,seq_len,dim]

        #print(context_rnn_output.size())
        #context_topic_weight = nn.functional.relu(context_topic_weight)
        #context_topic_weight = nn.functional.softmax(context_topic_weight, dim=0)     
        
        #context_rnn_last_output = torch.mul(context_rnn_output,context_attention_weight ).sum(dim=0)
        context_rnn_last_output = torch.mean(context_rnn_transform,1) #average the 
        classifier_input = context_rnn_last_output
        classifier_input_array = np.array(classifier_input.cpu().data)
        logit = self.classifier(classifier_input) 
        #attention_weight_array = np.array(context_attention_weight.data.cpu().squeeze(-1)).transpose(1,0)
        attention_weight_array = 0
        return logit,attention_weight_array,classifier_input_array,aspect_loss,torch.std(self_attention)

def evaluate(model,loss_function,batch_generator,cuda=None):
    model.eval()
    total_loss = 0
    total_hit = 0
    total_sample = 0
    batch_i = 0
    true_label_list = []
    predicted_label_list = []
    att_list = []
    attention_weight_list =[]
    classifier_input_list = []
    original_index_list = []
    with torch.no_grad():
        for batch in batch_generator:
            data, target, length, original_index = batch[0], batch[1], batch[2], batch[3]
            if cuda is None:
                data_var_list = [torch.LongTensor(chunk) for chunk in data]
                target_var = torch.LongTensor(target)
                length_var = torch.LongTensor(length)
            else:
                data_var_list = [torch.LongTensor(chunk).cuda(cuda) for chunk in data]
                target_var = torch.LongTensor(target).cuda(cuda)
                length_var = torch.LongTensor(length)
            predicted_target,attention_weight,classifier_input,_,senatt_std = model(data_var_list, length)
            att_list.append(senatt_std.cpu().detach().numpy())
            loss = loss_function(predicted_target, target_var)
            _,predicted_label = torch.max(predicted_target,dim=1)
            total_hit += torch.sum(predicted_label.data == target_var.data)
            total_loss += loss.item()
            total_sample += data[0].shape[1]
            true_label_list.append(target)
            predicted_label_array_batch = np.array(predicted_label.data.cpu())
            predicted_label_list.append(predicted_label_array_batch)
            # attention_weight_list.append(attention_weight)
            classifier_input_list.append(classifier_input)
            original_index_list.append(original_index)
            batch_i += 1

    true_label_array = np.asarray(list(chain(*true_label_list)),dtype=np.int32)
    predicted_label_array = np.asarray(list(chain(*predicted_label_list)),dtype=np.int32)
    original_index_array = np.asarray(list(chain(*original_index_list)),dtype=np.int32)
    classifier_input_array = np.concatenate(classifier_input_list,axis=0) 
    acc = float(total_hit)/float(total_sample)
    returned_document_list = [batch_generator.sample_document(index) for index in original_index_array]
    model.train()
    return total_loss/(batch_i+1),acc,true_label_array,predicted_label_array,attention_weight_list,returned_document_list,classifier_input_array,sum(att_list)/len(att_list)

def vis_dec(id2word, indices_decoder,model,rank,indices_rank,epoch_i):
    topic = ""
    t = 0
    decoder_ranking = []
    encoder_ranking = []
    label_topic = []

    #costume the maker styles
    markers_1 = ['o', 's', '^', 'x', '+','s']
    markers_2 = ["$\u2660$", "$\u2661$", "$\u2662$", "$\u2663$","$\u2680$",'+']
    color = ['r', 'b', 'r', 'c', 'm', 'y', 'k', 'w']

    t_m,t_c = 0,0
    for i in indices_decoder.t():# enumerate the vocabs for all the topics(50)
        t += 1
        for j in i: #the j-th vocab for the topic
            topic = topic + " "+ id2word[j.item()]
            decoder_ranking.append(model.embedding.weight[j.item()].cpu().detach().numpy())
        #for t-th topic, the attention weight for positive label: 
        print("decoder topic \#"+ str(t) + ":" + topic+ ". score: " + str(rank[t-1,0].item()) + ", " + str(rank[t-1,1].item()))
        print(" ")
        topic = " "
    decoder_embedded = TSNE(n_components=2).fit_transform(decoder_ranking)

    t0 = 0
    t1 = 0
    t_c = 0
    t_m0 = -1
    t_m1 = -1
    topic_indices = -1
    #enumerate all the top10 words for the top5 topics
    for i in range(decoder_embedded.shape[0]):
        if (i+1)%10 == 0: #begin the new topic
            topic_indices += 1
        a = decoder_embedded[i,]
        if topic_indices in indices_rank[:,0]: #the topic predominates in the positive label
            if t0%10 == 0:
                t_m0 += 1
            plt.plot(a[0],a[1],c=color[0],marker=markers_1[t_m0],markersize=12)
            t0 += 1
        if topic_indices in indices_rank[:,1]:
            if t1%10 == 0:
                t_m1 += 1
            plt.plot(a[0],a[1],c=color[1],marker=markers_2[t_m1],markersize=12)
            t1 += 1
    path = './imdb_output/full_decoder_' + str(epoch_i) + '.png'
    plt.savefig(path)
    plt.clf()

    return

def train_model(model,optimizer,loss_function,num_epoch,train_batch_generator,test_batch_generator,vocab,cuda=None,d_t=0):
    logging.info("Start Tranining")
    if cuda != None:
        model.cuda(cuda)
    reverse_vocab = {vocab[key]:key for key in vocab.keys()}
    best_model_loss = 1e7
    temp_batch_index = 0
    loss_C_total = 0
    loss_A_total = 0
    loss_R_total = 0
    log_loss = open('loss.txt', 'a')
    best_dev_acc = 0
    for epoch_i in range(num_epoch):
        logging.info("Epoch {}".format(epoch_i))
        for train_batch in train_batch_generator:
        # for stidx in range(0, len(data_label), 50):
            temp_batch_index += 1
            # print("temp_batch_index is%.4f"%temp_batch_index)
            train_data,train_target,length_data = train_batch[0],train_batch[1],train_batch[2] 
            if cuda is None:
                train_data_var_list = [torch.LongTensor(chunk) for chunk in train_data]
                train_target_var = torch.LongTensor(train_target)
                length_var = torch.LongTensor(length_data)
            else:
                train_data_var_list = [torch.LongTensor(chunk).cuda(cuda) for chunk in train_data]
                train_target_var = torch.LongTensor(train_target).cuda(cuda)
                length_var = torch.LongTensor(length_data)

            predicted_train_target,_,_,aspect_loss,senatt = model(train_data_var_list,length_var)
            optimizer.zero_grad()
            loss_C = loss_function(predicted_train_target,train_target_var)
            loss_A = aspect_loss.mean()
            loss_R = torch.norm(torch.eye(d_t).cuda(cuda) - torch.mm(model.topic_encoder.weight, model.topic_encoder.weight.t()))
            loss_C_total += loss_C
            loss_A_total += loss_A
            loss_R_total += loss_R
            loss = loss_C + 0.05 * loss_A +  0.01 * loss_R
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
        # if temp_batch_index%100 == 0: #every minibatch, check the intepretatbility
            # id2word = {vocab[key]:key for key in vocab.keys()}
            if temp_batch_index%100==0: #change to minibatch
                train_loss,train_acc = 0,0
                C_loss = loss_C_total/(1000)
                A_loss = loss_A_total/(1000)
                R_loss = loss_R_total
                log_loss.write("{0:6f},{1:6f},{2:6f}\n".format(C_loss,A_loss,R_loss))
                loss_A_total, loss_C_total, loss_R_total = 0,0,0

                test_loss,test_acc,true_label_array,predicted_label_array,attention_weight,document_list,classifier_input_array,senatt_std = evaluate(model,loss_function,test_batch_generator,cuda)#ori_code
                # logging.info("The std of attention weight:{0:0.6f}".format(senatt_std))
                _,predicted_label = torch.max(predicted_train_target,dim=1)
                train_hits= torch.sum(predicted_label.data == train_target_var.data)
                train_acc = train_hits/len(predicted_label)
                logging.info("\nEpoch :{0:8d}\ntrain_loss:{1:0.6f}\ttrain_acc:{2:0.6f}\ntest_loss:{3:0.6f}\ttest_acc:{4:0.6f}".format(epoch_i, C_loss,train_acc.data,test_loss,test_acc))
                if test_acc>best_dev_acc:
                    best_dev_acc = test_acc
                    print("higher dev acc: %.4f"%best_dev_acc)
                ''''VISUALIZATION'''
                id2word = {vocab[key]:key for key in vocab.keys()}
                #vocabsize->topic(d_t) matrix: dis_decoder
                dis_decoder = torch.mm(model.embedding.weight, model.topic_decoder.weight)
                dis_encoder = torch.mm(model.embedding.weight, model.topic_encoder.weight.t())
                #trace.shape[topic(d_t),word_embedding(rnn_units)]
                trace = torch.mm(model.context_topic_attention_linear.weight.t(), model.context_conv_attention_linear.weight)
                trace = torch.mm(trace, torch.squeeze(model.context_conv_attention_layer.weight))
                #use trace to claasify, topics as the sequence token, rnn units keep unchange
                rank = torch.squeeze(model.classifier(trace))
                #select top10 words for each topic
                sim_decoder , indices_decoder = torch.topk(dis_decoder, 10, dim=0)
                #select top5 topics
                sim_rank, indices_rank = rank.topk(5,dim=0)
                # vis_dec(id2word, indices_decoder,model,rank,indices_rank,epoch_i)

    logging.info("best_dev_acc:{0:0.4f}".format(best_dev_acc))
    return best_dev_acc


                


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
    

def main():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s\t%(message)s")
    dataset = 'yelp'
    if dataset=="yelp":
        train_data_file = "./input_data/yelp/medical_train.txt"
        test_data_file = "./input_data/yelp/medical_test.txt"
    elif dataset == 'imdb':
        train_data_file = "./input_data/imdb/train.jsonlist"
        test_data_file = "./input_data/imdb/test.jsonlist"
    elif dataset =="news":
        train_data_file = './input_data/guadian_news/train_news_data.txt'
        test_data_file = './input_data/guadian_news/test_news_data.txt'

    vocab = get_vocabulary(train_data_file,vocabsize=5000,dataset=dataset,use_stem=False)
    train_data,train_label = load(train_data_file,vocab,max_value=60,max_utterance=10,dataset=dataset)
    test_data,test_label = load(test_data_file,vocab,max_value=60,max_utterance=10,dataset=dataset)
    pretrain_emb = 'googlenews' #glove or googlenews

    if pretrain_emb == 'googlenews':
        pretrained_embedding,hit_rate = getVectors(embed_dim=300, wordvocab=vocab)
    else:
        pretrained_embedding = load_embedding(vocab,"./data/glove.840B.300d.txt",embedding_size=300)

    grid_search = {}
    params = {
        "batch_size": [64],
        "d_t":[50],
        "lr":[1e-4],
        "num_epoch":[30],
        "cuda":[1,2],
        'num_word':[15000], #time-consuming!
        'pretrain_emb':[pretrain_emb]
        }
    params_search = list(ParameterGrid(params))
    acc_list =[]
    for i, param in enumerate(params_search):
        print("*************params_search_{}*************".format(i + 1))
        print(param)
        for key, value in param.items():
            setattr(args, key, value)
        # p_list, r_list, f1_list = [], [], []
        train_batch = DataIter(train_data,train_label,args.batch_size,args.cuda)#ori_code
        test_batch = DataIter(test_data,test_label,args.batch_size,args.cuda)#ori_code
        model = HierachicalClassifier(d_t=args.d_t, num_word=5000, emb_size=300, word_rnn_size=150, word_rnn_num_layer=1, word_rnn_dropout = 0.4, word_rnn_bidirectional=True,
            word_attention_size =150, context_rnn_size=150, context_rnn_dropout = 0.3, context_rnn_bidirectional=True,
            context_attention_size=200, mlp_size = 200, num_label = 2, context_rnn_num_layer=1, pretrained_embedding=pretrained_embedding)

        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
        loss_function = nn.CrossEntropyLoss()
        best_dev_acc = train_model(model,optimizer,loss_function,args.num_epoch,train_batch,test_batch,vocab,cuda=args.cuda,d_t=args.d_t)#ori_code
        # acc_list.append(best_dev_acc)
        grid_search[str(param)] = {"best_dev_acc": [round(best_dev_acc, 4)]}
    
    # print("hit rate: ", hit_rate)
    for key, value in grid_search.items():
        print("Main: ", key, value)

if __name__ == "__main__":
    #with launch_ipdb_on_exception():
    main()
