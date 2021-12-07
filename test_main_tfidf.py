# Difference from jase.com:the omega shape is [bs,seq-len,emb_size], GPU resource limits!
import logging
import torch.nn.functional as F
import argparse
import copy
import numpy as np
import torch
import seaborn as sns
from torch import nn
from torch import optim
from collections import Counter
from itertools import chain
from progressbar import ProgressBar
#from ipdb import launch_ipdb_on_exception
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
import json
from gensim.models import KeyedVectors
# import dill
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import ParameterGrid
from torch.nn.init import xavier_uniform_

parser = argparse.ArgumentParser(description='MASK_LSTM text classificer')
parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('-num_epoch', type=int, default=30, help='epochs')
parser.add_argument('-batch_size', type=int, default=8, help='batch size')
parser.add_argument('-d_t', type=int, default=50, help='number of topics in code')
parser.add_argument('-output_gate', type=float, default=0.5, help='proportion of the aggregate and original input')
parser.add_argument('-cuda', type=int, default=3, help='gpu id')
parser.add_argument('-pretrain_emb', type=str, default=1, help='use glove or googlenews word embedding')
parser.add_argument('-tsoftmax', type=str, default=1, help='the temperature of softmax in co_attention_weight')
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
    tfidf_list = []
    for w in words:
        w = w.strip('\'"?!,.():;')
        w = w.lower()
        if use_stem:
            porter_stemmer = PorterStemmer()
            w = str(porter_stemmer.stem(w))
        if w in vocab:#add the tf-idf list, the same length as id_list
            id_list.append(vocab[w])
        else:
            id_list.append(unk)

    if sos is not None:
        id_list.insert(0, sos)
    if eos is not None:
        id_list.append(eos)
    return id_list[:max_value]

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
    norm_arr=list(arr/arr.sum(axis=0))
    return tfidf_list[:max_value]

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
    # logging.info("\nThe hit rate is :{0:8d}".format(hit_rate))
    print(("The hit rate is {}".format(hit_rate)))
    return embedding_matrix
 
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



class DataIter(object):
    def __init__(self, document_list, label_list, tfidf_list,batch_size, padded_value):
        self.document_list = document_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.padded_value = padded_value
        self.batch_starting_point_list = self._batch_starting_point_list()
        self.tfidf_list = tfidf_list

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
        tfidf_batch = self.tfidf_list[batch_starting:batch_end]
        transeposed_batch = map(list, zip(*raw_batch))
        transeposed_tfidf_batch = map(list,zip(*tfidf_batch))
        padded_batch = []
        length_batch = []
        padded_tfidf_batch = []
        for transeposed_doc,transeposed_tfidf in zip(transeposed_batch,transeposed_tfidf_batch): #padding for each batch data.
            length_list = [len(sent) for sent in transeposed_doc]
            max_length = max(length_list)
            new_doc = [sent+[self.padded_value]*(max_length-len(sent)) for sent in transeposed_doc]
            new_tfidf = [sent+[self.padded_value]*(max_length-len(sent)) for sent in transeposed_tfidf]
            padded_batch.append(np.asarray(new_doc, dtype=np.int32).transpose(1,0))
            padded_tfidf_batch.append(np.asarray(new_tfidf, dtype=np.float32).transpose(1,0))
            length_batch.append(length_list)
        padded_length = np.asarray(length_batch)
        padded_label = np.asarray(label_batch, dtype=np.int32) -1
        original_index =  np.arange(batch_starting,batch_end)
        self.batch_index += 1
        return padded_batch, padded_label, padded_tfidf_batch,padded_length ,original_index

class HierachicalClassifier(nn.Module):
    def __init__(self, d_t, num_word, emb_size, word_rnn_size, word_rnn_num_layer, word_rnn_dropout, word_rnn_bidirectional,word_attention_size, 
                context_rnn_size, context_rnn_num_layer, context_rnn_dropout, context_rnn_bidirectional, context_attention_size, mlp_size, num_label,output_gate, pretrained_embedding=None,tsoftmax=1,mask_k=10,alpha=1.0):
        self.emb_size = emb_size
        self.d_t = d_t
        self.output_gate = output_gate
        self.word_rnn_size = word_rnn_size
        self.word_rnn_num_layer = word_rnn_num_layer
        self.word_rnn_bidirectional = word_rnn_bidirectional
        self.context_rnn_size = context_rnn_size
        self.context_rnn_num_layer = context_rnn_num_layer
        self.context_rnn_bidirectional = context_rnn_bidirectional
        self.num_label = num_label
        self.tsoftmax = tsoftmax
        super(HierachicalClassifier, self).__init__()
        self.embedding = nn.Embedding(num_word, emb_size)
        self.word_dropout = nn.Dropout(word_rnn_dropout)
        self.word_rnn = nn.GRU(input_size = emb_size, hidden_size = word_rnn_size,
                num_layers = word_rnn_num_layer, bidirectional = word_rnn_bidirectional)
        word_rnn_output_size = word_rnn_size * 2 if word_rnn_bidirectional else word_rnn_size
        self.word_conv_attention_layer = nn.Conv1d(emb_size, word_attention_size, 3, padding=2, stride=1)
        self.word_conv_attention_linear = nn.Linear(word_attention_size, 1, bias=False)
        #the same attention for aspect
        self.aspect_conv_attention_layer = nn.Conv1d(emb_size, word_attention_size, 3, padding=2, stride=1)
        self.aspect_conv_attention_linear = nn.Linear(word_attention_size, 1, bias=False)

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
        self.graph_linear_trans = nn.Linear(context_rnn_output_size,context_rnn_output_size,bias=True)
        self.classifier = nn.Sequential(nn.Linear(context_rnn_output_size, mlp_size),
                                        nn.LeakyReLU(),
                                        nn.Linear(mlp_size, num_label),
                                        nn.Tanh())
        self.device = args.cuda
        self.query_linear = nn.Linear(self.d_t,self.d_t)
        self.key_linear = nn.Linear(self.d_t,self.d_t)
        self.mask_k = mask_k
        self.var_scale =1.0

        #VAE part for encoder-decoder
        if np.array(alpha).size == 1:
            # if alpha is a scalar, create a symmetric prior vector
            self.alpha = alpha * np.ones((1, self.emb_size)).astype(np.float32)
        else:
            # otherwise use the prior as given
            self.alpha = np.array(alpha).astype(np.float32)
            assert len(self.alpha) == self.emb_size

        self.encoder_dropout_layer = nn.Dropout(p=0.2)
        # create the mean and variance components of the VAE
        self.mean_layer = nn.Linear(self.emb_size,self.emb_size )
        self.logvar_layer = nn.Linear(self.emb_size, self.emb_size)

        self.mean_bn_layer = nn.BatchNorm1d(self.emb_size, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.emb_size))).to(self.device)
        self.mean_bn_layer.weight.requires_grad = False
        self.logvar_bn_layer = nn.BatchNorm1d(self.emb_size, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.emb_size))).to(self.device)
        self.logvar_bn_layer.weight.requires_grad = False

        self.z_dropout_layer = nn.Dropout(p=0.2)


        # create the decoder
        self.beta_layer = nn.Linear(self.d_t, self.emb_size)

        xavier_uniform_(self.beta_layer.weight)
        # if bg_init is not None:
        #     self.beta_layer.bias.data.copy_(torch.from_numpy(bg_init))
        #     self.beta_layer.bias.requires_grad = False
        self.beta_layer = self.beta_layer.to(self.device)

        if pretrained_embedding is not None:
            self.embedding.weight.data = self.embedding.weight.data.new(pretrained_embedding)

        # create a final batchnorm layer
        self.eta_bn_layer = nn.BatchNorm1d(self.emb_size, eps=0.001, momentum=0.001, affine=True).to(self.device)
        self.eta_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.emb_size)).to(self.device))
        self.eta_bn_layer.weight.requires_grad = False

        # create the document prior terms
        prior_mean = (np.log(self.alpha).T - np.mean(np.log(self.alpha), 1)).T #
        prior_var = (((1.0 / self.alpha) * (1 - (2.0 / self.emb_size))).T + (1.0 / (self.d_t * self.emb_size)) * np.sum(1.0 / self.alpha, 1)).T

        prior_mean = np.array(prior_mean).reshape((1, self.emb_size))
        prior_logvar = np.array(np.log(prior_var)).reshape((1, self.emb_size))
        self.prior_mean = torch.from_numpy(prior_mean).to(self.device)
        self.prior_mean.requires_grad = False
        self.prior_logvar = torch.from_numpy(prior_logvar).to(self.device)
        self.prior_logvar.requires_grad = False

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
# remove_data_var_list, length,id2vocab=id2vocab,evalute_wordatt = True,mask_model=mask_model
    def forward(self, input_list,input_tfidf, length_list,id2vocab, evalute_wordatt,mask_model='ori'):
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
        kld_loss = torch.zeros(batch_size).cuda(self.device)
        removed_words_l=[]
        for utterance_index in range(num_utterance):
            word_rnn_input = self.embedding(input_list[utterance_index])
            word_rnn_input = self.word_dropout(word_rnn_input) #[seq_len,bs,emb_size]
            sent_tfidf = input_tfidf[utterance_index] #[seq_len,bs]
            tfidf_word_rnn_input = sent_tfidf.unsqueeze(2).repeat(1,1,word_rnn_input.shape[-1])*word_rnn_input

            word_rnn_output, word_rnn_hidden = self.word_rnn(word_rnn_input, word_rnn_hidden)


            word_attention_weight = self.word_conv_attention_layer(word_rnn_input.permute(1,2,0))#
            word_attention_weight = word_attention_weight[:,:,1:-1]
            word_attention_weight = word_attention_weight.permute(2, 0 ,1)
            word_attention_weight = self.word_conv_attention_linear(word_attention_weight)

            word_attention_weight = nn.functional.relu(word_attention_weight)
            word_attention_weight = nn.functional.softmax(word_attention_weight, dim=0)
            word_rnn_last_output = torch.mul(word_rnn_output,word_attention_weight).sum(dim=0) 
            
            #sentiment representation of the sentence,s
            word_rnn_output_list.append(word_rnn_last_output)
            word_rnn_hidden = word_rnn_hidden.detach()


            # aggregate the sentence represntation by sent_tfidf [bs,seq_len] [bs,seq_len,dim]
            encoder_output = F.softplus(tfidf_word_rnn_input) #word embeddings
            encoder_output_do = self.encoder_dropout_layer(encoder_output)#[seq_len,bs,emb_size]
            # compute the mean and variance of the document posteriors
            posterior_mean = torch.transpose(self.mean_layer(encoder_output_do),1,0) #[seq_len,bs,emb_size]
            posterior_logvar = torch.transpose(self.logvar_layer(encoder_output_do),1,0)#[seq_len,bs,emb_size]

            posterior_mean_bn = self.mean_bn_layer(torch.transpose(posterior_mean,1,2)).permute(0,2,1)#the batchnorm1d, input should be [N,C,L]
            posterior_logvar_bn = self.logvar_bn_layer(torch.transpose(posterior_logvar,1,2)).permute(0,2,1)

            posterior_var = posterior_logvar_bn.exp().to(self.device)

            # sample noise from a standard normal
            eps = encoder_output_do.data.new().resize_as_(posterior_mean_bn.data).normal_().to(self.device)
            #[]

            # compute the sampled latent representation
            z = posterior_mean_bn + posterior_var.sqrt() * eps * self.var_scale
            z_do = self.z_dropout_layer(z) #[N,C]#latentc

            # pass the document representations through a softmax
            theta = F.softmax(z_do, dim=1)#this is omega [bs,300]

            # expand_theta = theta.unsqueeze(0).repeat(word_rnn_input.shape[0],1,1)
            word_aspect_output = (theta.transpose(1,0)*word_rnn_input).sum(0) #[seq_len, bs,dim]
            #use new_word_rnn_input as the previous word_rnn_input
            word_aspect = self.topic_encoder(word_aspect_output)#h
            recons_word = self.topic_decoder(word_aspect)#z'
            r = nn.functional.normalize(recons_word)
            z = nn.functional.normalize(word_aspect_output)
            n = nn.functional.normalize(word_rnn_last_output)

            y = torch.ones(batch_size).cuda(self.device) - torch.sum(r*z, 1) + torch.sum(r*n, 1) #
            word_aspect_output_list.append(word_aspect)
            aspect_loss += nn.functional.relu(y) #why use a relu
            word_rnn_hidden = word_rnn_hidden.detach() 

            prior_mean   = self.prior_mean.expand_as(posterior_mean)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)

            do_average = False
            KLD = self._loss(prior_mean, prior_logvar, posterior_mean_bn, posterior_logvar_bn, do_average)
            kld_loss+=nn.functional.relu(KLD)
            

        context_rnn_hidden = self.init_rnn_hidden(batch_size, level="context")
        context_rnn_input = torch.stack(word_rnn_output_list, dim=0)
        context_rnn_input = self.context_dropout(context_rnn_input)
        context_rnn_output,context_rnn_hidden = self.context_rnn(context_rnn_input, context_rnn_hidden)#[seq_len,bs,dim]

        context_attention_weight = self.context_conv_attention_layer(context_rnn_input.permute(1,2,0))#[bs,dim,seq_len]
        context_attention_weight = context_attention_weight.permute(2, 0, 1)#[seq_len,bs,dim]
        context_attention_weight = nn.functional.relu(context_attention_weight)
        context_attention_weight = self.context_conv_attention_linear(context_attention_weight)

        word_aspect = torch.stack(word_aspect_output_list, dim=0)#[seq_len, bs,d_t]
        context_topic_weight = self.context_topic_attention_linear(word_aspect.squeeze(1)) #[seq_len,bs,d_t]

        context_topic_weight = context_topic_weight.permute(1,2,0)#[bs,d_t,seq_len]
        context_attention_weight = context_attention_weight.permute(1,0,2) #[bs,seq_len,d_t]

        context_topic_weight = context_topic_weight.permute(0,2,1) #[bs,doc_len,d_t]
        context_rnn_output = context_rnn_output.permute(1,0,2)#[bs,doc_len,dim]
        strategy = 'raw' #
        if strategy == 'raw':
            topic_weight_matrix =  context_topic_weight
        elif strategy == 'normlized_topic':
            topic_norm = torch.norm(context_topic_weight, p=2, dim=-1, keepdim=True)
            norm_topic_weight = context_topic_weight/topic_norm
            topic_weight_matrix = norm_topic_weight

        co_topic_weight = nn.functional.softmax(torch.bmm(self.key_linear(topic_weight_matrix),self.query_linear(topic_weight_matrix).transpose(2,1)/self.tsoftmax),2)

        update_context_rep = torch.bmm(co_topic_weight,context_rnn_output)
        # update_context_rep = self.output_gate*torch.bmm(co_topic_weight,context_rnn_output)+(1-self.output_gate)*context_rnn_output #[bs,doc_len,dim]
        use_nolinear = False #apply non-linear to the current graph layer to update the next layer
        if use_nolinear:
            update_context_rep = nn.functional.relu(update_context_rep)
        mean_context_output = torch.mean(update_context_rep,dim=1)
        classifier_input = mean_context_output
        # context_rnn_last_output = torch.mean(context_rnn_transform,1) #average the 
        # classifier_input = context_rnn_last_output
        classifier_input_array = np.array(classifier_input.cpu().data)
        logit = self.classifier(classifier_input) 
        #attention_weight_array = np.array(context_attention_weight.data.cpu().squeeze(-1)).transpose(1,0)
        attention_weight_array = 0
        return logit,attention_weight_array,classifier_input_array,aspect_loss,co_topic_weight,kld_loss

    def _loss(self, prior_mean, prior_logvar, posterior_mean, posterior_logvar,do_average=True):

        # compute KLD
        prior_var = prior_logvar.exp()
        posterior_var = posterior_logvar.exp()
        var_division    = posterior_var / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar

        # put KLD together
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.emb_size)


        return KLD.sum(1)

def evaluate(model,loss_function,batch_generator,id2vocab,cuda=None,mask_model='ori',topic_words=None):
    model.eval()
    total_loss,total_kld_loss = 0,0
    total_hit = 0
    total_sample = 0
    batch_i = 0
    true_label_list = []
    predicted_label_list = []
    attention_weight_list = []
    classifier_input_list = []
    original_index_list = []
    senatt_arr_list = []
    with torch.no_grad():
        for batch in batch_generator:
            data, target, tfidf,length, original_index = batch[0], batch[1], batch[2], batch[3],batch[4]
            if cuda is None:
                data_var_list = [torch.LongTensor(chunk) for chunk in data]
                target_var = torch.LongTensor(target)
                # remove_data_var_list = [torch.LongTensor(chunk) for chunk in remove_data_list]
                tfidf_var_list = [torch.tensor(chunk) for chunk in tfidf]
                length_var = torch.LongTensor(length)
            else:
                data_var_list = [torch.LongTensor(chunk).cuda(cuda) for chunk in data]
                target_var = torch.LongTensor(target).cuda(cuda)
                tfidf_var_list = [torch.tensor(chunk).cuda(cuda) for chunk in tfidf]
                length_var = torch.LongTensor(length)
            if mask_model=='topic':
                predicted_target,attention_weight,classifier_input,_,senatt_arr,kld_loss= model(remove_data_var_list, tfidf_var_list,length,id2vocab=id2vocab,evalute_wordatt = True,mask_model=mask_model)
                # removed_words_batch = removed_words_batch
            else:
                predicted_target,attention_weight,classifier_input,_,senatt_arr,kld_loss = model(data_var_list,tfidf_var_list,length,id2vocab=id2vocab,evalute_wordatt = True,mask_model=mask_model)
            # senatt_arr_list.append(senatt_arr)
            loss = loss_function(predicted_target, target_var)
            _,predicted_label = torch.max(predicted_target,dim=1)
            total_hit += torch.sum(predicted_label.data == target_var.data)
            total_loss += loss.item()
            total_kld_loss +=kld_loss.mean().item()
            total_sample += data[0].shape[1]
            true_label_list.append(target)
            predicted_label_array_batch = np.array(predicted_label.data.cpu())
            predicted_label_list.append(predicted_label_array_batch)
            # attention_weight_list.append(attention_weight)
            # classifier_input_list.append(classifier_input)
            original_index_list.append(original_index)
            # removed_words_l.append(removed_words_batch)
            batch_i += 1

    true_label_array = np.asarray(list(chain(*true_label_list)),dtype=np.int32)
    predicted_label_array = np.asarray(list(chain(*predicted_label_list)),dtype=np.int32)
    original_index_array = np.asarray(list(chain(*original_index_list)),dtype=np.int32)
    acc = float(total_hit)/float(total_sample)
    returned_document_list = [batch_generator.sample_document(index) for index in original_index_array]
    model.train()
    return total_loss/(batch_i+1),acc,true_label_array,predicted_label_array,returned_document_list,total_kld_loss/(batch_i+1)

def vis_senatt(document_list,id2word, senatt, nbatch,acc):
    #flatten the senatt
    senatt_list = []
    for batch_i in range(len(senatt)):
        senatt_arr = senatt[batch_i].cpu().detach().numpy()
        for sample_i in range(senatt_arr.shape[0]):
            senatt_list.append(senatt_arr[sample_i])
    #save the doc_text and the senatt_pic
    doc_file = "./imdb_output/senatt_vis/tsoftmax08/senatt_doc"+"epoch-"+str(nbatch)+"acc-"+str(format(acc,'.4f'))+".txt"
    output_doc = open(doc_file,'w')
    for doc_i, doc in enumerate(document_list):
        data = senatt_list[doc_i]
        ax = sns.heatmap(data, linewidth=0.3,annot=True)
        plt.savefig("./imdb_output/senatt_vis/tsoftmax08/"+"epoch-"+str(nbatch)+"doc"+str(doc_i)+'.png')
        plt.clf()
        output_doc.writelines(str(doc_i)+"\n")
        for sen in document_list[doc_i]:
            sen_text = [id2word[w] for w in sen]
            output_doc.writelines(" ".join(sen_text))
            output_doc.writelines(str(doc_i)+"\n")
    output_doc.close()

def vis_dec(id2word, indices_decoder,model,rank,indices_rank,epoch_i,sim_decoder):
    topic = ""
    weight = ''
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
            # weight = weight + " "+ sim_decoder[j,t-1]
            decoder_ranking.append(model.embedding.weight[j.item()].cpu().detach().numpy())
        #for t-th topic, the attention weight for positive label: 
        print("decoder topic \#"+ str(t) + ":" + topic+ ". score: " + str(rank[t-1,0].item()) + ", " + str(rank[t-1,1].item()))
        print(sim_decoder[:,t-1])
        if rank[t-1,0].item()>rank[t-1,1].item():
            print("Neg")
        else:
            print("Pos")
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
    loss_kld_total = 0
    log_loss = open('loss.txt', 'a')
    best_dev_acc = 0
    # evalute_wordatt = False
    times = 0
    for epoch_i in range(num_epoch):
        logging.info("Epoch {}".format(epoch_i))
        for train_batch in train_batch_generator:
        # for stidx in range(0, len(data_label), 50):
            temp_batch_index += 1
            # print("temp_batch_index is%.4f"%temp_batch_index)
            train_data,train_target,train_tfidf,length_data = train_batch[0],train_batch[1],train_batch[2],train_batch[3] 
            if cuda is None:
                train_data_var_list = [torch.LongTensor(chunk) for chunk in train_data]
                train_target_var = torch.LongTensor(train_target)
                train_tfidf_var_list = [torch.tensor(chunk) for chunk in train_tfidf]
                length_var = torch.LongTensor(length_data)
            else:
                train_data_var_list = [torch.LongTensor(chunk).cuda(cuda) for chunk in train_data]
                train_target_var = torch.LongTensor(train_target).cuda(cuda)
                train_tfidf_var_list = [torch.tensor(chunk).cuda(cuda) for chunk in train_tfidf]
                length_var = torch.LongTensor(length_data)
            id2word = {vocab[key]:key for key in vocab.keys()}
            predicted_train_target,_,_,aspect_loss,senatt,train_kld_loss = model(train_data_var_list,train_tfidf_var_list,length_var,id2word, evalute_wordatt=False)
            optimizer.zero_grad()
            loss_C = loss_function(predicted_train_target,train_target_var)
            loss_A = aspect_loss.mean()
            loss_R = torch.norm(torch.eye(d_t).cuda(cuda) - torch.mm(model.topic_encoder.weight, model.topic_encoder.weight.t()))
            loss_kld = train_kld_loss.mean()
            loss_C_total += loss_C
            loss_A_total += loss_A
            loss_R_total += loss_R
            loss_kld_total += loss_kld/100
            loss = loss_C + 0.05 * loss_A +  0.01 * loss_R
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
        train_loss,train_acc = 0,0
        C_loss = loss_C_total/(1000)
        A_loss = loss_A_total/(1000)
        R_loss = loss_R_total
        kld_loss = loss_kld_total/(temp_batch_index)
        log_loss.write("{0:6f},{1:6f},{2:6f}\n".format(C_loss,A_loss,R_loss))
        loss_A_total, loss_C_total, loss_R_total = 0,0,0
        id2word = {vocab[key]:key for key in vocab.keys()}

        test_loss,test_acc,true_label_array,predicted_label_array,_,test_kld_loss = evaluate(model,loss_function,test_batch_generator,id2word,cuda,mask_model='ori',topic_words=None)#
        
        # print(torch.std(senatt[0]))

        _,predicted_label = torch.max(predicted_train_target,dim=1)
        train_hits= torch.sum(predicted_label.data == train_target_var.data)
        train_acc = train_hits/len(predicted_label)
        logging.info("\nEpoch:{0:8d}\ntrain_loss:{1:0.6f}\ttrain_acc:{2:0.6f}\ttrain_kld:{3:0.6f}\ntest_loss:{4:0.6f}\ttest_kld_loss{5:0.6f}\ttest_acc:{6:0.6f}".format(epoch_i, C_loss,train_acc.data,kld_loss,test_loss,test_kld_loss,test_acc))

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

        if test_acc>best_dev_acc and test_acc>0.87:
            # vis_dec(id2word, indices_decoder,model,rank,indices_rank,epoch_i, sim_decoder)
            best_dev_acc = test_acc
            print("higher dev acc: %.4f"%best_dev_acc)
            topic_words = np.unique(indices_decoder.cpu().detach().numpy()).tolist()
            topic_test_loss,topic_test_acc,true_label_array,topic_predicted_label_array,remove_topic_words,returned_document_list = evaluate(model,loss_function,test_batch_generator,id2word,cuda,mask_model='topic',topic_words=topic_words)#
            context_test_loss,context_test_acc,true_label_array,context_predicted_label_array,remove_context_words,returned_document_list= evaluate(model,loss_function,test_batch_generator,id2word,cuda,mask_model='context',topic_words=None)#
            tfidf_test_loss,tfidf_test_acc,true_label_array,tf_idf_predicted_label_array,remove_tfidf_words,returned_document_list= evaluate(model,loss_function,test_batch_generator,id2word,cuda,mask_model='tfidf',topic_words=topic_words)#
            logging.info("\nEpoch :{0:8d}\ntopic_acc:{1:0.6f}\tcontext_acc:{2:0.6f}\ntfidf_acc:{3:0.6f}".format(epoch_i, topic_test_acc,context_test_acc,tfidf_test_acc))
            #write the label/pre/removed_words/ori_doc into the files for comparing
            # labels_file = open("imdb_label_pre.txt",'w')
            # rwords_file = open("imdb_rwords.txt",'w')
            # doc_file = open('imdbtest_doc.txt','w')
            # labels_file.writelines("label"+"\t"+"topic_pre"+"\t"+"context_pre"+"\t"+"tfidf_pre"+"\n")
            # rwords_file.writelines("context_words"+"\t"+"tfidf_words"+"\n")
            # true_label_array.shape(24700,) =len(return_document_list)
            # for batch_i in range(len(remove_context_words)):
            #     re_context, retfidf=[],[]
            #     for sent_i in range(len(remove_context_words[batch_i][0])):
            #         recontext_words = [id2word[word]  for word in remove_context_words[batch_i][0][sent_i] if word in id2word.keys()]
            #         retfidf_words = [id2word[word] for word in remove_tfidf_words[batch_i][0][sent_i] if word in id2word.keys()]
            #         re_context.extend(recontext_words)
            #         retfidf.extend(retfidf_words)
            #     rwords_file.writelines(" ".join(recontext_words)+"\t"+" ".join(retfidf_words))
            # rwords_file.close()

            # for idx in range(len(returned_document_list)):
            #     doc = ""
            #     doc_file.writelines(str(idx)+"\n")
            #     labels_file.writelines(str())
            #     for sent_id in range(len(returned_document_list[idx])):
            #         sents = [id2word[word] for word in returned_document_list[idx][sent_id]]
            #         doc +=" ".join(sents)+"\t"
            #     doc_file.writelines(doc+"\n")
            #     labels_file.writelines(str(true_label_array[idx])+"\t"+str(predicted_label_array[idx])+"\t"+str(topic_predicted_label_array[idx])+"\t"+str(context_predicted_label_array[idx])+"\t"+str(tf_idf_predicted_label_array[idx])+"\n")
            # labels_file.close()
            # doc_file.close()
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

def cal_tfidf(inputfile,dataset,vocab):#each document is a dict, save the word's tf-idf frequency
    contents = open(inputfile).readlines()
    corpus = []
    for line in contents:
        if dataset=='yelp':
            _,_,label,text = line.strip().split("\t\t") #2 pos, 1neg
            sents = text.strip().split("<sssss>")
            corpus.append(" ".join(sents))
        else:
            # label = imbd_dic[json.loads(line)['sentiment']]
            text = json.loads(line)['text']
            sents = [x for i in text.strip().split('.') for x in i.split('<br /><br />')]
            corpus.append(" ".join(sents))
    vectorizer = TfidfVectorizer(vocabulary=vocab)#use the vocab in original code
    tfidf_matrix = vectorizer.fit_transform(corpus) #[doc_num, voc]
    feature_names=vectorizer.get_feature_names()
    doc_tfidf = []
    for doc_id in range(tfidf_matrix.shape[0]):
        feature_index = tfidf_matrix[doc_id,:].nonzero()[1]
        tfidf_scores = zip([feature_names[i] for i in feature_index], [tfidf_matrix[doc_id, x] for x in feature_index])
        doc_tfidf.append(dict(tfidf_scores))
    with open('tfidf_weight/norm_news_tfidf_train.json', 'w') as fout:
        json.dump(doc_tfidf, fout)
    print("save the tfidf dict to file!!!")
  

def main():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s\t%(message)s")
    dataset = 'imdb'
    if dataset=="yelp":
        train_data_file = "medical_train.txt"
        test_data_file = "medical_test.txt"
    elif dataset == 'imdb':
        train_data_file = "./input_data/imdb/train.jsonlist"
        test_data_file = "./input_data/imdb/test.jsonlist"
    elif dataset =="news":
        train_data_file = './input_data/guadian_news/train_news_data.txt'
        test_data_file = './input_data/guadian_news/test_news_data.txt'

    vocab = get_vocabulary(train_data_file,vocabsize=15000,dataset=dataset,use_stem=False)
    # calculate the tf-idf value for the corpus
    # cal_tfidf(train_data_file,'news',vocab)
    train_tfidffile=json.loads(open("tfidf_weight/norm_imdb_tfidf_train.json").readlines()[0])
    test_tfidffile=json.loads(open("tfidf_weight/norm_imdb_tfidf_test.json").readlines()[0])
    # doc_tfidf=json.loads(open("imdb_tfidf.json").readlines()[0])
    train_data,train_label,train_tfidf = load(train_data_file,vocab,max_value=60,max_utterance=10,dataset=dataset,doc_tfidf=train_tfidffile)
    test_data,test_label,test_tfidf = load(test_data_file,vocab,max_value=60,max_utterance=10,dataset=dataset,doc_tfidf=test_tfidffile)
    pretrain_emb = 'glove' #glove or googlenews

    if pretrain_emb == 'googlenews':
        pretrained_embedding,hit_rate = getVectors(embed_dim=300, wordvocab=vocab)
    else:
        pretrained_embedding = load_embedding(vocab,"/mnt/sda/media/Data2/hanqi/sine/glove.840B.300d.txt",embedding_size=300)

    grid_search = {}
    params = {
        "batch_size": [8],
        "d_t":[30],
        "lr":[1e-4],
        "num_epoch":[30],
        "cuda":[0],
        'num_word':[15000], #time-consuming!
        'pretrain_emb':[pretrain_emb],
        'tsoftmax':[0.8],
        'mask_k':[10]
        }
    params_search = list(ParameterGrid(params))
    acc_list =[]
    for i, param in enumerate(params_search):
        print("*************params_search_{}*************".format(i + 1))
        print(param)
        for key, value in param.items():
            setattr(args, key, value)
        # p_list, r_list, f1_list = [], [], []
        train_batch = DataIter(train_data,train_label,train_tfidf,args.batch_size,args.cuda)#ori_code
        test_batch = DataIter(test_data,test_label,test_tfidf,args.batch_size,args.cuda)#ori_code
        model = HierachicalClassifier(d_t=args.d_t, num_word=15000, emb_size=300, word_rnn_size=150, word_rnn_num_layer=1, word_rnn_dropout = 0.4, word_rnn_bidirectional=True,
            word_attention_size =150, context_rnn_size=150, context_rnn_dropout = 0.3, context_rnn_bidirectional=True,
            context_attention_size=200, mlp_size = 200, num_label = 2, context_rnn_num_layer=1, output_gate=args.output_gate,pretrained_embedding=pretrained_embedding,tsoftmax=args.tsoftmax,mask_k=args.mask_k)
        print(model.parameters())
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
