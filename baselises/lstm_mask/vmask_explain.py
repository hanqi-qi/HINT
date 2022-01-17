"""Load the pretrained model to generate interpretations"""
import argparse
import torch
from torch import nn
from mask_lstm_model import MASK_LSTM
from getvectors import getVectors2
from itertools import chain
import os
import numpy as np
import random
import dill
import pandas as pd
import copy
import torch.optim as optim
from collections import Counter
import json
from progressbar import ProgressBar
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

parser = argparse.ArgumentParser(description='MASK_LSTM text classificer')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('-beta', type=float, default=1, help='beta')
parser.add_argument('--weight_decay', default=0, type=float, help='adding l2 regularization')
parser.add_argument('--clip', type=float, default=1, help='gradient clipping')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-dropout', type=float, default=0.2, help='the probability for dropout')
parser.add_argument('-embed-dim', type=int, default=300, help='original number of embedding dimension')
parser.add_argument('-lstm-hidden-dim', type=int, default=100, help='number of hidden dimension')
parser.add_argument('-lstm-hidden-layer', type=int, default=1, help='number of hidden layers')
parser.add_argument('-mask-hidden-dim', type=int, default=300, help='number of hidden dimension')
parser.add_argument("--max_sent_len", type=int, dest="max_sent_len", default=250, help='max sentence length')
parser.add_argument("--activation", type=str, dest="activation", default="tanh", help='the choice of \
        non-linearity transfer function')
parser.add_argument("--doc_id",type=int, default=0)
parser.add_argument('--save', type=str, default='masklstm.pt', help='path to save the final model')
parser.add_argument('--mode', type=str, default='static', help='available models: static, non-static')
parser.add_argument('--gpu', default=0, type=int, help='0:gpu, -1:cpu')
parser.add_argument('--gpu_id', default="3", type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--use_lstmatt', type=bool, default=False, help='random seed')
parser.add_argument('--mask_k', type=int, default=30, help='the number of masked words')
parser.add_argument('--ifmask', type=bool, default=True, help='whether mask the words or softmax')
parser.add_argument("--dataset_name",type=str, default="eraser_movie",help="dataset name")
parser.add_argument("--class_num",type=int, default=2,help="number of labels")
parser.add_argument("--output_file",type=str, default="/home/hanqiyan/latent_topic/VMASK/lstm_VMASK/output/explainations.txt")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

class input_example:
    def __init__(self):
        self.pos_idx = None
        self.word = None
        self.voc_idx = None
    def feed(self,id,word,vid):
        self.id = id
        self.word = word
        self.vid = vid

def convert_words2ids_new(words, vocab, unk, max_value, sos=None, eos=None,use_stem=False,stopwords=None):
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

    # if sos is not None:
    #     id_list.insert(0, sos)

    # if eos is not None:
    #     for pad_i in range(max_value-len(id_list)): #pad to max_value with '<eos>'
    #         id_list.append(eos)
    #only return the sentence longer than 3 words
    if isinstance(item,input_example):
        if len(resultx) > 1:
            return resultx
    else:
        if len(id_list) > 1:
            result = input_example()
            result.feed(id_list,word_list)
            return result

def load_rats(rat_file,vocab,max_value=60,max_utterance=50,dataset=None,type="original",stopwords=None):    
    eraser_dict  =  {"NEG":"1","POS":"2"}
    document_list = []
    document_tfidf_list = []
    label_list = []
    doc_idx_list = []
    word_pos_list = []
    doc_length = []
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
            text  = " ".join(text)
            sen_len = 0
            #couple the idx and word
            doc_list =  []
            for idx, wid in enumerate(text.split()):
                resultx = input_example()
                resultx.word = wid
                resultx.pos_idx = idx
                doc_list.append(resultx)
            assert len(text.split()) == len(doc_list)
            tokenids = []
            doc_sample = convert_words2ids_new(doc_list,vocab,max_value=max_value,unk=vocab['<unk>'],use_stem=False,eos=vocab['<eos>'],stopwords=stopwords)
            doc_length.append(len(doc_sample))
            if len(doc_sample) == 0:
                continue
            else:
                #post-processing
                doc_word_list =  []
                doc_wordwpos_list = []
                for i in range(max_value):
                    if len(doc_sample)>i+1:
                        doc_word_list.append(doc_sample[i].voc_idx)
                        doc_wordwpos_list.append(doc_sample[i].pos_idx)
                    else:
                        doc_word_list.append(vocab['<eos>'])
                        doc_wordwpos_list.append(999)
                assert len(doc_word_list) == len(doc_wordwpos_list) == max_value

                document_list.append(torch.tensor(np.array(doc_word_list[:max_value]).reshape(1,max_value)))
                label_list.append(torch.reshape(torch.tensor(int(label)-1),[1]))
                doc_idx_list.append(filename)
                word_pos_list.append(np.array(doc_wordwpos_list[:max_value]).reshape(1,max_value))
    return document_list,label_list,doc_idx_list,word_pos_list,doc_length

def convert_words2ids(words, vocab, unk, max_value, sos=None, eos=None):
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
    id_list = [ vocab[w] if w in vocab else unk for w in words ]
    if sos is not None:
        id_list.insert(0, sos)
    if eos is not None:
        for pad_i in range(max_value-len(id_list)): #pad to max_value with '<eos>'
            id_list.append(eos)
    return np.array(id_list[:max_value]).reshape(1,max_value)


def load_bin_dataset(metric,num_bin,vocab,max_value):
    document_list = []
    document_tfidf_list = []
    label_file = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/VMASK/original_set/label.txt").readlines()
    label_list=[]
    idx_list =  []
    if metric == "sset":
        file_dir =  "/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/VMASK/sset"
        bin_dir = "bin"+str(num_bin)
        target_dir = file_dir+"/"+bin_dir+"/"
    elif metric == "cset":
        file_dir =  "/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/VMASK/cset"
        bin_dir = "bin"+str(num_bin)
        target_dir = file_dir+"/"+bin_dir+"/"
    elif metric ==  "original":
        file_dir = "/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/VMASK/original_set/"
        target_dir = file_dir
    def filter_key(sent):
        unk_count = sent.count(vocab['<unk>'])
        return unk_count/len(sent) < 0.3
    for file_num in range(1,200):
        text = open(target_dir+str(file_num)+".txt","r").readlines()
        idx = file_num
        label = int(label_file[file_num-1])+1
        sent_list = [x.strip() for x in text if len(x.split())>1] 
        sents =  " ".join(sent_list)
        doc_list = [word for word in sents.split(" ")]
        doc_array = convert_words2ids(doc_list,vocab,max_value=max_value,unk=vocab['<unk>'], sos=None,eos=vocab['<eos>'])
        label_list.append(torch.reshape(torch.tensor(int(label)-1),[1]))
        document_list.append(torch.tensor(doc_array))
        idx_list.append(idx)

    return  document_list,label_list,idx_list

def load(textfile, vocab, max_value,dataset='yelp'):
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
    eraser_dict  =  {"NEG":"1","POS":"2"}
    def filter_key(sent):
        unk_count = list(sent).count(vocab['<unk>'])
        return unk_count/len(sent) < 0.3

    with open(textfile, "r") as f:
        line_list = f.readlines()
        line_len = len(line_list)
        random_index = np.random.permutation(line_len)
        line_list = [line_list[index] for index in random_index]
        idx_list = []
        for doc_id,line in enumerate(line_list):
            if dataset=='yelp':
                _,_,label,sents = line.strip().split("<h>") #2 pos, 1neg
                # sents = text.strip().split("<sssss>")
            elif dataset=='imdb':
                label = imbd_dic[json.loads(line)['sentiment']]
                sents = json.loads(line)['text'].strip()
                # sents = [x for i in text.strip().split('.') for x in i.split('<br /><br />')]
            elif dataset=='news':
                sents,label = line.strip().split("\t")
                label = int(label)+1
                # sents = text.strip().split(".")
            elif dataset=='eraser_movie':
                label = eraser_dict[json.loads(line)['classification']]
                filename = json.loads(line)['annotation_id']
                idx = doc_id
                idx_list.append(idx)
                text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/docs/"+filename,"r").readlines()
                sent_list = [x.strip() for x in text if len(x.split())>1] 
                sents =  " ".join(sent_list)
            doc_list = [word for word in sents.split(" ")]
            # sent_list = list(filter(lambda x:x!=["."],sent_list))
            doc_array = convert_words2ids(doc_list,vocab,max_value=max_value,unk=vocab['<unk>'], sos=None,eos=vocab['<eos>'])
            label_list.append(torch.reshape(torch.tensor(int(label)-1),[1]))
            document_list.append(torch.tensor(doc_array))
    return document_list,label_list,idx_list

    
def get_vocabulary(textfile, initial_vocab={'<unk>':0,'<sssss>':1,'<eos>':2}, vocabsize=0,dataset=None,use_stem=False,stopwords=None):
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
            # w = w.strip('\'"?!,.():;')# will be a disaster
            w = w.lower()
            if w not in stopwords:
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


def random_seed():
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return

if args.gpu > -1:
    args.device = "cuda"
else:
    args.device = "cpu"

class B:
    text = torch.zeros(1).to(args.device)
    label = torch.zeros(1).to(args.device)



def batch_from_list(textlist, labellist):
    batch = B()
    batch.text = textlist[0]
    batch.label = labellist[0]
    for txt, la in zip(textlist[1:], labellist[1:]):
        batch.text = torch.cat((batch.text, txt), 0)
        # you may need to change the type of "la" to torch.tensor for different datasets, sorry for the inconvenience
        batch.label = torch.cat((batch.label, la), 0) # for SST and IMDB dataset, you do not need to change "la" type
    batch.text = batch.text.to(args.device)
    batch.label = batch.label.to(args.device)
    return batch


# evaluate
def evaluation(model, data_text, data_label, data_idx,data_wpos,flag,id2word,output_file,metric,k_bin,doc_length):
    model.eval()
    mask_acc, ori_acc, loss, size = 0,0, 0, 0
    count,total_hit = 0, 0
    pred_l, label_l,mask_l=[],[],[]
    test_sample_idx,neg_logits,pos_logits = [],[],[]
    doc_id=0
    rats_level = 0.02
    annotations =  []
    for stidx in range(0, len(data_label), args.batch_size):
        count += 1
        batch = batch_from_list(data_text[stidx:stidx + args.batch_size],
                                data_label[stidx:stidx + args.batch_size]
                                )
        batch_wpos = data_wpos[stidx:stidx + args.batch_size]
        batch_docidx = data_idx[stidx:stidx + args.batch_size]
        batch_len = doc_length[stidx:stidx + args.batch_size]
        test_sample_idx.extend(data_idx[stidx:stidx + args.batch_size])
        ori_pred,_,_,doc_id,annots = model(batch=batch, flag=flag,id2word=id2word,doc_id=doc_id,doc_wpos_id=batch_wpos,doc_name_list=batch_docidx,rats_level=rats_level,doc_len=batch_len)
        annotations.extend(annots)
        neg_logits.extend(list(np.array(ori_pred[:,0].detach().cpu().numpy())))
        pos_logits.extend(list(np.array(ori_pred[:,1].detach().cpu().numpy())))
        
        _, ori_pred_label = ori_pred.max(dim=1)
        total_hit += torch.sum(ori_pred_label.data == batch.label.data)
        size += len(ori_pred)
        pred_l.append(ori_pred_label.cpu().numpy())
        # mask_l.append(mask_pred)
        label_l.append(batch.label.cpu().numpy())

    true_label_array = np.asarray(list(chain(*label_l)),dtype=np.int32)
    pre_array = np.asarray(list(chain(*pred_l)),dtype=np.int32)

    if metric == "original":
        pre_results = pd.DataFrame({"idx":test_sample_idx,"base_pre":pre_array,"base_label":true_label_array,"base_neg_logits":neg_logits,"base_pos_logits":pos_logits})
    else:
        pre_results = pd.DataFrame({"idx":test_sample_idx,"pre":pre_array,"label":true_label_array,"neg_logits":neg_logits,"pos_logits":pos_logits})
    if metric == "gen_hard_rats":
        with open('/home/hanq1yanwarwick/SINE/explain_metric/VMASK_test_decoder_{}bin_eraser.json'.format(rats_level), 'w') as fout:
            json.dump(annotations , fout)
    else:
        pre_results.to_csv("/home/hanq1yanwarwick/SINE/eraser_metric/output_logits/VMASK/prediction_results_{}_{}bin.csv".format(metric,k_bin),index_label="idx",index=False)
        print("Predict Probalibility file of %s for %d bins is saved"%(metric,k_bin))
    ori_acc = total_hit.data/size
    print(ori_acc)

def main(metric,k_bin,dataset):
    #load dataset

    '''use the data from JSAE project'''
    if dataset=="yelp":
        train_data_file = "/home/hanqiyan/latent_topic/SINE/input_data/yelp/medical_train.txt"
        test_data_file = "/home/hanqiyan/latent_topic/SINE/input_data/yelp/yelp_samples.txt"
        args.class_num = 2
    elif dataset == 'imdb':
        train_data_file = "/home/hanqiyan/latent_topic/SINE/input_data/imdb/train.jsonlist"
        test_data_file = "/home/hanqiyan/latent_topic/SINE/input_data/imdb/imdb_samples.jsonlist"
        args.class_num = 2
    elif dataset=='news':
        train_data_file = '/home/hanqiyan/latent_topic/SINE/input_data/guadian_news/train_news_data.txt'
        test_data_file = '/home/hanqiyan/latent_topic/SINE/input_data/guadian_news/test_news_data.txt'
        args.class_num = 5
    elif dataset == "eraser_movie":
        train_data_file = '/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/train.jsonl'
        test_data_file = '/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/test.jsonl'
        num_label = 2

    print("load vocabulary")
    vocab = get_vocabulary(train_data_file,vocabsize=15000,dataset=dataset,stopwords=stopwords)
    print("load test data")
    if metric == "acc" or metric == "gen_bin_datasets":
        test_text,test_label,test_idx = load(test_data_file,vocab,max_value=1000,dataset="eraser_movie")
        print("Generate_Bin_datasets")
        label_file = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/VMASK/original_set/label.txt","w")
        label_file.close()
    elif metric == "cset" or metric == "sset" or metric == "original":
        print("Generate_Predict_Logits")
        test_text,test_label,test_idx,test_wpos = load_bin_dataset(metric,k_bin,vocab,max_value=1000)
    elif metric ==  "gen_hard_rats":
        test_text,test_label,test_idx,test_wpos,doc_length = load_rats(test_data_file,vocab,max_value=1000,max_utterance=50,dataset=dataset,type="original",stopwords=stopwords)


    print("load pretrained wordvec")

    args.embed_num = len(vocab)

    class B:
        text = torch.zeros(1).to(args.device)
        label = torch.zeros(1).to(args.device)

    # load model
    model = MASK_LSTM(args, vectors=None)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_acc = None
    beta = args.beta
    # print("The masked words number is %d"%args.mask_k)

    # load best model and test
    filename = "/home/hanq1yanwarwick/SINE/baselises/lstm_mask/masklstm-{}.pt".format(args.dataset_name)
    id2word = {v: k for k, v in vocab.items()}
    loaded_dict = torch.load(filename)
    model.load_state_dict(loaded_dict)
    # with open(filename, 'rb') as f:
    #     model = torch.load(f)
    model.to(torch.device(args.device))
    evaluation(model, test_text.copy(), test_label.copy(), test_idx, test_wpos,flag=metric,id2word=id2word,output_file=args.output_file,metric=metric,k_bin=k_bin,doc_length=doc_length)


if __name__ == "__main__":
    random_seed()
    criterion = nn.CrossEntropyLoss()
    dataset = args.dataset_name
    metric = "gen_hard_rats"
    for k_bin in [999]:
        main(metric,k_bin,dataset)
