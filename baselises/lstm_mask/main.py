import argparse
import torch
from torch import nn
from mask_lstm_model_old import *
from getvectors import getVectors2
from itertools import chain
import os
import numpy as np
import random
import dill
import copy
import torch.optim as optim
from collections import Counter
import json
from progressbar import ProgressBar
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from nltk.corpus import stopwords
stopwords = stopwords.words('english')


os.environ["CUDA_VISIBLE_DEVICES"]="3"
parser = argparse.ArgumentParser(description='MASK_LSTM text classificer')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('-beta', type=float, default=1, help='beta')
parser.add_argument('--weight_decay', default=0, type=float, help='adding l2 regularization')
parser.add_argument('--clip', type=float, default=1, help='gradient clipping')
parser.add_argument('-epochs', type=int, default=200, help='number of epochs for training')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-dropout', type=float, default=0.2, help='the probability for dropout')
parser.add_argument('-embed-dim', type=int, default=300, help='original number of embedding dimension')
parser.add_argument('-lstm-hidden-dim', type=int, default=100, help='number of hidden dimension')
parser.add_argument('-lstm-hidden-layer', type=int, default=1, help='number of hidden layers')
parser.add_argument('-mask-hidden-dim', type=int, default=300, help='number of hidden dimension')
parser.add_argument("--max_sent_len", type=int, dest="max_sent_len", default=250, help='max sentence length')
parser.add_argument("--activation", type=str, dest="activation", default="tanh", help='the choice of \
        non-linearity transfer function')
parser.add_argument('--save', type=str, default='masklstm.pt', help='path to save the final model')
parser.add_argument('--mode', type=str, default='static', help='available models: static, non-static')
parser.add_argument('--gpu', default=0, type=int, help='0:gpu, -1:cpu')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--use_lstmatt', type=bool, default=False, help='random seed')
parser.add_argument('--mask_k', type=int, default=30, help='the number of masked words')
parser.add_argument('--ifmask', type=bool, default=True, help='whether mask the words or softmax')
parser.add_argument("--dataset_name",type=str, default="eraser_movie",help="dataset name")
parser.add_argument("--class_num",type=int, default=2,help="number of labels")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
torch.manual_seed(args.seed)

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
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
    eraser_dict  =  {"NEG":"1","POS":"2"}
    def filter_key(sent):
        unk_count = list(sent).count(vocab['<unk>'])
        return unk_count/len(sent) < 0.3

    with open(textfile, "r") as f:
        line_list = f.readlines()
        line_len = len(line_list)
        random_index = np.random.permutation(line_len)
        line_list = [line_list[index] for index in random_index]
        progressbar = ProgressBar(maxval=len(line_list))
        word_list_buffer = []
        for line in line_list:
            if dataset=='yelp':
                _,_,label,sents = line.strip().split("\t\t") #2 pos, 1neg
                # sents = text.strip().split("<sssss>")
            elif dataset=='imdb':
                label = imbd_dic[json.loads(line)['sentiment']]
                sents = json.loads(line)['text'].strip()
                # sents = [x for i in text.strip().split('.') for x in i.split('<br /><br />')]
            elif dataset=='guardian_news':
                text,label = line.strip().split("\t")
                label = int(label)+1
                sents = text.strip().split(".") 
                # sents = text.strip().split(".")
            elif dataset == "eraser_movie":
                label = eraser_dict[json.loads(line)['classification']]
                filename = json.loads(line)['annotation_id']
                text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/docs/"+filename,"r").readlines()
                sent_list = [x.strip() for x in text if len(x.split())>1] 
                sents = " ".join(sent_list)
            doc_list = [word for word in sents.split(" ")]
            doc_array = convert_words2ids(doc_list,vocab,max_value=max_value,unk=vocab['<unk>'], sos=None,eos=vocab['<eos>'])
            sent_id_list = list(filter(filter_key,doc_array))
            new_sent_id_list = []
            previous_sent = []
            label_list.append(torch.reshape(torch.tensor(int(label)-1),[1]))
            document_list.append(torch.tensor(doc_array))
    return document_list,label_list
    
def get_vocabulary(textfile, initial_vocab={'<unk>':0,'<sssss>':1,'<eos>':2}, vocabsize=0,dataset='yelp',stopwords=None):
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
            text, _ = line.strip().split("\t")
        elif dataset == "eraser_movie":
            filename = json.loads(line)['annotation_id']
            text = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/docs/"+filename,"r").readlines()#one doc
            text = " ".join(text)
        for w in text.split(): # skip speaker indicator
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


'''use the data from JSAE project'''
dataset = args.dataset_name
if dataset=="yelp":
    train_data_file = "/home/hanq1yanwarwick/SINE/input_data/yelp/medical_train.txt"
    test_data_file = "/home/hanq1yanwarwick/SINE/input_data/yelp/medical_test.txt"
    args.class_num = 2
elif dataset == 'imdb':
    train_data_file = "/home/hanqiyan/latent_topic/SINE/input_data/imdb/train.jsonlist"
    test_data_file = "/home/hanqiyan/latent_topic/SINE/input_data/imdb/test.jsonlist"
    args.class_num = 2
elif dataset=='news':
    train_data_file = '/home/hanqiyan/latent_topic/SINE/input_data/guadian_news/train_news_data.txt'
    test_data_file = '/home/hanqiyan/latent_topic/SINE/input_data/guadian_news/test_news_data.txt'
    args.class_num = 5
elif dataset == "eraser_movie":
    train_data_file = '/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/train.jsonl'
    test_data_file = '/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/test.jsonl'
    args.class_num = 2

vocab = get_vocabulary(train_data_file,vocabsize=15000,dataset=dataset,stopwords=stopwords)


train_text,train_label = load(train_data_file,vocab,max_value=1000,max_utterance=10,dataset=dataset)
test_text,test_label = load(test_data_file,vocab,max_value=1000,max_utterance=10,dataset=dataset)
dev_text,dev_label = test_text,test_label

vectors = getVectors2(embed_dim=300, wordvocab=vocab)

args.embed_num = len(vocab)
# args.class_num = 5


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
def evaluation(model, data_text, data_label,flag,vocab):
    model.eval()
    mask_acc, ori_acc, loss, size = 0,0, 0, 0
    count = 0
    pred_l, label_l,mask_l=[],[],[]
    for stidx in range(0, len(data_label), args.batch_size):
        count += 1
        batch = batch_from_list(data_text[stidx:stidx + args.batch_size],
                                data_label[stidx:stidx + args.batch_size])
        # mask_pred,return_doc,probs = model(batch, 'eval_mask',vocab)
        ori_pred,_,_ = model(batch,'eval_ori',vocab)

        


        batch_loss = criterion(ori_pred, batch.label)
        loss += batch_loss.item()

        # _, mask_pred = mask_pred.max(dim=1)
        # mask_acc += (mask_pred == batch.label).sum().float()

        _, ori_pred = ori_pred.max(dim=1)
        ori_acc += (ori_pred == batch.label).sum().float()

        size += len(ori_pred)
        pred_l.append(ori_pred)
        # mask_l.append(mask_pred)
        label_l.append(batch.label)



    true_label_array = np.asarray(list(chain(*label_l)),dtype=np.int32)
    pre_array = np.asarray(list(chain(*pred_l)),dtype=np.int32)

    # if flag=='test':
    
    # #extract the removed words
    #     id2vocab = {vocab[key]:key for key in vocab.keys()}
    #     _, word_indices = torch.topk(probs,args.mask_k,dim=0)
    #     removed_words = []
    #     for bs in range(probs.shape[1]):
    #         # mask[word_indices[:,bs,0],bs,0] = 1
    #         removed_words.append(return_doc[bs,word_indices[:,bs,0]])
    #         labels_file = open("imdbcase_label_pre.txt",'w')
    #         rwords_file = open("imdbcase_rwords.txt",'w')
    #         labels_file.writelines("label"+"\t"+"ori_pre"+"\t"+"mask_pre"+"\n")
    #         rwords_file.writelines("removed_words"+"\n")
    #         for idx in range(len(label_l)):
    #             labels_file.writelines(str(label_l[0][idx])+"\t"+str(pred_l[0][idx])+"\t"+str(mask_l[0][idx])+"\n")
    #             rwords_file.writelines(" ".join([id2vocab[word_id] for word_id in removed_words[idx] if word_id in id2vocab.keys()])+"\n")

    # mask_acc /= size
    ori_acc /= size
    loss /= count
    return loss, mask_acc,ori_acc, true_label_array, pre_array



def main():
    # load model
    model = MASK_LSTM(args, vectors)
    model.to(torch.device(args.device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    print(model.parameters())
    best_val_acc = None
    beta = args.beta
    print("The masked words number is %d"%args.mask_k)
    # for epoch in range(1, args.epochs+1):
    #     model.train()
    #     print("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, args.epochs))
    #     lstm_count = 0
    #     trn_lstm_size, trn_lstm_corrects, trn_lstm_loss = 0, 0, 0

    #     # shuffle
    #     textlist1 = train_text.copy()
    #     labellist1 = train_label.copy()
    #     listpack = list(zip(textlist1, labellist1))
    #     random.shuffle(listpack)
    #     textlist1[:], labellist1[:] = zip(*listpack)

    #     for stidx in range(0, len(labellist1), args.batch_size):
    #         lstm_count += 1
    #         batch = batch_from_list(textlist1[stidx:stidx + args.batch_size],
    #                                 labellist1[stidx:stidx + args.batch_size])
    #         pred,_,_ = model(batch, 'train',vocab)
    #         optimizer.zero_grad()
    #         model_loss = criterion(pred, batch.label)
            

    #         batch_loss = model_loss + beta * model.infor_loss
    #         trn_lstm_loss += batch_loss.item()
    #         batch_loss.backward()
    #         nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
    #         optimizer.step()

    #         _, pred = pred.max(dim=1)
    #         trn_lstm_corrects += (pred == batch.label).sum().float()
    #         trn_lstm_size += len(pred)
    #     if epoch>0:
    #         dev_lstm_loss, dev_mask_lstm_acc, dev_ori_lstm_acc, _,_= evaluation(model, dev_text.copy(), dev_label.copy(),flag='dev',vocab=vocab)
    #         if not best_val_acc or dev_ori_lstm_acc > best_val_acc:
    #             best_model_file = "masklstm-{}.pt".format(args.dataset_name)
    #             print("Save better model to {}".format(best_model_file))
    #             with open(best_model_file, 'wb') as f:
    #                 torch.save(model.state_dict(), f)
    #             best_val_acc = dev_ori_lstm_acc

    #         train_lstm_acc = trn_lstm_corrects / trn_lstm_size
    #         train_lstm_loss = trn_lstm_loss / lstm_count
    #         train_vae_loss = model.infor_loss/ lstm_count
    #         print('local_epoch {} | train_vae_loss {} |train_lstm_loss {:.6f} | train_lstm_acc {:.6f} | dev_lstm_loss {:.6f} | '
    #             'dev_mask_lstm_acc {:.6f} | dev_ori_lstm_acc {:.6f} best_dev_acc {:.6f}'.format(epoch, train_vae_loss, train_lstm_loss, train_lstm_acc,
    #                                                                 dev_lstm_loss, dev_mask_lstm_acc, dev_ori_lstm_acc,best_val_acc))

    #         # annealing
    #         if epoch % 10 == 0:
    #             if beta > 0.01:
    #                 beta -= 0.099

    # # load best model and test
    # del model
    best_model_file = "masklstm-{}.pt".format(args.dataset_name)
    loaded_dict = torch.load(best_model_file)
    model.load_state_dict(loaded_dict)
    # with open(filename, 'rb') as f:
    #     model = torch.load(f)
    model.to(torch.device(args.device))
    _, mask_acc,ori_acc,pred_l,label_l = evaluation(model, test_text.copy(), test_label.copy(),flag='test',vocab=vocab)
    print('\nfinal_mask_test_acc {:.6f}'.format(ori_acc))
    # print("Test Multilabel Results:")
    # results =precision_recall_fscore_support(label_l, pred_l)
    # print("Precious for 5 labels:")
    # print(results[0])
    # print("Recall for 5 labels:")
    # print(results[1])
    # print("f1 for 5 labels:")
    # print(results[2])
    # test_macro_f1 = f1_score(label_l, pred_l,average="macro")
    # print("test macro f:%f"%test_macro_f1)


if __name__ == "__main__":
    random_seed()
    criterion = nn.CrossEntropyLoss()
    main()
