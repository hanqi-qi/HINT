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
from sklearn.manifold import TSNE
import pickle
from matplotlib import pyplot as plt
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import ParameterGrid

from funcs import load_general_dataset, load_general_dataset_explain, load_process,getVectors,get_vocabulary,load_embedding,load, load_rats
from sine_model import HierachicalClassifier
from dataloader import DataIter
import json
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

def evaluate(model,loss_function,batch_generator,cuda=None):
    model.eval()
    total_loss,total_kld_loss = 0,0
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
            data, target, tfidf,idx,wpos,length, original_index = batch[0], batch[1], batch[2], batch[3],batch[4],batch[5],batch[6]
            if cuda is None:
                data_var_list = [torch.LongTensor(chunk) for chunk in data]
                target_var = torch.LongTensor(target)
                length_var = torch.LongTensor(length)
                tfidf_var_list = [torch.tensor(chunk) for chunk in tfidf]
            else:
                data_var_list = [torch.LongTensor(chunk).cuda(cuda) for chunk in data]
                target_var = torch.LongTensor(target).cuda(cuda)
                length_var = torch.LongTensor(length)
                tfidf_var_list = [torch.tensor(chunk).cuda(cuda) for chunk in tfidf]
            predicted_target,attention_weight,classifier_input,_,senatt_arr,kld_loss,recon_loss= model(data_var_list,tfidf_var_list,length)
            att_list.append(senatt_arr.cpu().detach().numpy())
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

    # true_label_array = np.asarray(list(chain(*true_label_list)),dtype=np.int32)
    # predicted_label_array = np.asarray(list(chain(*predicted_label_list)),dtype=np.int32)
    # original_index_array = np.asarray(list(chain(*original_index_list)),dtype=np.int32)
    # classifier_input_array = np.concatenate(classifier_input_list,axis=0) 
    acc = float(total_hit)/float(total_sample)
    print(total_sample)
    total_kld_loss += kld_loss.mean().item()
    # returned_document_list = [batch_generator.sample_document(index) for index in original_index_array]
    model.train()
    return total_loss/(batch_i+1),acc
    #acc,true_label_array,predicted_label_array,returned_document_list,total_kld_loss/(batch_i+1),sum(att_list)/len(att_list)

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

def train_model(model,optimizer,loss_function,num_epoch,train_batch_generator,test_batch_generator,vocab,cuda=None,d_t=0,topic_learning="autoencoder",dataset=None,gat_layers=None):
    logging.info("Start Tranining")
    if cuda != None:
        model.cuda(cuda)
    reverse_vocab = {vocab[key]:key for key in vocab.keys()}
    best_model_loss = 1e7
    temp_batch_index = 0
    loss_C_total = 0
    loss_A_total = 0
    loss_R_total = 0
    loss_KLD_total, loss_Recon_total = 0,0
    log_loss = open('loss.txt', 'a')
    best_dev_acc = 0
    # num_epoch = 0
    for epoch_i in range(num_epoch):
        logging.info("Epoch {}".format(epoch_i))
        for train_batch in train_batch_generator:
        # for stidx in range(0, len(data_label), 50):
            temp_batch_index += 1
            # print("temp_batch_index is%.4f"%temp_batch_index)
            train_data,train_target,train_tfidf,train_idx,wordpos,length_data,original_index = train_batch[0],train_batch[1],train_batch[2],train_batch[3],train_batch[4],train_batch[5],train_batch[6]
            if cuda is None:
                train_data_var_list = [torch.LongTensor(chunk) for chunk in train_data]
                train_target_var = torch.LongTensor(train_target)
                train_tfidf_var_list = [torch.tensor(chunk) for chunk in train_tfidf]
                length_var = torch.LongTensor(length_data)
                # wordpos_var_list = [torch.tensor(chunk) for chunk in wordpos]
            else:
                train_data_var_list = [torch.LongTensor(chunk).cuda(cuda) for chunk in train_data]
                train_target_var = torch.LongTensor(train_target).cuda(cuda)
                train_tfidf_var_list = [torch.tensor(chunk).cuda(cuda) for chunk in train_tfidf]
                length_var = torch.LongTensor(length_data)
                # wordpos_var_list = [torch.tensor(chunk).cuda(cuda) for chunk in wordpos]

            predicted_train_target,_,_,aspect_loss,senatt,kld_loss,recon_loss = model(train_data_var_list,train_tfidf_var_list,length_var)
            optimizer.zero_grad()
            loss_C = loss_function(predicted_train_target,train_target_var)
            loss_A = aspect_loss.mean()
            loss_R = torch.norm(torch.eye(d_t).cuda(cuda) - torch.mm(model.topic_encoder.weight, model.topic_encoder.weight.t()))
            loss_C_total += loss_C.item()
            loss_A_total += loss_A.item()
            loss_R_total += loss_R.item()
            loss_KLD_total += kld_loss.item()
            loss_Recon_total += recon_loss.item()
            loss = loss_C + 0.05 * loss_C +  0.01 * loss_R
            if topic_learning == "bayesian":
                loss = loss+kld_loss.mean()+recon_loss.mean()
            loss.backward()
            del loss,loss_C,loss_A,loss_R
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
        # if temp_batch_index%100 == 0: #every minibatch, check the intepretatbility
            # id2word = {vocab[key]:key for key in vocab.keys()}
        if True: #change to minibatch
            train_loss,train_acc = 0.0,0.0
            C_loss = loss_C_total/100
            A_loss = loss_A_total/100
            R_loss = loss_R_total/100
            KLD_loss = loss_KLD_total/100
            Recon_loss = loss_Recon_total/100
            log_loss.write("{0:6f},{1:6f},{2:6f}\n".format(C_loss,A_loss,R_loss))
            loss_A_total, loss_C_total, loss_R_total,loss_KLD_total, loss_Recon_total= 0,0,0,0,0
#total_loss/(batch_i+1),acc,true_label_array,predicted_label_array,returned_document_list,total_kld_loss/(batch_i+1)
            test_loss,test_acc = evaluate(model,loss_function,test_batch_generator,cuda)#ori_code
            # logging.info("The std of attention weight:{0:0.6f}".format(senatt_std))
            predicted_label = np.argmax(predicted_train_target.detach().cpu().numpy(),axis=1)
            train_hits = np.count_nonzero(predicted_label == train_target_var.data.detach().cpu().numpy())
            train_acc = train_hits/len(predicted_label)
            logging.info("\nEpoch :{0:8d}\ntrain_labelloss:{1:0.6f}\ttrain_vaeloss:{2:0.6f}\ttrain_reconloss:{3:0.6f}\ttrain_acc:{4:0.6f}\ntest_loss:{5:0.6f}\ttest_acc:{6:0.6f}".format(epoch_i, C_loss,KLD_loss,Recon_loss,train_acc,test_loss,test_acc))
            filename = "bestmodel_woMean_{}_autoencoder_wholedataset_softmaxLogits_0106.pt".format(dataset)
            if test_acc>best_dev_acc:
                best_dev_acc = test_acc
                #save the best model with arguments
                # with open(filename, 'wb') as f:
                torch.save(model.state_dict(), filename)
                    # torch.save(model, f)
                print("higher dev acc and best model saved: %.4f"%best_dev_acc)
            ''''VISUALIZATION'''
                # id2word = {vocab[key]:key for key in vocab.keys()}
                # #vocabsize->topic(d_t) matrix: dis_decoder
                # dis_decoder = torch.mm(model.embedding.weight, model.topic_decoder.weight)
                # dis_encoder = torch.mm(model.embedding.weight, model.topic_encoder.weight.t())
                # #trace.shape[topic(d_t),word_embedding(rnn_units)]
                # trace = torch.mm(model.context_topic_attention_linear.weight.t(), model.context_conv_attention_linear.weight)
                # trace = torch.mm(trace, torch.squeeze(model.context_conv_attention_layer.weight))
                # #use trace to claasify, topics as the sequence token, rnn units keep unchange
                # rank = torch.squeeze(model.classifier(trace))
                # #select top10 words for each topic
                # sim_decoder , indices_decoder = torch.topk(dis_decoder, 10, dim=0)
                # #select top5 topics
                # sim_rank, indices_rank = rank.topk(5,dim=0)
                # vis_dec(id2word, indices_decoder,model,rank,indices_rank,epoch_i)
    test_loss,test_acc = evaluate(model,loss_function,test_batch_generator,cuda)
    logging.info("best_dev_acc:{0:0.4f}".format(test_acc))
    # logging.info("best_dev_acc:{0:0.4f}".format(best_dev_acc))
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
    parser = argparse.ArgumentParser(description='MASK_LSTM text classificer')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--num_epoch', type=int, default=30, help='epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--d_t', type=int, default=50, help='number of topics in code')
    parser.add_argument('--cuda', type=int, default=3, help='gpu id')
    parser.add_argument('--emb_size', type=int, default=300, help='word embedding size')
    parser.add_argument('--mlp_size', type=int, default=200, help='word embedding size')
    parser.add_argument('--word_rnn_size', type=int, default=150, help='word_rnn_size')
    parser.add_argument('--word_rnn_num_layer', type=int, default=1, help='word_rnn_num_layer')
    parser.add_argument('--glayers', type=int, default=1, help='number of graph layers')
    parser.add_argument('--context_rnn_size', type=int, default=150, help='word_rnn_num_layer')
    parser.add_argument('--context_attention_size', type=int, default=200, help='context_attention_size')
    parser.add_argument('--pretrain_emb', type=str, default="glove", help='use glove or googlenews word embedding')
    parser.add_argument('--topic_learning', type=str, default="bayesian", help='using tfidf or average to aggregate the words')
    parser.add_argument('--word_rnn_bidirectional', type=bool, default=True, help='use glove or googlenews word embedding')
    parser.add_argument('--context_rnn_bidirectional', type=bool, default=True, help='use glove or googlenews word embedding')
    parser.add_argument('--context_rnn_num_layer', type=int, default=1, help='use glove or googlenews word embedding')
    parser.add_argument('--dropout', type=float, default=0.3, help='use glove or googlenews word embedding')
    parser.add_argument('--word_attention_size', type=int, default=150, help='use glove or googlenews word embedding')
    parser.add_argument('--num_word', type=int, default=15000, help='vocabulary size')
    parser.add_argument('--num_label', type=int, default=2, help='number of sentiment labels')
    parser.add_argument('--sentenceEncoder',type=str,default="GAT",help="using GAT or Transformer as sentence encoder")
    parser.add_argument("--context_att",type=int,default=1,help="using context attention $\alpha$ or not")
    parser.add_argument("--topic_weight",type=str,default="tfidf",help="bayesian or tfidf or None")
    parser.add_argument("--regularization",type=int,default=1,help="using regularization term or not")
    parser.add_argument("--dataset",type=str,default="guardian_news",help="using imdb/yelp/guardian news")
    parser.add_argument("--vae_scale",type=float,default=0.01,help="using imdb/yelp/guardian news")
    parser.add_argument('--tsoftmax', type=str, default=1, help='the temperature of softmax in co_attention_weight')
    parser.add_argument('--data_processed', type=int, default=0, help='the temperature of softmax in co_attention_weight')
    parser.add_argument('--load_pretrained', type=int, default=0, help='if loads the pretrained model weight')
    parser.add_argument('--seed', type=int, default=1, help='if loads the pretrained model weight')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,format="%(asctime)s\t%(message)s")
    dataset = args.dataset
    print("Dataset is %s"%(args.dataset))
    if dataset=="yelp":
        train_data_file = "./input_data/yelp/medical_train.txt"
        test_data_file = "./input_data/yelp/medical_test.txt"
        args.num_label = 2
    elif dataset == 'imdb':
        train_data_file = "./input_data/imdb/train.jsonlist"
        test_data_file = "./input_data/imdb/test.jsonlist"
        args.num_label = 2
    elif dataset =="guardian_news":
        train_data_file = './input_data/guardian_news/train_news_data.txt'
        test_data_file = './input_data/guardian_news/test_news_data.txt'
        args.num_label = 5
    elif dataset == "eraser_movie":
        train_data_file = './input_data/eraser_movie/movies/train.jsonl'
        test_data_file = './input_data/eraser_movie/movies/test.jsonl'
        args.num_label = 2
    vocab = get_vocabulary(train_data_file,vocabsize=args.num_word,dataset=dataset,use_stem=False,stopwords=stopwords)
    # train_tfidffile=json.loads(open("tfidf_weight/norm_yelp_tfidf_train.json").readlines()[0])
    # test_tfidffile=json.loads(open("tfidf_weight/norm_yelp_tfidf_test.json").readlines()[0])
    if args.data_processed > 0:
        train_data = load_process(args.dataset,"train","data")
        test_data = load_process(args.dataset,"test","data")
        train_label = load_process(args.dataset,"train","label")
        test_label = load_process(args.dataset,"test","label")
        train_tfidf = load_process(args.dataset,"train","tfidf")
        test_tfidf = load_process(args.dataset,"test","tfidf")
        print("Data Loaded")
    else:
        if dataset!="eraser_movie":
            train_data,train_label,train_tfidf,train_idx,train_wpos_idx = load_general_dataset(train_data_file,vocab,max_value=60,max_utterance=50,dataset=dataset,type="original",stopwords=stopwords)
            test_data,test_label,test_tfidf,test_idx,test_wpos_idx = load_general_dataset(test_data_file,vocab,max_value=60,max_utterance=50,dataset=dataset,type="original",stopwords=stopwords)
            print("Data Processed and Loaded")
        elif dataset == "eraser_movie":
            train_data,train_label,train_tfidf,train_idx,train_wpos_idx = load_rats(train_data_file,vocab,max_value=60,max_utterance=50,dataset=dataset,type="original",stopwords=stopwords)
            test_data,test_label,test_tfidf,test_idx,test_wpos_idx = load_rats(test_data_file,vocab,max_value=60,max_utterance=50,dataset=dataset,type="original",stopwords=stopwords)
            print("Data Processed and Loaded")

    pretrain_emb = args.pretrain_emb

    if pretrain_emb == 'googlenews':
        pretrained_embedding,hit_rate = getVectors(embed_dim=300, wordvocab=vocab)
    elif pretrain_emb == 'glove':
        pretrained_embedding = load_embedding(vocab,"/mnt/sda/media/Data2/hanqi/sine/glove.840B.300d.txt",embedding_size=300)
    else:
        pretrained_embedding = None

    grid_search = {}
    params = {
        'dataset':["guardian_news"],
        'sentenceEncoder':[args.sentenceEncoder],
        "context_att":[1],
        "batch_size":[32],
        "topic_weight":["average"],
        "regularization":[0],
        'num_word':[args.num_word], #time-consuming!
        'pretrain_emb':["glove"],
        'topic_learning':["autoencoder"],        
        'seed':[args.seed],
        "glayers":[1]
        }
    params_search = list(ParameterGrid(params))
    acc_list =[]
    for i, param in enumerate(params_search):
        print("*************params_search_{}*************".format(i + 1))
        print(param)
        for key, value in param.items():
            setattr(args, key, value)
        # p_list, r_list, f1_list = [], [], []
        torch.manual_seed(args.seed)
        train_batch = DataIter(train_data,train_label,train_tfidf,train_idx,train_wpos_idx,args.batch_size,padded_value=2)#ori_code
        test_batch = DataIter(test_data,test_label,test_tfidf,test_idx,test_wpos_idx,args.batch_size,padded_value=2)#ori_code
        model = HierachicalClassifier(args,pretrained_embedding=pretrained_embedding)
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
        loss_function = nn.CrossEntropyLoss()
        #load the pretrained model
        if args.load_pretrained>0:
            print("Loading the pretrained model weight")
            weight_file = "bestmodel_woMean_{}_HINT_wholedataset_softmaxLogits_0106.pt".format(args.dataset,args.topic_learning)
            loaded_dict = torch.load(weight_file)
            model.load_state_dict(loaded_dict)
        best_dev_acc = train_model(model,optimizer,loss_function,args.num_epoch,train_batch,test_batch,vocab,cuda=args.cuda,d_t=args.d_t,topic_learning=args.topic_learning,dataset=args.dataset,gat_layers=args.glayers)#ori_code
        grid_search[str(param)] = {"best_dev_acc": [round(best_dev_acc, 4)]}
    
    # print("hit rate: ", hit_rate)
    for key, value in grid_search.items():
        print("Main: ", key, value)

if __name__ == "__main__":
    main()
