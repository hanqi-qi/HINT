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
import pandas as pd
from torch import nn
from torch import optim
from collections import Counter
from itertools import chain
from sklearn.manifold import TSNE
import pickle
from matplotlib import pyplot as plt
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import ParameterGrid
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from funcs import load_process,getVectors,get_vocabulary,load_embedding,load,load_bin_dataset,load_rats,load_general_dataset,load_general_dataset_explain
from sine_exmodel_latex import HierachicalClassifier
from dataloader import DataIter
import json
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

def evaluate(model,loss_function,batch_generator,cuda=None,id2word=None,output_file=None,dis_encoder=None,dis_decoder=None,args=None):
    if not cuda is None:
        model.cuda(cuda)
    model.eval()
    total_loss,total_kld_loss = 0,0
    total_hit = 0
    total_sample = 0
    batch_i = 0
    true_label_list = []
    test_sample_idx = []
    predicted_label_list = []
    logits0,logits1,logits2,logits3,logits4 = [] ,[],[],[],[]
    att_list = []
    attention_weight_list =[]
    classifier_input_list = []
    original_index_list = []
    batch_num = 0
    doc_id = 0
    annotations = []
    with torch.no_grad():
        for batch in batch_generator:
            batch_num +=1
            # print("batch_num%d"%batch_num)
            if batch_num > 10000:
                break
            data, target, tfidf,idx,wordpos,length, original_index = batch[0], batch[1], batch[2], batch[3],batch[4],batch[5],batch[6]
            if cuda is None:
                data_var_list = [torch.LongTensor(chunk) for chunk in data]
                target_var = torch.LongTensor(target)
                length_var = torch.LongTensor(length)
                tfidf_var_list = [torch.tensor(chunk) for chunk in tfidf]
                wordpos_var_list = [torch.tensor(chunk) for chunk in wordpos]
            else:
                data_var_list = [torch.LongTensor(chunk).cuda(cuda) for chunk in data]
                target_var = torch.LongTensor(target).cuda(cuda)
                length_var = torch.LongTensor(length)
                tfidf_var_list = [torch.tensor(chunk).cuda(cuda) for chunk in tfidf]
                wordpos_var_list = [torch.tensor(chunk).cuda(cuda) for chunk in wordpos]
            if args.metric == "gen_bin_dataset" or args.metric=="general" or args.metric == "hard_rats" or args.metric == "case_study":
                flag = "gen_ex"
            else:
                flag = "gen_noex"
            predicted_target,attention_weight,classifier_input,_,senatt_arr,kld_loss,recon_loss,doc_id,anot= model(data_var_list,tfidf_var_list,wordpos_var_list,length,flag=flag,id2word=id2word,output_file=output_file,doc_id=doc_id,target=target_var,dis_encoder=dis_encoder,dis_decoder=dis_decoder,doc_name=idx)
            if anot is not None:
                annotations.extend(anot)
            att_list.append(senatt_arr.cpu().detach().numpy())
            # loss = loss_function(predicted_target, target_var)
            _,predicted_label = torch.max(predicted_target,dim=1)
            logits0.extend(list(np.array(predicted_target[:,0].cpu())))
            logits1.extend(list(np.array(predicted_target[:,1].cpu())))
            logits2.extend(list(np.array(predicted_target[:,2].cpu())))
            logits3.extend(list(np.array(predicted_target[:,3].cpu())))
            logits4.extend(list(np.array(predicted_target[:,4].cpu())))
            total_hit += torch.sum(predicted_label.data == target_var.data)
            # total_loss += loss.item()
            total_sample += data[0].shape[1]
            true_label_list.append(target)
            test_sample_idx.extend(idx)
            predicted_label_array_batch = np.array(predicted_label.data.cpu())
            predicted_label_list.append(predicted_label_array_batch)
            # # attention_weight_list.append(attention_weight)
            # classifier_input_list.append(classifier_input)
            # original_index_list.append(original_index)
            # batch_i += 1
    true_label_array = np.asarray(list(chain(*true_label_list)),dtype=np.int32)
    predicted_label_array = np.asarray(list(chain(*predicted_label_list)),dtype=np.int32)
    #save the generate_rat.json
    if len(annotations) > 1:
        with open('/home/hanq1yanwarwick/SINE/explain_metric/SINE_test_decoder_05bin_att_imdbweight.json', 'w') as fout:
            json.dump(annotations , fout)

    if args.metric == "original":
        pre_results = pd.DataFrame({"idx":test_sample_idx,"base_pre":predicted_label_array,"base_label":true_label_array,"base_0logits":logits0,"base_1logits":logits1,"base_2logits":logits2,"base_3logits":logits3,"base_4logits":logits4})
    elif "set" in args.metric:
        pre_results = pd.DataFrame({"idx":test_sample_idx,"pre":predicted_label_array,"label":true_label_array,"0logits":logits0,"1logits":logits1,"2logits":logits2,"3logits":logits3,"4logits":logits4})
    pre_results.to_csv("./eraser_metric/output_logits/SINE/{}/{}/prediction_results_{}_{}bin_SelfExplanation.csv".format(args.dataset,args.variants,args.metric,args.k_bin),index_label="idx",index=False)
    print("Predict Probalibility file of %s for %d bins is saved"%(args.metric,args.k_bin))
    # original_index_array = np.asarray(list(chain(*original_index_list)),dtype=np.int32)
    # classifier_input_array = np.concatenate(classifier_input_list,axis=0)

    acc = float(total_hit)/float(total_sample)
    print(acc)
    return None

def vis_dec(id2word, indices_decoder,sim_decoder,model,rank,indices_rank,epoch_i=None):
    output_file = "/home/hanq1yanwarwick/SINE/log/topic_words.txt"
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
    f = open(output_file,"w")
    
    topic_wordcloud = []
    for i in indices_decoder.t():# enumerate the vocabs for all the topics(50)
        t += 1
        word_fre = {}
        for wid,j in enumerate(i): #the j-th vocab for the topic
            topic = topic + " "+ id2word[j.item()]
            decoder_ranking.append(model.embedding.weight[j.item()].cpu().detach().numpy())
            word_fre[id2word[j.item()]] = sim_decoder[wid,t-1].item()
        #for t-th topic, the attention weight for positive label: 
        topic_wordcloud.append(word_fre)
        f.writelines("decoder topic \#"+ str(t) + ":" + topic+ ". score: " + str(rank[t-1,0].item()) + ", " + str(rank[t-1,1].item()))
        f.writelines(" ")
        topic = " "
        
        #draw wordcloud for each topic, using the word weight as frequency
        if rank[t-1,0] < rank[t-1,1]:
            sen_label = "Positive:{:.2f}".format(rank[t-1,1].item())
        else:
            sen_label = "Negative:{:.2f}".format(rank[t-1,0].item())
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(stopwords=stopwords,background_color="white")
        wordcloud.generate_from_frequencies(frequencies=word_fre)
        plt.figure()
        plt.title(sen_label)
        plt.axis("off")
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.savefig("./topic_wc/topic:{}_wordcloud.png".format(t),dpi=300)
    return None

    

def main():
    parser = argparse.ArgumentParser(description='MASK_LSTM text classificer')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--num_epoch', type=int, default=1, help='epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--d_t', type=int, default=50, help='number of topics in code')
    parser.add_argument('--cuda', type=int, default=0, help='gpu id')
    parser.add_argument('--emb_size', type=int, default=300, help='word embedding size')
    parser.add_argument('--mlp_size', type=int, default=200, help='word embedding size')
    parser.add_argument('--word_rnn_size', type=int, default=150, help='word_rnn_size')
    parser.add_argument('--word_rnn_num_layer', type=int, default=1, help='word_rnn_num_layer')
    parser.add_argument('--context_rnn_size', type=int, default=150, help='word_rnn_num_layer')
    parser.add_argument('--context_attention_size', type=int, default=200, help='context_attention_size')
    parser.add_argument('--pretrain_emb', type=str, default=None, help='use glove or googlenews word embedding')
    parser.add_argument('--topic_learning', type=str, default="bayesian", help='using tfidf or average to aggregate the words')
    parser.add_argument('--word_rnn_bidirectional', type=bool, default=True, help='use glove or googlenews word embedding')
    parser.add_argument('--context_rnn_bidirectional', type=bool, default=True, help='use glove or googlenews word embedding')
    parser.add_argument('--context_rnn_num_layer', type=int, default=1, help='use glove or googlenews word embedding')
    parser.add_argument('--dropout', type=float, default=0.3, help='use glove or googlenews word embedding')
    parser.add_argument('--word_attention_size', type=int, default=150, help='use glove or googlenews word embedding')
    parser.add_argument('--num_word', type=int, default=15000, help='vocabulary size')
    parser.add_argument('--num_label', type=int, default=2, help='number of sentiment labels')
    parser.add_argument('--sentenceEncoder',type=str,default="GAT",help="using GAT or Transformer as sentence encoder")
    parser.add_argument('--glayers', type=int, default=1, help='number of graph layers')
    parser.add_argument("--context_att",type=int,default=1,help="using context attention $\alpha$ or not")
    parser.add_argument("--topic_weight",type=str,default="tfidf",help="bayesian or tfidf or None")
    parser.add_argument("--regularization",type=int,default=1,help="using regularization term or not")
    parser.add_argument("--dataset",type=str,default="yelp",help="using imdb/yelp/guardian news")
    parser.add_argument("--vae_scale",type=float,default=0.01,help="using imdb/yelp/guardian news")
    parser.add_argument('--tsoftmax', type=str, default=1, help='the temperature of softmax in co_attention_weight')
    parser.add_argument('--data_processed', type=int, default=0, help='the temperature of softmax in co_attention_weight')
    parser
    parser.add_argument('--metric', type=str, default="gen_dataset", help='use completeness or sufficience as metric')
    parser.add_argument('--k_bin', type=int, default=999, help='use the k-th bin dataset')
    parser.add_argument('--variants', type=str, default="hint", help='use different variant for interpretability')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,format="%(asctime)s\t%(message)s")
    dataset = args.dataset

    grid_search = {}
    params = {
        'dataset':["yelp"],
        'sentenceEncoder':[args.sentenceEncoder],
        "context_att":[1],
        "cuda":[3],
        "topic_weight":["tfidf"],
        "num_label":[2],
        "regularization":[1],
        "batch_size":[args.batch_size],
        'num_word':[args.num_word], #time-consuming!
        'pretrain_emb':["glove"],
        'topic_learning':["bayesian"],
        'seed':[29],
        "glayers":[1],
        "metric":["gen_bin_dataset"],
        "variants":["HINT"],
        "k_bin":[999]
        }
    params_search = list(ParameterGrid(params))
    acc_list =[]
    dataset = args.dataset
    if dataset=="yelp":
        train_data_file = "./input_data/yelp/medical_train.txt"
        # test_data_file = "./input_data/yelp/medical_test.txt"
        test_data_file = "/home/hanq1yanwarwick/SINE/baselises/yelp_casestudy.txt"
        args.num_label = 2
    elif dataset == 'imdb':
        train_data_file = "./input_data/imdb/train.jsonlist"
        # test_data_file = "/home/hanq1yanwarwick/SINE/baselises/imdb_samples.jsonlist"
        # test_data_file = "/home/hanq1yanwarwick/SINE/baselises/imdb_case.jsonlist"
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
    pretrain_emb = args.pretrain_emb

    if pretrain_emb == 'googlenews':
        pretrained_embedding,hit_rate = getVectors(embed_dim=300, wordvocab=vocab)
    elif pretrain_emb == 'glove':
        pretrained_embedding = load_embedding(vocab,"/mnt/sda/media/Data2/hanqi/sine/glove.840B.300d.txt",embedding_size=300)
    else:
        pretrained_embedding = None
    for i, param in enumerate(params_search):
        print("*************params_search_{}*************".format(i + 1))
        print(param)
        for key, value in param.items():
            setattr(args, key, value)
        # p_list, r_list, f1_list = [], [], []
        # train_batch = DataIter(train_data,train_label,train_tfidf,args.batch_size,args.cuda)#ori_code
        print("Dataset is %s"%(args.dataset))
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
            if args.metric == "gen_bin_dataset":
            # if dataset == "imdb" or dataset == "yelp":
                test_data,test_label,test_tfidf,test_idx,test_wordpos_idx = load_general_dataset_explain(test_data_file,vocab,max_value=60,max_utterance=50,dataset=dataset,type="train",stopwords=stopwords)
                label_file = open("/home/hanq1yanwarwick/SINE/input_data/{}/variants/{}/original_set/label.txt".format(dataset,args.variants),"w")
                label_file.close()
            # elif dataset == "eraser_movie":
            elif args.metric == "hard_rats":#test tokenf1
                test_data,test_label,test_tfidf,test_idx,test_wordpos_idx = load_rats(test_data_file,vocab,max_value=60,max_utterance=50,dataset=dataset,type="original",stopwords=stopwords)
                # if args.metric == "gen_bin_dataset" or args.metric == "general":
                #     test_data,test_label,test_tfidf,test_idx,test_wordpos_idx = load_rats(test_data_file,vocab,max_value=60,max_utterance=50,dataset=dataset,type="original",stopwords=stopwords)
                #     label_file = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/SINE/original_set/label.txt","w")
                #     label_file.close()
            elif args.metric == "cset" or args.metric == "sset" or args.metric == "original" or args.metric=="rcset" or args.metric == "rsset":
                test_data,test_label,test_tfidf,test_idx,test_wordpos_idx = load_bin_dataset(args.metric,args.k_bin,vocab,max_value=60,max_utterance=50,stopwords=stopwords,model="SINE",dataset=args.dataset,variants=args.variants)
                print("Data Processed and Loaded")

        test_batch = DataIter(test_data,test_label,test_tfidf,test_idx,test_wordpos_idx,args.batch_size,2,shuffle=False)#ori_code

        model = HierachicalClassifier(args,pretrained_embedding=pretrained_embedding)
        print("Loading the pretrained model weight")
        weight_file = "bestmodel_woMean_yelp_HINT_wholedataset_softmaxLogits_0106.pt"
        # weight_file = "bestmodel_yelp_bayesian.pt"
        loaded_dict = torch.load(weight_file)
        model.load_state_dict(loaded_dict)
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
        loss_function = nn.CrossEntropyLoss()
        id2word = {vocab[key]:key for key in vocab.keys()}
        output_file = "/mnt/sda/media/Data2/hanqi/sine/sine_ex/{}_explainations.txt".format(args.dataset)
        f = open(output_file,"w")
        f.close()
        
        #print topic words fot $d_t$ topics
        # #vocabsize->topic(d_t) matrix: dis_decoder
        dis_decoder = torch.mm(model.embedding.weight, model.topic_decoder.weight)
        dis_encoder = torch.mm(model.embedding.weight, model.topic_encoder.weight.t())
        # #trace.shape[topic(d_t),word_embedding(rnn_units)]
        # trace = torch.mm(model.context_topic_attention_linear.weight.t(), model.context_conv_attention_linear.weight)
        # trace = torch.mm(trace, torch.squeeze(model.context_conv_attention_layer.weight))
        # #use trace to claasify, topics as the sequence token, rnn units keep unchange
        # rank = torch.squeeze(model.classifier(trace))#[ntopics,2]
        # #select top50 words for each topic
        # sim_decoder, indices_decoder = torch.topk(dis_decoder, 50, dim=0)#weight/word_indices:[nwords,ntopics]
        # #select top5 topics
        # sim_rank, indices_rank = rank.topk(50,dim=0)
        # vis_dec(id2word, indices_decoder,sim_decoder,model,rank,indices_rank)
        evaluate(model,loss_function,test_batch,cuda=args.cuda,id2word=id2word,output_file=output_file,dis_encoder=dis_encoder,dis_decoder=dis_decoder,args=args)
    
    # print("hit rate: ", hit_rate)
    for key, value in grid_search.items():
        print("Main: ", key, value)

if __name__ == "__main__":
    main()
