
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.activation import Threshold
from sine_bayesian import Vae
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import STOPWORDS
from vis_sine import generate, gen_hard_rats
import os
import random
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

class HierachicalClassifier(nn.Module):
    def __init__(self, args, pretrained_embedding=None):
        self.args = args
        self.emb_size = args.emb_size
        self.d_t = args.d_t
        self.word_rnn_size = args.word_rnn_size
        self.word_rnn_num_layer = args.word_rnn_num_layer
        self.word_rnn_bidirectional = args.word_rnn_bidirectional
        self.context_rnn_size = args.context_rnn_size
        self.context_rnn_num_layer = args.context_rnn_num_layer
        self.context_rnn_bidirectional = args.context_rnn_bidirectional
        self.num_label = 5
        super(HierachicalClassifier, self).__init__()
        self.embedding = nn.Embedding(args.num_word, args.emb_size)
        self.word_dropout = nn.Dropout(args.dropout)
        self.word_rnn = nn.GRU(input_size = args.emb_size, hidden_size = args.word_rnn_size,
                num_layers = args.word_rnn_num_layer, bidirectional = args.word_rnn_bidirectional)
        word_rnn_output_size = args.word_rnn_size * 2 if args.word_rnn_bidirectional else args.word_rnn_size
        self.word_conv_attention_layer = nn.Conv1d(args.emb_size, args.word_attention_size, 3, padding=2, stride=1)
        self.word_conv_attention_linear = nn.Linear(args.word_attention_size, 1, bias=False)
        self.word_aspect_attention_linear = nn.Linear(word_rnn_output_size, self.d_t, bias=False)
        self.word_aspect_attention_linear2 = nn.Linear(self.d_t, 1, bias=False)
        self.topic_encoder = nn.Linear(args.emb_size, self.d_t, bias=False)
        self.topic_decoder = nn.Linear(self.d_t, args.emb_size, bias=False)
        self.context_dropout = nn.Dropout(args.dropout)
        self.context_rnn = nn.GRU(input_size = word_rnn_output_size, hidden_size = args.context_rnn_size,
                num_layers = args.context_rnn_num_layer,bidirectional=args.context_rnn_bidirectional)
        context_rnn_output_size = args.context_rnn_size * 2 if args.context_rnn_bidirectional else args.context_rnn_size
        self.context_conv_attention_layer = nn.Conv1d(word_rnn_output_size, args.context_attention_size, kernel_size=1, stride=1)
        self.context_conv_attention_linear = nn.Linear(args.context_attention_size, self.d_t, bias=False)
        self.context_topic_attention_linear = nn.Linear(self.d_t, self.d_t, bias = True)
        
        self.classifier = nn.Sequential(nn.Linear(context_rnn_output_size, args.mlp_size),
                                        nn.LeakyReLU(),
                                        nn.Linear(args.mlp_size, args.num_label),
                                        nn.Softmax())
        self.device = args.cuda
        if pretrained_embedding is not None:
            self.embedding.weight.data = self.embedding.weight.data.new(pretrained_embedding)
        else:
            self.embedding.weight.data.uniform_(-1.0, 1.0)
        
        self.Vae = Vae(self.emb_size,self.d_t,self.device)
        self.stopwords = set(STOPWORDS)

        self.dataset = args.dataset
        self.variants = args.variants

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
    def generate_wordEx(self,sent,alpha,omega,id2word,output_file,wpos):
        """sent: wordids [sen_len,bs]
            alpha: word context attention [sen_len,bs,1]
            omega: word topic attention [sen_len,bs,1]"""
        outfile = open(output_file,"a+")
        text = sent.detach().cpu().numpy()
        alpha = alpha.squeeze(-1).detach().cpu().numpy()
        omega = omega.squeeze(-1).detach().cpu().numpy()
        wpos = wpos.detach().cpu().numpy()
        doc_list, doc_word_att, doc_tokenids, doc_wordpos_ids = [], [],[], []
        for bs in range(text.shape[1]):
            if len((text[:,bs]==2).nonzero()[0]) > 0 :
                doc_len = (text[:,bs]==2).nonzero()[0][0]
            else:
                doc_len = len(text[:,bs])
            doc,doc_tokenid,doc_wordpos =  [],[],[]
            sen_word_att = []
            for watt, wid,wbtt,wpid in zip(alpha[:,bs],text[:,bs],omega[:,bs],wpos[:,bs]):
                if wid>2:
                    doc.append(id2word[wid])
                    doc_tokenid.append(wid)
                    doc_wordpos.append(wpid)
                    # sen_word_att.append(max(watt,wbtt))
                    sen_word_att.append(watt)
            doc_list.append(doc)
            doc_tokenids.append(doc_tokenid)
            doc_word_att.append(sen_word_att)
            doc_wordpos_ids.append(doc_wordpos)
        return doc_list,doc_word_att,doc_tokenids,doc_wordpos_ids
    
    def symbol(self,input):
        if input>0:
            symbol = r"$\bigstar$"
        else:
            symbol = r"$\blacksquare$"
        return symbol
    def groups_attwords(self,doc,doc_word_att,doc_id,target):
        """remove important words and create new documents for completeness&sufficient metrics"""
        doc_len = len(doc)
        bins = [0.01,0.05,0.1,0.2,0.5]
        print("Generate Completeness&Sufficient dataset for EraserMovie")
        ofile_dir = "/home/hanq1yanwarwick/SINE/input_data/{}/variants/{}/original_set/".format(self.dataset,self.variants)
        label_file = open("/home/hanq1yanwarwick/SINE/input_data/{}/variants/{}/original_set/label.txt".format(self.dataset,self.variants),"a+")
        if not os.path.isdir(ofile_dir):
            os.mkdir(ofile_dir)  
        for k, level in enumerate(bins):
            print("The level is %.4f"%level)
            sfile_dir = "/home/hanq1yanwarwick/SINE/input_data/{}/variants/{}/sset/bin".format(self.dataset,self.variants)+str(k)+"/"
            cfile_dir = "/home/hanq1yanwarwick/SINE/input_data/{}/variants/{}/cset/bin".format(self.dataset,self.variants)+str(k)+"/"
            # rcfile_dir = "/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/SINE/rcset/bin"+str(k)+"/"
            # rsfile_dir = "/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/SINE/rsset/bin"+str(k)+"/"
            if not os.path.isdir(sfile_dir):
                print("Save New Files for Intepretability Metrics")
                os.mkdir(sfile_dir)
                os.mkdir(cfile_dir)
                # os.mkdir(rcfile_dir)
                # os.mkdir(rsfile_dir)
            s_file = open(sfile_dir+str(doc_id)+".txt","w")
            c_file = open(cfile_dir+str(doc_id)+".txt","w")
            # rc_file = open(rcfile_dir+str(doc_id)+".txt","w")
            # rs_file = open(rsfile_dir+str(doc_id)+".txt","w")
            if k == 0:
                o_file = open(ofile_dir+str(doc_id)+".txt","w")
            for sen_id in range(doc_len):
                #use topK words, K is the same for each model
                deleted_num = max(int(len(doc_word_att[sen_id])*level),1)
                important_idx = list(np.array(doc_word_att[sen_id]).argsort()[-deleted_num:][::-1])
                #use the topK important words
                # threshold = np.percentile(np.array(doc_word_att[sen_id]),100*(1-level))
                #calculate the  deleted number
                # deleted_idx = np.where(np.array(doc_word_att[sen_id])>threshold)
                # deleted_num = deleted_idx[0].shape[0]
                random_ids = random.choices(range(len(doc[sen_id])), k=deleted_num)
                for i in range(len(doc_word_att[sen_id])):
                    if k==0:
                        o_file.write(doc[sen_id][i]+" ")
                    # if doc_word_att[sen_id][i]>threshold:
                    if i in important_idx:
                        s_file.write(doc[sen_id][i]+" ")
                    elif i not in important_idx:
                        c_file.write(doc[sen_id][i]+" ")
                    # if i not in random_ids:
                        # rc_file.write(doc[sen_id][i]+" ")
                    # elif i in random_ids:
                        # rs_file.write(doc[sen_id][i]+" ")
                s_file.write("\n")
                c_file.write("\n")
                # rs_file.write("\n")
                # rc_file.write("\n")
                if k==0:
                    o_file.write("\n")
            s_file.close()
            c_file.close()
            o_file.close()
            # rc_file.close()
            # rs_file.close()
        label_file.writelines(str(target)+"\n")
        print("")

    def genEx(self,doc_words,doc_att,sen_att,sen_label,logit,co_topic_weight,topic_weight_matrix,output_file,doc_id,target,doc_tokenids,doc_wordpos_ids,dis_encoder,id2word,gen_dataset = None,doc_name_list=None):
        """
        all the input are list of N elements for N sentences in a document
        doc_words:
        doc_att:
        doc_btt: """
        #add sentence topic assignment
        sen_att = sen_att.transpose(1,0)
        assert len(doc_att)==len(doc_words)==len(sen_label)==sen_att.shape[0] #doc_len
        pre_label = torch.argmax(logit,-1).detach().cpu().numpy()
        f = open(output_file,"a+")
        bsize = len(doc_words[0])
        target = target.detach().cpu().numpy()
        co_topic_weight = co_topic_weight.detach().cpu().numpy()
        topic_weight_matrix = topic_weight_matrix.detach().cpu().numpy()
        anots =  []
        for bs in range(bsize):
            #get the sentences belonging to 2 different topics
            data = co_topic_weight[bs,:,:]
            mask = np.triu(np.ones_like(data, dtype=bool))
            new_mask = (mask==False)
            new_data = new_mask*data
            ptopic_senids = np.unravel_index(np.argmax(new_data), new_data.shape)
            stopic_senid = np.argmin(data[ptopic_senids[0],:])
            topic_vec = np.argmax(topic_weight_matrix[bs,:,:],-1) #(5,)
            topic1_id = np.argmax(topic_weight_matrix[bs,ptopic_senids[0]])
            topic2_id = np.argmax(topic_weight_matrix[bs,stopic_senid])
            doc_id += 1
            print(doc_id)
            doc, doc_word_att, doc_sen_att,sen_labels,doc_token_id,doc_wpos_id = [],[],[],[], [],[]
            for doc_bs, att_bs, psen_att, label_bs,doc_id_bs,doc_wpid_bs in zip(doc_words,doc_att,sen_att,sen_label,doc_tokenids,doc_wordpos_ids):
                assert len(doc_bs[bs]) == len(att_bs[bs])
                doc.append(doc_bs[bs])
                doc_word_att.append(att_bs[bs])#will filter some unimportant word att
                doc_sen_att.append(psen_att[bs].detach().cpu().numpy().tolist())
                sen_labels.append(label_bs[bs])
                doc_token_id.append(doc_id_bs[bs])
                doc_wpos_id.append(doc_wpid_bs[bs])
            # if gen_dataset is not None:
            self.groups_attwords(doc,doc_word_att,doc_id,target[bs])
            # else:
            # level = 0.5 #control the percentage of important words
            # print(doc_name_list[bs])
            # rat_list = gen_hard_rats(doc,doc_word_att,level,doc_wpos_id)
            # anot = {"annotation_id":doc_name_list[bs],"rationales":[{"docid":doc_name_list[bs],"hard_rationale_predictions":rat_list}]}

            #generate latex file
            # if len(doc)>3 and len(doc)<10:
            #     self.doc_wordcloud_all(doc_token_id,doc_sen_att,topic_vec,dis_encoder,id2word,doc_id)
            #     # self.doc_wordcloud(ptopic_senids,stopic_senid,topic1_id,topic2_id,doc_token_id,dis_encoder,id2word,doc_id)
            #     generate(doc, doc_word_att, doc_sen_att,sen_labels,"/home/hanq1yanwarwick/SINE/latex_files/case_study/yelp/Sine_doc{}_casestudy.tex".format(doc_id), sen_color='green',word_color='red',pre_label=pre_label[bs],target=target[bs])
            # anots.append(anot)
        return doc_id,None

    def doc_wordcloud_all(self,doc_token_id,doc_sen_att,topic_vec,dis_encoder,id2word,doc_id):
        #calculate topic importance
        topic_weight = {}
        for sid,tid in enumerate(topic_vec):
            if tid not in topic_weight:
                topic_weight[tid] = doc_sen_att[sid]
            else:
                topic_weight[tid] += doc_sen_att[sid]

        encoder_weight = dis_encoder.detach().cpu().numpy() 
        #select 50 words for wordcloud
        wordcloud = WordCloud(background_color="white")
        # k = 50
        #build local vo
        
        for tid in topic_weight.keys():
            doc_vocab = set()
            #determine the sen_ids belonging to the tid
            string = " "
            sid_list = []
            for sid,tmpid in enumerate(topic_vec):
                if tmpid == tid:
                    sid_list.append(sid)
                    doc_vocab.update(doc_token_id[sid])
                    string = string + "S"+str(sid+1)+" "
            #filter the words in doc_vocab
            local_vocab = list(doc_vocab)
            twords_weight = encoder_weight[local_vocab,tid]
            wc_dict = {}
            for idx in range(len(local_vocab)):
                wc_dict[id2word[local_vocab[idx]]] = twords_weight[idx]
            # if len(wc_dict)<4:
                # continue
            try:  
                wordcloud.generate_from_frequencies(frequencies=wc_dict)
            except:
                print("fail to generate wordcloud")
                continue
            plt.figure()
            plt.title("Topic from {} with weight {}".format(string,round(topic_weight[tid],4)),fontsize=20)
            plt.axis("off")
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.savefig("./topic_wc/case_study/yelp/DocID{}_t{}_weight_{}_casestudy.png".format(doc_id,tid,round(topic_weight[tid],4)),dpi=300)
            plt.close()

    def doc_wordcloud(self,ptopic_senids,stopic_senid,topic1_id,topic2_id,doc_token_id,dis_encoder,id2word,doc_id):
        #humanEva: only 2 topics or case study: all the sentences assigned to a topic, need a loop. 
        # topic vector sent to classifier for polarity, sentence importance aggregate for the topic size. then generate topics from larger to smaller.
        doc_mask = []
        doc_vocab = set()
        encoder_weight = dis_encoder.detach().cpu().numpy()
        for sen_id in range(len(doc_token_id)):
            doc_vocab.update(doc_token_id[sen_id])
        for mask_i in range(15000):
            if mask_i in doc_vocab:
                doc_mask.append(1)
            else:
                doc_mask.append(0)
        t1_vocab = encoder_weight[:,topic1_id]*doc_mask
        t2_vocab = encoder_weight[:,topic2_id]*doc_mask
        #create word_weight dict for wordcloud
        t1_wordcloud,t2_wordcloud = {},{}
        stopwords = set(STOPWORDS)
        for wid,w_t1weight in enumerate(t1_vocab):
            if w_t1weight > 0.001 and id2word[wid] not in stopwords:
                t1_wordcloud[id2word[wid]] = w_t1weight
            if t2_vocab[wid] > 0.001 and id2word[wid] not in stopwords:
                t2_wordcloud[id2word[wid]] = t2_vocab[wid]
        # t1_weights = list(t1_wordcloud.values())
        wordcloud = WordCloud(background_color="white",width=400,height=200,max_font_size=65,min_font_size=15)
        wordcloud.generate_from_frequencies(frequencies=t1_wordcloud)
        plt.figure()
        plt.title("Topic1 from S{} S{}".format(ptopic_senids[0],ptopic_senids[1]))
        plt.axis("off")
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.savefig("./topic_wc/IMDB/IMDB_DocID{}_t1_wordcloud.png".format(doc_id),dpi=300)
        plt.close()
        wordcloud = WordCloud(background_color="white")
        wordcloud.generate_from_frequencies(frequencies=t2_wordcloud)
        plt.figure()
        plt.title("Topic2 from S{}".format(stopic_senid))
        plt.axis("off")
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.savefig("./topic_wc/IMDB/IMDB_DocID{}_t2_wordcloud.png".format(doc_id),dpi=300)

    def gatlayer(self,in_feats,att):
        update_context_rep = torch.bmm(att,in_feats)
        return update_context_rep
    
    def forward(self, input_list, input_tfidf,wordpos_list,length_list,flag=None,id2word=None,output_file=None,doc_id=0,target=None,dis_encoder=None,dis_decoder=None,doc_name=None):
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
        # word-level rnn
        word_rnn_hidden = self.init_rnn_hidden(batch_size, level="word")
        word_rnn_output_list = []
        word_aspect_output_list = []
        aspect_loss = torch.zeros(batch_size).cuda(self.device)
        kld_loss = torch.zeros(batch_size).cuda(self.device)
        recon_loss = torch.zeros(batch_size).cuda(self.device)
        doc_words, doc_att, doc_btt,sen_label,doc_tokenids,doc_wordpos_ids = [],[],[],[],[],[]
        for utterance_index in range(num_utterance):
            """for the context-learning"""
            word_rnn_input = self.embedding(input_list[utterance_index])
            word_rnn_input = self.word_dropout(word_rnn_input)
            word_rnn_output, word_rnn_hidden = self.word_rnn(word_rnn_input, word_rnn_hidden)
            if self.args.context_att > 0:
                word_attention_weight = self.word_conv_attention_layer(word_rnn_input.permute(1,2,0))
                word_attention_weight = word_attention_weight[:,:,1:-1]
                word_attention_weight = word_attention_weight.permute(2, 0 ,1)
                word_attention_weight = self.word_conv_attention_linear(word_attention_weight)
                word_attention_weight = nn.functional.relu(word_attention_weight)
                word_attention_weight = nn.functional.softmax(word_attention_weight, dim=0)#[len,bs,1]
                word_rnn_last_output = torch.mul(word_rnn_output,word_attention_weight).sum(dim=0)
            #sentiment representation of the sentence,s
            else:
                word_rnn_last_output = torch.mean(word_rnn_output,dim=0)
            word_rnn_output_list.append(word_rnn_last_output)
            word_rnn_hidden = word_rnn_hidden.detach()
            """for the topic-learning"""
            if self.args.topic_learning == "autoencoder":
                word_aspect_weight = self.word_aspect_attention_linear(word_rnn_output) 
                word_aspect_weight = self.word_aspect_attention_linear2(word_aspect_weight)
                word_aspect_weight = nn.functional.relu(word_aspect_weight)
                word_aspect_weight = nn.functional.softmax(word_aspect_weight, dim=0)
            elif self.args.topic_learning=="bayesian":
                # aggregate the sentence represntation by sent_tfidf [bs,seq_len] [bs,seq_len,dim]
                if self.args.topic_weight == "tfidf":
                    sent_tfidf = input_tfidf[utterance_index]
                    vae_input = sent_tfidf.unsqueeze(2).repeat(1,1,word_rnn_input.shape[-1])*word_rnn_input
                elif self.args.topic_weight == "average":
                    vae_input = word_rnn_input
                    sent_tfidf = torch.ones_like(input_tfidf[utterance_index]).cuda(self.device)

                word_aspect_weight,vae_kld_loss,vae_recon_loss = self.Vae(vae_input,sent_tfidf.unsqueeze(2))
                kld_loss += vae_kld_loss
                recon_loss += vae_recon_loss

            word_aspect_output = torch.mul(word_rnn_input,word_aspect_weight).sum(dim=0)#sum along seq_len axis
            word_aspect = self.topic_encoder(word_aspect_output)#latent rep
            recons_word = self.topic_decoder(word_aspect)#z'
            r = nn.functional.normalize(recons_word)#recon topi rep
            z = nn.functional.normalize(word_aspect_output)#original topic rep
            n = nn.functional.normalize(word_rnn_last_output)#context rep

            if self.args.regularization > 0:
                y = torch.ones(batch_size).cuda(self.device) - torch.sum(r*z, 1) + torch.sum(r*n, 1) #
                aspect_loss += nn.functional.relu(y) #why use a relu 
            word_aspect_output_list.append(word_aspect)

            """gen_wordlevel_explanations"""
            if flag == "gen_ex":
                #get alpha, omega
                sen_list,doc_word_att,sen_token_list,sen_wordpos_list =self.generate_wordEx(input_list[utterance_index],word_attention_weight,word_aspect_weight,id2word,output_file,wordpos_list[utterance_index])
                doc_words.append(sen_list)
                doc_att.append(doc_word_att)
                doc_tokenids.append(sen_token_list)
                doc_wordpos_ids.append(sen_wordpos_list)
                #get sentiment label for sentence
                label=self.classifier(word_rnn_last_output)#[bs,2]
                sen_label.append(torch.argmax(label,dim=-1).detach().cpu().numpy())
        """organize the sentence in each batch size and write out"""
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
        strategy = 'normalized_topic' #
        if strategy == 'raw':
            topic_weight_matrix =  context_topic_weight
        elif strategy == 'normalized_topic':
            topic_norm = torch.norm(context_topic_weight, p=2, dim=-1, keepdim=True)
            norm_topic_weight = context_topic_weight/topic_norm
            topic_weight_matrix = norm_topic_weight

        co_topic_weight = nn.functional.softmax(torch.bmm(topic_weight_matrix,topic_weight_matrix.transpose(2,1)/self.args.tsoftmax),2)
        in_feats = context_rnn_output
        for layer in range(self.args.glayers):
            in_feats=self.gatlayer(in_feats,co_topic_weight)
        # update_context_rep = self.output_gate*torch.bmm(co_topic_weight,
        mean_context_output = torch.mean(in_feats,dim=1)
        classifier_input = mean_context_output
        classifier_input_array = np.array(classifier_input.cpu().data)
        logit = self.classifier(classifier_input)
        #calculate the sentence importance according to the co_topic_weight
        sen_weight = torch.softmax(torch.sum(co_topic_weight,dim=1),dim=-1) #[bs,doc_len]
        if flag == "gen_ex":
            doc_id,anots=self.genEx(doc_words,doc_att,sen_weight,sen_label,logit,co_topic_weight,topic_weight_matrix,output_file,doc_id,target,doc_tokenids,doc_wordpos_ids,dis_encoder,id2word,gen_dataset=True,doc_name_list=doc_name)
        #attention_weight_array = np.array(csontext_attention_weight.data.cpu().squeeze(-1)).transpose(1,0)
        attention_weight_array = 0
        return logit,attention_weight_array,classifier_input_array,aspect_loss,co_topic_weight,self.args.vae_scale*kld_loss.mean(), recon_loss.mean(),doc_id,None

    