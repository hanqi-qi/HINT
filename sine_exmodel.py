
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sine_bayesian import Vae
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import STOPWORDS
# from docx import Document
# from docx.shared import RGBColor
# from matplotlib.cbook import get_sample_data

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
        self.num_label = args.num_label
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
                                        nn.Tanh())
        self.device = args.cuda
        if pretrained_embedding is not None:
            self.embedding.weight.data = self.embedding.weight.data.new(pretrained_embedding)
        else:
            self.embedding.weight.data.uniform_(-1.0, 1.0)
        
        self.Vae = Vae(self.emb_size,self.d_t,self.device)
        self.stopwords = set(STOPWORDS)

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
    def generate_wordEx(self,sent,alpha,omega,id2word,output_file):
        """sent: wordids [sen_len,bs]
            alpha: word context attention [sen_len,bs,1]
            omega: word topic attention [sen_len,bs,1]"""
        outfile = open(output_file,"a+")
        text = sent.detach().cpu().numpy()
        alpha = alpha.squeeze(-1).detach().cpu().numpy()
        omega = omega.squeeze(-1).detach().cpu().numpy()
        doc_list, awords_list, bwords_list = [], [], []
        for bs in range(text.shape[1]):
            if len((text[:,bs]==2).nonzero()[0]) > 0 :
                doc_len = (text[:,bs]==2).nonzero()[0][0]
            else:
                doc_len = len(text[:,bs])
            doc =  []
            for wid in text[:,bs]:
                if wid>2:
                    doc.append(id2word[wid])
            doc_list.append(doc)
            # outfile.writelines("DOCUMENT:\n%s\n"%(" ".join(doc)))
            k = 5
            a_wordidx = alpha[:doc_len,bs].argsort()[-k:][::-1]
            b_wordidx = omega[:doc_len,bs].argsort()[-k:][::-1]
            # outfile.writelines("Context WORDS:\n")
            awords,bwords = [],[]
            for wpos in a_wordidx:
                if text[wpos,bs] > 2 and id2word[text[wpos,bs]] not in self.stopwords:
                    awords.append(id2word[text[wpos,bs]])
                # outfile.writelines(",".join(awords))
                # outfile.writelines("\n")
            # outfile.writelines("Aspect WORDS:\n")
            awords_list.append(awords)
            for wpos in b_wordidx:
                if text[wpos,bs] > 2 and id2word[text[wpos,bs]] not in self.stopwords:
                    bwords.append(id2word[text[wpos,bs]])
                # outfile.writelines(",".join(bwords))
                # outfile.writelines("\n")
            # outfile.writelines("\n\n")
            bwords_list.append(bwords)
        return doc_list,awords_list,bwords_list
    
    def symbol(self,input):
        if input>0:
            symbol = r"$\bigstar$"
        else:
            symbol = r"$\blacksquare$"
        return symbol

    def genEx(self,doc_words,doc_att,doc_btt,doc_label,logit,co_topic_weight,output_file,doc_id,target):
        """
        all the input are list of N elements for N sentences in a document
        doc_words:
        doc_att:
        doc_btt: """
        assert len(doc_att)==len(doc_words)==len(doc_btt)==len(doc_label)
        pre_label = torch.argmax(logit,-1).detach().cpu().numpy()
        f = open(output_file,"a+")
        bsize = len(doc_words[0])
        co_topic_weight = co_topic_weight.detach().cpu().numpy()
        #save a sentence attention heatmaps
        #select the most important/representative sentences from data
        target = target.detach().cpu().numpy()
        for bs in range(bsize):
            doc_str = ""
            ptopic_str = " T1: {"
            stopic_str = ""
            doc_id += 1
            data = co_topic_weight[bs,:,:]
            mask = np.triu(np.ones_like(data, dtype=bool))
            new_mask = (mask==False)
            new_data = new_mask*data
            ptopic_senids = np.unravel_index(np.argmax(new_data), new_data.shape)
            stopic_senid = np.argmin(data[ptopic_senids[0],:])
            senids = range(data.shape[1])#all the sentence labels
            f.writelines("DocID{}:\n".format(doc_id))
            print(doc_id)
            sen_id = 0
            for doc_bs, att_bs, btt_bs, label_bs in zip(doc_words,doc_att,doc_btt,doc_label):
                if len(doc_bs)>bs:
                    f.writelines("SID"+str(sen_id)+": "+" ".join(doc_bs[bs])+"("+"Context: "+",".join(att_bs[bs])+"/"+"Topic: "+",".join(btt_bs[bs])+"/"+"sen_label: "+str(label_bs[bs])+")"+"\n")
                    if sen_id in ptopic_senids:
                        # sen_label = label_bs[bs]
                        symbol = self.symbol(label_bs[bs])
                        doc_text = " ".join(doc_bs[bs])
                        doc_str += symbol+" S{}: ".format(sen_id)+doc_text+" ("+" ".join(att_bs[bs])+") "+"\n"
                        ptopic_str += "S{}: ".format(sen_id)+" ".join(att_bs[bs])+ "/ "
                    elif sen_id == stopic_senid:
                        sen_label = label_bs[bs]
                        symbol = self.symbol(label_bs[bs])
                        doc_text = " ".join(doc_bs[bs])
                        doc_str += symbol+" S{}: ".format(sen_id)+doc_text+" ("+" ".join(att_bs[bs])+") "+"\n"
                        stopic_str += " T2: { "+"S{}: ".format(sen_id)+" ".join(att_bs[bs])+"} "+"\n"
                    else:
                        symbol = self.symbol(label_bs[bs])
                        # symbol =  r"$\odot$"
                        doc_str += symbol+" S{}: ".format(sen_id)+" ".join(doc_bs[bs])+" ("+" ".join(att_bs[bs])+") "+"\n"
                    sen_id+=1

            f.writelines("Predict Document Label: %d\n"%pre_label[bs])
            f.writelines("GT Document Label: %d\n"%target[bs])
            f.writelines("\n")
            symbol = self.symbol(pre_label[bs])
            ptopic_str += "}"+"\n"
            doc_str += ptopic_str + stopic_str
            doc_str += "Predict Document Label: "+symbol+"\n"
            symbol = self.symbol(target[bs])
            doc_str += "GT Document Label: "+symbol+"\n"
            # if (sen_label == pre_label[bs]) and (pre_label[bs]==target[bs]):
            fig, axs = plt.subplots(2, 1,gridspec_kw={
                        'height_ratios': [1,2]})
            plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5, wspace=0.4)
            # sns.heatmap(data, linewidth=0.3,annot=True,cmap="YlGn",ax=axs[0],mask=mask)
            axs[0].axis('off')
            axs[1].text(0.01,0,doc_str,multialignment="left",wrap=True)
            axs[1].axis('off')
            #insert the wordcloud pic, should be the most predominent one. 
            # im = plt.imread(get_sample_data("/mnt/sda/media/Data2/hanqi/sine/sen_att/107.png"))
            plt.savefig("/mnt/sda/media/Data2/hanqi/sine/sine_ex/human_evaluation/fullLabelAspect_yelp_woMean_{}".format(doc_id))
            # plt.clf()
            plt.close()
            # document.add_picture("/mnt/sda/media/Data2/hanqi/sine/sen_att/{}.png".format(doc_id))
            # document.save('/mnt/sda/media/Data2/hanqi/sine/sen_att/orientation_test{}.docx'.format(doc_id))
        return doc_id
        

    
    def forward(self, input_list, input_tfidf,length_list,flag=None,id2word=None,output_file=None,doc_id=0,target=None):
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
        doc_words, doc_att, doc_btt,sen_label = [],[],[],[]
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
                sen_list,awords_list,bwords_list=self.generate_wordEx(input_list[utterance_index],word_attention_weight,word_aspect_weight,id2word,output_file)
                doc_words.append(sen_list)
                doc_att.append(awords_list)
                doc_btt.append(bwords_list)
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

        # co_topic_weight = nn.functional.softmax(torch.bmm(self.key_linear(topic_weight_matrix),self.query_linear(topic_weight_matrix).transpose(2,1)/self.args.tsoftmax),2)
        co_topic_weight = nn.functional.softmax(torch.bmm(topic_weight_matrix,topic_weight_matrix.transpose(2,1)),2)#sum is 1 for each row
        update_context_rep = torch.bmm(co_topic_weight,context_rnn_output)
        # update_context_rep = self.output_gate*torch.bmm(co_topic_weight,context_rnn_output)+(1-self.output_gate)*context_rnn_output #[bs,doc_len,dim]
        use_nolinear = False #apply non-linear to the current graph layer to update the next layer
        if use_nolinear:
            update_context_rep = nn.functional.relu(update_context_rep)
        #yhq: Replace with mean as the summation
        # mean_context_output = torch.mean(update_context_rep,dim=1)
        #yhq using the 
        mean_context_output = torch.sum(update_context_rep,dim=1)
        classifier_input = mean_context_output
        # context_rnn_last_output = torch.mean(context_rnn_transform,1) #average the 
        # classifier_input = context_rnn_last_output
        classifier_input_array = np.array(classifier_input.cpu().data)
        logit = self.classifier(classifier_input)
        #calculate the sentence importance according to the co_topic_weight
        # dd = [torch.diagonal(co_topic_weight[i]) for i in range(co_topic_weight.shape[0])]
        # sen_weight = torch.softmax(torch.sum(co_topic_weight,dim=1)) #[bs,doc_len]
        if flag == "gen_ex":
            doc_id=self.genEx(doc_words,doc_att,doc_btt,sen_label,logit,co_topic_weight,output_file,doc_id,target)
        #attention_weight_array = np.array(csontext_attention_weight.data.cpu().squeeze(-1)).transpose(1,0)
        attention_weight_array = 0
        return logit,attention_weight_array,classifier_input_array,aspect_loss,co_topic_weight,self.args.vae_scale*kld_loss.mean(), recon_loss.mean(),doc_id

    