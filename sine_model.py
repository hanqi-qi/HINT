
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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
        
        alpha = 1.0
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

        self.query_linear = nn.Linear(self.d_t,self.d_t)
        self.key_linear = nn.Linear(self.d_t,self.d_t)
        self.var_scale =1.0

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

    def forward(self, input_list, input_tfidf,length_list):
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
        for utterance_index in range(num_utterance):
            word_rnn_input = self.embedding(input_list[utterance_index])
            word_rnn_input = self.word_dropout(word_rnn_input)
            word_rnn_output, word_rnn_hidden = self.word_rnn(word_rnn_input, word_rnn_hidden)
            sent_tfidf = input_tfidf[utterance_index]
            tfidf_word_rnn_input = sent_tfidf.unsqueeze(2).repeat(1,1,word_rnn_input.shape[-1])*word_rnn_input
            aver_word_rnn_input = torch.mean(word_rnn_input,dim=0)
            if self.args.context_att > 0:
                word_attention_weight = self.word_conv_attention_layer(word_rnn_input.permute(1,2,0))
                word_attention_weight = word_attention_weight[:,:,1:-1]
                word_attention_weight = word_attention_weight.permute(2, 0 ,1)
                word_attention_weight = self.word_conv_attention_linear(word_attention_weight)
                word_attention_weight = nn.functional.relu(word_attention_weight)
                word_attention_weight = nn.functional.softmax(word_attention_weight, dim=0)
                word_rnn_last_output = torch.mul(word_rnn_output,word_attention_weight).sum(dim=0)
            #sentiment representation of the sentence,s
            else:
                word_rnn_last_output = torch.mean(word_rnn_output,dim=0)
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
            if self.args.regularization > 0:
                y = torch.ones(batch_size).cuda(self.device) - torch.sum(r*z, 1) + torch.sum(r*n, 1)
                aspect_loss += nn.functional.relu(y)
            # word_aspect_output_list.append(word_aspect)
            # word_rnn_hidden = word_rnn_hidden.detach()

            if self.args.topic_learning=="bayesian":
                # aggregate the sentence represntation by sent_tfidf [bs,seq_len] [bs,seq_len,dim]
                if self.args.topic_weight == "tfidf":
                    encoder_output = F.softplus(tfidf_word_rnn_input) #word embeddings
                elif self.args.topic_weight == "average":
                    encoder_output = F.softplus(aver_word_rnn_input)
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
                z = posterior_mean_bn + posterior_var.sqrt() * eps * self.args.var_scale
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

        co_topic_weight = nn.functional.softmax(torch.bmm(self.key_linear(topic_weight_matrix),self.query_linear(topic_weight_matrix).transpose(2,1)/self.args.tsoftmax),2)

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
        #attention_weight_array = np.array(csontext_attention_weight.data.cpu().squeeze(-1)).transpose(1,0)
        attention_weight_array = 0
        return logit,attention_weight_array,classifier_input_array,aspect_loss,co_topic_weight,kld_loss

        