import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss
import logging
import numpy as np
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.modules.loss import NLLLoss

logger = logging.getLogger(__name__)

class Vae(nn.Module):
    def __init__(self,emb_size,d_t,device):
        super(Vae,self).__init__()
        alpha = 1.0
        self.emb_size = emb_size
        self.d_t = d_t
        self.device = device
        if np.array(alpha).size == 1:
            # if alpha is a scalar, create a symmetric prior vector
            self.alpha = alpha * np.ones((1, self.d_t)).astype(np.float32)
        else:
            # otherwise use the prior as given
            self.alpha = np.array(alpha).astype(np.float32)
            assert len(self.alpha) == self.emb_size

        self.encoder_dropout_layer = nn.Dropout(p=0.2)
        # create the mean and variance components of the VAE
        self.mean_layer = nn.Linear(self.emb_size,self.d_t )
        self.logvar_layer = nn.Linear(self.emb_size, self.d_t)

        self.mean_bn_layer = nn.BatchNorm1d(self.d_t, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.d_t))).to(self.device)
        self.mean_bn_layer.weight.requires_grad = False
        self.logvar_bn_layer = nn.BatchNorm1d(self.d_t, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.d_t))).to(self.device)
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

        prior_mean = np.array(prior_mean).reshape((1, self.d_t))
        prior_logvar = np.array(np.log(prior_var)).reshape((1, self.d_t))
        self.prior_mean = torch.from_numpy(prior_mean).to(self.device)
        self.prior_mean.requires_grad = False
        self.prior_logvar = torch.from_numpy(prior_logvar).to(self.device)
        self.prior_logvar.requires_grad = False

        self.concat_att = nn.Linear(self.emb_size+self.d_t,1)
    

    def _loss(self, prior_mean, prior_logvar, posterior_mean, posterior_logvar,do_average=True):
        #reconstruct_loss

        # compute KLD
        prior_var = prior_logvar.exp()
        posterior_var = posterior_logvar.exp()
        var_division    = posterior_var / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar

        # put KLD together
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.d_t)

        return KLD.mean(1)
    
    def forward(self,rnn_input):
    # aggregate the sentence represntation by sent_tfidf [bs,seq_len] [bs,seq_len,dim]
        encoder_output = F.softplus(rnn_input) #[len,bs,dim]
        encoder_output_do = self.encoder_dropout_layer(encoder_output)#[seq_len,bs,emb_size]
        # compute the mean and variance of the document posteriors
        posterior_mean = torch.transpose(self.mean_layer(encoder_output_do),1,0) #[seq_len,bs,emb_size]
        posterior_logvar = torch.transpose(self.logvar_layer(encoder_output_do),1,0)#[seq_len,bs,emb_size]

        posterior_mean_bn = self.mean_bn_layer(posterior_mean.transpose(1,2)).permute(0,2,1)#the batchnorm1d, input should be [bs,len,dim]
        posterior_logvar_bn = self.logvar_bn_layer(torch.transpose(posterior_logvar,1,2)).permute(0,2,1)#[[bs,len,dim]]

        posterior_var = posterior_logvar_bn.exp().to(self.device)

        # sample noise from a standard normal
        eps = encoder_output_do.data.new().resize_as_(posterior_mean_bn.data).normal_().to(self.device)
        #[]
        # compute the sampled latent representation
        z = posterior_mean_bn + posterior_var.sqrt() * eps
        z_do = self.z_dropout_layer(z) #[N,C]#latentc

        # pass the document representations through a softmax
        theta = F.softmax(z_do, dim=1)#this is omega [bs,len,300]
        recon_rnn_input= F.relu(self.beta_layer(theta))

        # recon_loss = -(rnn_input.transpose(1,0) * (recon_rnn_input+1e-10).log()).sum(-1)#sum along the seq_len axis
        recon_loss = F.mse_loss(rnn_input.transpose(1,0),recon_rnn_input)

        prior_mean   = self.prior_mean.expand_as(posterior_mean)
        prior_logvar = self.prior_logvar.expand_as(posterior_logvar)

        do_average = False
        KLD = self._loss(prior_mean, prior_logvar, posterior_mean_bn, posterior_logvar_bn, do_average)
        # kl_loss = nn.functional.relu(KLD)

        word_aspect_weight = self.concat_att(torch.cat((rnn_input,z_do.transpose(1,0)),-1)) #[len,bs,1]
        return word_aspect_weight,KLD.mean(),recon_loss.mean()