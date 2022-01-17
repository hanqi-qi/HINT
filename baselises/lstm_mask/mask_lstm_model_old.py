import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from matplotlib import pyplot as plt

SMALL = 1e-08

def gen_explanations(text,word_index,id2word,pre_label,target,doc_id,output_file=None):
	text = text.detach().cpu().numpy()
	word_index = word_index.detach().cpu().numpy()
	f = open(output_file,"a+")
	for bs in range(word_index.shape[1]):
		sen_id = 0
		doc_str = ""
		doc_id += 1
		print(doc_id)
		doc= []
		if len((text[:,bs]==2).nonzero()[0]) > 0 :
			doc_len = (text[:,bs]==2).nonzero()[0][0]
		else:
			doc_len = len(text[:,bs])
		sid = "S{}".format(sen_id)
		doc.append(sid)
		for wid in text[:,bs]:
			if wid!=2:
				if wid==1:
					doc.append("\n")
					sen_id += 1
					sid = "S{}".format(sen_id)
					doc.append(sid)
				else:
					doc.append(id2word[wid])
		# doc = [id2word[wid] for wid in text[:,bs] if wid != 2 ]
		# print("Document:\n%s"%(" ".join(doc))) 
		doc_content = "DocID{}:\n{}".format(doc_id," ".join(doc))+"\n"
		f.writelines(doc_content)
		# doc_str += "Document{}:".format(doc_id)+"\n"+" ".join(doc)+"\n"
		# unmask_id = ((word_index[:,bs,0] == 1).nonzero())[0]
		masked_words = []
		# print("Important Words:\n")
		# doc_str += "Important Words:"+"\n"
		f.writelines("Important Words:"+"\n")
		unmask_id = word_index[:,bs,0]
		for wpos in unmask_id:
			if wpos > doc_len:
				break
			else:
				if text[wpos,bs]>2:
					masked_words = id2word[text[wpos,bs]]
			# print(masked_words,end="\t")
			# doc_str += masked_words+","
			f.writelines(masked_words+" ")
		# print("Predict Document Label: %d"%pre_label[bs])
		# print("GT Document Label: %d"%pre_label[bs])
		# print("########")
		# doc_str += "\n"+"Predict Document Label:{}".format(pre_label[bs])+"\n"+"GT Document Label: {}".format(target[bs])+"\n"
		f.writelines("\n"+"Predict Document Label:{}".format(pre_label[bs])+"\n"+"GT Document Label: {}".format(target[bs])+"\n\n")
		# fig, axs = plt.subplots(1, 1,figsize=(30,80))
		# # plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5, wspace=0.4)
		# # sns.heatmap(data, linewidth=0.3,annot=True,cmap="YlGn",ax=axs[0],mask=mask)
		# axs.text(0.01,0,doc_str,multialignment="left",wrap=True)
		# axs.axis('off')
		# plt.savefig("/mnt/Data3/hanqiyan/latent_topic/VMASK/vmask_ex/VMASK_{}".format(doc_id))
		# plt.close()
	f.close()
	return doc_id

class VMASK(nn.Module):
	def __init__(self, args):
		super(VMASK, self).__init__()
		self.args = args
		self.device = args.device
		self.mask_hidden_dim = args.mask_hidden_dim
		self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu, 'leaky_relu': F.leaky_relu}
		self.activation = self.activations[args.activation]
		self.embed_dim = args.embed_dim
		self.linear_layer = nn.Linear(self.embed_dim, self.mask_hidden_dim)
		self.hidden2p = nn.Linear(self.mask_hidden_dim, 2)
		self.ifmask = args.ifmask
		self.mask_k = args.mask_k

	def forward_sent_batch(self, embeds):

		temps = self.activation(self.linear_layer(embeds))
		p = self.hidden2p(temps)  # seqlen, bsz, dim
		return p

	def forward(self, x, p, flag):
		# return x
		
		if flag == 'train':
			probs = F.gumbel_softmax(p,hard=True,dim=2)[:,:,1:2] #r is the mask, return one-hot
			mask = torch.ones_like(probs)
			x_prime = probs * x #x_prime is the new word embedding, after masking
			return x_prime,mask,probs
		elif flag=='eval_mask':
			probs = F.softmax(p,dim=2)[:,:,1:2] #select the probs of being 1, like use a softmax [seq_len,bs,1]
			#use topk id to generate the x, calculate
			# if self.ifmask:
			# probs = F.gumbel_softmax(p,hard=True,dim=2)[:,:,1:2]
			_, word_indices = torch.topk(probs,30,dim=0)
			# word_indices = ((probs == 0).nonzero()) #[numberOfzero,3]
			x_prime = probs * x
			# mask = torch.ones_like(probs)
			# relevance_score = []
			# for bs in range(probs.shape[1]):
			# 	mask[word_indices[:,bs,0],bs,0] = 0
			# 	relevance_score.append(sum(probs[word_indices[:,bs,0],bs,0].cpu().detach().numpy()))
			# print('Relevance Score is %.4f'%(sum(relevance_score)/(bs+1)))
			return x_prime, word_indices, probs
				
		else:
			mask = F.softmax(p,dim=2)[:,:,1:2]
			x_prime = mask * x
			probs = p
			return x_prime,mask,probs

	def get_statistics_batch(self, embeds):
		p = self.forward_sent_batch(embeds)
		return p


class LSTM(nn.Module):
	def __init__(self, args, vectors):
		super(LSTM, self).__init__()

		self.args = args

		self.embed = nn.Embedding(args.embed_num, args.embed_dim, padding_idx=1)

		# initialize word embedding with pretrained word2vec
		#self.embed.weight.data.copy_(torch.from_numpy(vectors))

		# fix embedding
		if args.mode == 'static':
			self.embed.weight.requires_grad = False
		else:
			self.embed.weight.requires_grad = True

		# <unk> vectors is randomly initialized
		nn.init.uniform_(self.embed.weight.data[0], -0.05, 0.05)

		# <pad> vector is initialized as zero padding
		nn.init.constant_(self.embed.weight.data[1], 0)

		# lstm
		self.lstm = nn.LSTM(args.embed_dim, args.lstm_hidden_dim, num_layers=args.lstm_hidden_layer)
		# initial weight
		init.xavier_normal_(self.lstm.all_weights[0][0], gain=np.sqrt(6.0))
		init.xavier_normal_(self.lstm.all_weights[0][1], gain=np.sqrt(6.0))

		# linear
		self.hidden2label = nn.Linear(args.lstm_hidden_dim, args.class_num)
		# dropout
		self.dropout = nn.Dropout(args.dropout)
		self.dropout_embed = nn.Dropout(args.dropout)
		self.att_layer = nn.Linear(self.args.lstm_hidden_dim,1)

	def forward(self, x):
		# lstm
		lstm_out, _ = self.lstm(x) #lstm_out [seq_len, bs,output_dim]
		lstm_out = torch.transpose(lstm_out, 0, 1)
		lstm_out = torch.transpose(lstm_out, 1, 2)
		# pooling
		lstm_out = torch.tanh(lstm_out) #[bs,output_dim, seq_len]
		if self.args.use_lstmatt:
			word_seq = lstm_out.transpose(1,2)
			attention = F.softmax(self.att_layer(word_seq))
			lstm_out = torch.mul(word_seq,attention).sum(dim=1)
		else:
			lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2) #use the max output along the sequence length[64,100]
		lstm_out = torch.tanh(lstm_out)
		lstm_out = F.dropout(lstm_out, p=self.args.dropout, training=self.training)
		# linear
		logit = self.hidden2label(lstm_out)
		out = F.softmax(logit, 1)
		return out


		
		
class MASK_LSTM(nn.Module):

	def __init__(self, args, vectors):
		super(MASK_LSTM, self).__init__()
		self.args = args
		self.embed_dim = args.embed_dim
		self.device = args.device
		# self.sample_size = args.sample_size
		self.max_sent_len = args.max_sent_len
		self.mask_k = args.mask_k
		self.vmask = VMASK(args)
		self.lstmmodel = LSTM(args, vectors)
		
	
	def gen_explanations(self,text,probs,id2word,output_file,doc_id=None):
		"""
		text: word_ids [doc_len,bs]
		probs: word_mask [doc_len,bs]
		"""
		outfile = open(output_file,"a+")
		text = text.detach().cpu().numpy()
		probs = probs.detach().cpu().numpy()
		for bs in range(probs.shape[1]):
			if len((text[:,bs]==2).nonzero()[0]) > 0 :
				doc_len = (text[:,bs]==2).nonzero()[0][0]
				doc = [id2word[wid] for wid in text[:,bs] if wid >2 ]
				outfile.writelines("DOCUMENT:\n%s\n"%(" ".join(doc)))
				unmask_id = ((probs[:,bs,0] == 1).nonzero())[0]
				masked_words = []
				
				outfile.writelines("IMPORTANT WORDS:\n")
				for wpos in unmask_id:
					if wpos > doc_len:
						break
					else:
						if text[wpos,bs] > 2:
							masked_words.append(id2word[text[wpos,bs]])
				if len(masked_words) > 0:
					outfile.writelines(",".join(masked_words))
					outfile.writelines("\n\n")
				else:
					outfile.writelines("\n\n")

	def forward(self, batch, flag,id2word=None,doc_id=None,output_file=None):
		# embedding
		x = batch.text.t()
		embed = self.lstmmodel.embed(x)
		embed = F.dropout(embed, p=self.args.dropout, training=self.training)
		x = embed.view(len(x), embed.size(1), -1)  # seqlen, bsz, embed-dim
		# MASK
		p = self.vmask.get_statistics_batch(x)
		x_prime,word_index,probs = self.vmask(x, p, flag) #masked_x, masked_word_idx, probs


		# else:

		output = self.lstmmodel(x_prime) #[64,num_label]


		# # self.infor_loss = F.softmax(p,dim=2)[:,:,1:2].mean()
		probs_pos = F.softmax(p,dim=2)[:,:,1]
		probs_neg = F.softmax(p,dim=2)[:,:,0]
		self.infor_loss = torch.mean(probs_pos * torch.log(probs_pos+1e-8) + probs_neg*torch.log(probs_neg+1e-8))
		#extract the removed words
		pre_label = torch.argmax(output,dim=-1)
		# if flag == 'eval_mask':
			# text,probs,id2word,pre_label,target,doc_id
		# doc_id=gen_explanations(batch.text.transpose(1,0),word_index,id2word,pre_label,batch.label,doc_id,output_file=output_file)

		return output,batch.text,probs
