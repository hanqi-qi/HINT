import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from matplotlib import pyplot as plt
import os

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
		sid = " S0"
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
				elif text[wpos,bs]==1:
					f.writelines("\n")
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

	def forward(self, x, p, flag,doc_id=None,id2word=None,labels=None,doc_words=None,doc_wpos_id=None,doc_name_list=None,level=0.2,doc_len=None):
		# return x
		
		if flag == 'train':
			probs = F.gumbel_softmax(p,hard=True,dim=2)[:,:,1:2]#original train
			mask = torch.ones_like(probs)
			x_prime = probs * x #x_prime is the new word embedding, after masking
			return x_prime,mask,probs,doc_id,None
		elif flag=='eval_mask':
			probs = F.softmax(p,dim=2)[:,:,1:2]
			_, word_indices = torch.topk(probs,30,dim=0)
			x_prime = probs * x
			return x_prime, word_indices, probs,doc_id,None
		elif flag == "gen_bin_datasets":
			probs = F.softmax(p,dim=2)[:,:,1:2]#[doc_len,bs,1]
			_, word_indices = torch.topk(probs,30,dim=0)
			x_prime = probs * x
			bins = [0.01,0.05,0.1,0.2,0.5]
			print("Generate Completeness&Sufficient dataset for EraserMovie")
			ofile_dir = "/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/VMASK/original_set/"
			if not os.path.isdir(ofile_dir):
				os.mkdir(ofile_dir)
			label_file = open("/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/VMASK/original_set/label.txt","a+")
			for idx in range(doc_words.shape[0]):
				doc_id = doc_id+1
				print("Doc id is %d"%doc_id)
				for k, level in enumerate(bins):
					# print("The level is %.4f"%level)
					sfile_dir = "/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/VMASK/sset/bin"+str(k)+"/"
					cfile_dir = "/home/hanq1yanwarwick/SINE/input_data/eraser_movie/movies/VMASK/cset/bin"+str(k)+"/"
					if not os.path.isdir(sfile_dir):
						print("Save New Files for Intepretability Metrics")
						os.mkdir(sfile_dir)
						os.mkdir(cfile_dir)
					s_file = open(sfile_dir+str(doc_id)+".txt","w")
					c_file = open(cfile_dir+str(doc_id)+".txt","w")
					if k == 0:
						o_file = open(ofile_dir+str(doc_id)+".txt","w")
					word_att = probs.squeeze(-1)[:,idx].detach().cpu().numpy()
					words = doc_words[idx].detach().cpu().numpy()
					threshold = np.percentile(np.array(word_att),100*(1-level))
					for (prob,w) in zip(word_att,words):
						if k==0:
							o_file.write(id2word[w]+" ")
						if prob > threshold:
							s_file.write(id2word[w]+" ")
						else:
							c_file.write(id2word[w]+" ")
					s_file.close()
					c_file.close()
					o_file.close()
				label_file.writelines(str(labels[idx])+"\n")
			return x_prime,word_indices,probs,doc_id,None
		elif flag == "gen_hard_rats":
			#clip then adopt the top
			probs = F.softmax(p,dim=2)[:,:,1:2]#[doc_len,bs,1]
			x_prime = probs * x
			mask = torch.ones_like(probs)
			anots = []
			for idx in range(doc_words.shape[0]):
				rat_list = []
				doc_id = doc_id+1
				new_prob = p[:,idx,1:2]
				clip_prob = F.softmax(new_prob,dim=0)
				words = doc_words[idx].detach().cpu().numpy()
				word_att = clip_prob.squeeze(-1).detach().cpu().numpy()
				deleted_num = max(int(len(word_att)*level),1)
				important_idx = list(np.array(word_att).argsort()[-deleted_num:][::-1])
				# threshold = np.percentile(np.array(word_att),100*(1-level))
				for wid in range(doc_len[idx]-1):
					if wid in important_idx:
					# if word_att[wid] > threshold:
						rat_list.append({'start_token': int(doc_wpos_id[idx][0,wid]), 'end_token': int(doc_wpos_id[idx][0,wid])+1})
				anot = {"annotation_id":doc_name_list[idx],"rationales":[{"docid":doc_name_list[idx],"hard_rationale_predictions":rat_list}]}
				anots.append(anot)
			return x_prime,mask,probs,doc_id,anots

		else:
			mask = F.softmax(p,dim=2)[:,:,1:2]#original evaluation
			x_prime = mask * x
			probs = p
			return x_prime,mask,probs,doc_id,None

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

	def forward(self, batch, flag,id2word=None,doc_id=None,doc_wpos_id=None,doc_name_list=None,rats_level=0.2,doc_len=None):
		# embedding

		x = batch.text.t()
		labels = batch.label.t()
		embed = self.lstmmodel.embed(x)
		embed = F.dropout(embed, p=self.args.dropout, training=self.training)
		x = embed.view(len(x), embed.size(1), -1)  # seqlen, bsz, embed-dim
		# MASK
		p = self.vmask.get_statistics_batch(x)
		if flag == "gen_bin_datasets":
			x_prime,_,probs,doc_id=self.vmask(x, p, flag,doc_id,id2word,labels.cpu().numpy(),doc_words = batch.text)
		elif flag == "gen_hard_rats":
			x_prime,_,probs,doc_id,anots=self.vmask(x, p, flag,doc_id,id2word,labels.cpu().numpy(),doc_words = batch.text,doc_wpos_id=doc_wpos_id,doc_name_list=doc_name_list,level=rats_level,doc_len=doc_len)
		else:
			x_prime,_,probs,_ = self.vmask(x, p, flag="eval_mask") #masked_x, masked_word_idx, probs

		# else:

		output = self.lstmmodel(x_prime) #[64,num_label]


		# # self.infor_loss = F.softmax(p,dim=2)[:,:,1:2].mean()
		probs_pos = F.softmax(p,dim=2)[:,:,1]
		probs_neg = F.softmax(p,dim=2)[:,:,0]
		self.infor_loss = torch.mean(probs_pos * torch.log(probs_pos+1e-8) + probs_neg*torch.log(probs_neg+1e-8))
		#extract the removed words
		# pre_label = torch.argmax(output,dim=-1)
		# if flag == 'eval_mask':
			# text,probs,id2word,pre_label,target,doc_id
		# doc_id=gen_explanations(batch.text.transpose(1,0),word_index,id2word,pre_label,batch.label,doc_id,output_file=output_file)

		return output,batch.text,probs,doc_id,anots
