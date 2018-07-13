import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from my_lstm import MyLSTM

class LookupTablePosMy(nn.Module):
	"""description"""
	
	def __init__(self, vocab_size, embedding_size, output_size, k_factors):
		super(LookupTablePosMy, self).__init__()
		self.embedding_size = embedding_size
		self.k_factors = k_factors
		self.embedding = nn.Embedding(vocab_size, embedding_size)
		self.lstm = MyLSTM(2 * embedding_size, 3 * k_factors * embedding_size)
		self.linear = nn.Linear(1, output_size)
		self.pos_embedding = nn.Embedding(4 * k_factors - 1, embedding_size)
	
	"""
	def init_hidden(self, batch_size):
		return (autograd.Variable(torch.zeros(1, batch_size, 5 * 3 * self.embedding_size)),
		        autograd.Variable(torch.zeros(1, batch_size, 5 * 3 * self.embedding_size)))
	"""
	
	def init_hidden(self, batch_size):
		return (torch.zeros(batch_size, 3 * self.k_factors * self.embedding_size),
		        torch.zeros(batch_size, 3 * self.k_factors * self.embedding_size))
	
	def forward(self, input1, input2):
		"""
		batch_size = input1.size(0)
		es = []
		for i in range(5):
			es.append(self.embedding(input1.narrow(1, 4*i, 3)))
		embeddings1 = torch.stack(es, 1)
		"""
		
		batch_size = input1.size(0)
		seq_len = input1.size(1)
		embeddings1 = self.embedding(input1)
		positions = autograd.Variable(torch.LongTensor(np.tile(np.arange(seq_len), (batch_size, 1))))
		pos_embeddings = self.pos_embedding(positions)
		hidden = self.init_hidden(batch_size)
		x = torch.cat([embeddings1, pos_embeddings], 2).transpose(0, 1)
		lstm_out, hidden, self.input_gates, self.forget_gates, self.output_gates = self.lstm(x, hidden)
		self.hidden = hidden
		
		es = []
		for i in range(18):
			es.append(self.embedding(input2.narrow(1, i, 3)))
		embeddings2 = torch.stack(es, 1)
		
		ts = hidden[0].view(batch_size, self.k_factors, -1)
		ps = embeddings2.view(batch_size, 18, -1).transpose(1, 2)
		
		rs = torch.bmm(ts, ps)
		ms = torch.max(rs, dim = 2)[0]
		ms = torch.max(ms, dim = 1)[0]
		logits = F.log_softmax(self.linear(ms.unsqueeze(1)))
		return logits