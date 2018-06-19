import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class LookupTableManual(nn.Module):
	"""description"""
	
	def __init__(self, vocab_size, embedding_size, output_size, k_factors):
		super(LookupTableManual, self).__init__()
		self.embedding_size = embedding_size
		self.embedding = nn.Embedding(vocab_size, embedding_size)
		self.linear = nn.Linear(1, output_size)
		self.num_layers = 1
		self.hidden_size = embedding_size
		self.k_factors = k_factors
		self.lstm = nn.LSTM(embedding_size, self.hidden_size)
	
	def init_hidden(self, batch_size):
		return (autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)),
		        autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)))
	
	def forward(self, input1, input2):
		"""
		batch_size = input1.size(0)
		es = []
		for i in range(5):
			es.append(self.embedding(input1.narrow(1, 4*i, 3)))
		embeddings1 = torch.stack(es, 1)
		"""
		
		batch_size = input1.size(0)
		embeddings1 = self.embedding(input1).transpose(0, 1)
		hidden = self.init_hidden(batch_size)
		lstm_out, hidden = self.lstm(embeddings1, hidden)
		
		hs = []
		for i in range(self.k_factors):
			hs.append(lstm_out.transpose(0, 1).narrow(1, 4*i, 3))
		hiddens = torch.stack(hs, 1)
		
		es = []
		for i in range(18):
			es.append(self.embedding(input2.narrow(1, i, 3)))
		embeddings2 = torch.stack(es, 1)
		
		# ts = embeddings1.view(batch_size, 5, -1)
		ts = hiddens.view(batch_size, self.k_factors, -1)
		ps = embeddings2.view(batch_size, 18, -1).transpose(1, 2)
		
		rs = torch.bmm(ts, ps)
		ms = torch.max(rs, dim = 2)[0]
		ms = torch.max(ms, dim = 1)[0]
		logits = F.log_softmax(self.linear(ms.unsqueeze(1)))
		return logits
