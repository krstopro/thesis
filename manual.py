import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class LookupTableManual(nn.Module):
	"""description"""
	
	def __init__(self, vocab_size, embedding_size, output_size):
		super(DoubleLSTM, self).__init__()
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.embedding = nn.Embedding(vocab_size, embedding_size)
		self.linear = nn.Linear(hidden_size, output_size)
	
	def init_hidden(self, batch_size):
		return (autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)),
		        autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)))
	
	def forward(self, input1, input2):
		batch_size = input1.size(0)
		es = []
		for i in range(5):
			es.append(self.embedding(input1.narrow(1, 4*i, 3)))
		embeddings1 = torch.stack(es, 1)
		
		es = []
		for i in range(18):
			es.append(self.embedding(input2.narrow(1, i, 3)))
		embeddings2 = torch.stack(es, 1)
		return None