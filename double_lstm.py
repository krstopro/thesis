import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class DoubleLSTM(nn.Module):
	"""description"""
	
	def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers = 1):
		super(DoubleLSTM, self).__init__()
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.embedding = nn.Embedding(vocab_size, embedding_size)
		self.lstm1 = nn.LSTM(embedding_size, hidden_size, num_layers)
		self.lstm2 = nn.LSTM(embedding_size, hidden_size, num_layers)
		self.linear = nn.Linear(hidden_size, output_size)
	
	def init_hidden(self, batch_size):
		return (autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)),
		        autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)))
	
	def forward(self, input1, input2):
		batch_size = input1.size(0)
		embeddings1 = self.embedding(input1).transpose(0, 1)
		hidden = self.init_hidden(batch_size)
		lstm1_out, hidden = self.lstm1(embeddings1, hidden)
		embeddings2 = self.embedding(input2).transpose(0, 1)
		# lstm2_out, (h, c) = self.lstm2(embeddings2, (h, autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))))
		lstm2_out, hidden = self.lstm2(embeddings2, hidden)
		lin_out = self.linear(lstm2_out[-1])
		logits = F.log_softmax(lin_out, dim = 1)
		return logits