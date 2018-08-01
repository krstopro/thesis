import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class AttentionDoubleGRU(nn.Module):
	"""description"""
	
	def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers = 1, max_length = 19):
		super(AttentionDoubleGRU, self).__init__()
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.embedding = nn.Embedding(vocab_size, embedding_size)
		# self.embedding.weight.requires_grad = False
		self.gru1 = nn.GRU(embedding_size, hidden_size, num_layers)
		self.attention = nn.Linear(embedding_size + hidden_size, max_length)
		# self.attention_combined = nn.Linear(embedding_size + hidden_size, hidden_size)
		self.attention_combined = nn.Linear(embedding_size + hidden_size, hidden_size)
		self.gru2 = nn.GRU(hidden_size, hidden_size, num_layers)
		self.linear = nn.Linear(hidden_size, output_size)
	
	def init_hidden(self, batch_size):
		return autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
	
	def forward(self, input1, input2):
		batch_size = input1.size(0)
		embeddings1 = self.embedding(input1).transpose(0, 1)
		hidden = self.init_hidden(batch_size)
		gru1_out, hidden = self.gru1(embeddings1, hidden)
		embeddings2 = self.embedding(input2).transpose(0, 1)
		for t in range(input2.size(1)):
			attention_weights = F.softmax(self.attention(torch.cat( (embeddings2[t], hidden[-1]), dim = 1)), dim = 1)
			attention_applied = torch.bmm(attention_weights.unsqueeze(1), gru1_out.transpose(0, 1))
			output = torch.cat( (embeddings2[t], attention_applied.squeeze(dim = 1)), dim = 1)
			output = F.relu(self.attention_combined(output)).unsqueeze(0)
			gru2_out, hidden = self.gru2(output, hidden)
		lin_out = self.linear(gru2_out[-1])
		logits = F.log_softmax(lin_out, dim = 1)
		return logits