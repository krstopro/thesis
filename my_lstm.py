import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math

class MyLSTMCell(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(MyLSTMCell, self).__init__()
		
		self.hidden_size = hidden_size
		stdv = 1.0 / math.sqrt(hidden_size)
		
		self.wx = nn.Linear(input_size, 4 * hidden_size, bias = False)
		self.wx.weight.data.uniform_(-stdv, stdv)
		
		self.wh = nn.Linear(hidden_size, 4 * hidden_size, bias = False)
		self.wh.weight.data.uniform_(-stdv, stdv)
		
		self.b = nn.Parameter(torch.zeros(4 * hidden_size))
	
	def forward(self, input, hidden):
		sf, si, so, sg = torch.split(self.wx(input) + \
		                         self.wh(hidden[0]) + \
		                         self.b, self.hidden_size, 1)
		f = F.sigmoid(sf)
		i = F.sigmoid(si)
		o = F.sigmoid(so)
		g = F.tanh(sg)
		c = f * hidden[1] + i * F.tanh(g)
		h = o * F.tanh(hidden[1])
		return h, c, i, f, o


class MyLSTM(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(MyLSTM, self).__init__()
		
		self.hidden_size = hidden_size
		self.lstm_cell = MyLSTMCell(input_size, hidden_size)
	
	def forward(self, input, hidden):
		hiddens = []
		input_gates = []
		forget_gates = []
		output_gates = []
		h, c = hidden
		for t in range(input.size(0)):
			h, c, i, f, o  = self.lstm_cell(input[t], (h, c))
			hiddens.append(h)
			input_gates.append(i)
			forget_gates.append(f)
			output_gates.append(o)
		return torch.stack(hiddens), (h, c), torch.stack(input_gates), torch.stack(forget_gates), torch.stack(output_gates)