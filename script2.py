import argparse
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_utils import *
from double_lstm import *
from double_gru import *
from attention_double_lstm import *
from attention_double_gru import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

VOCAB_SIZE = 7
OUTPUT_SIZE = 2
NUM_LAYERS = 1
NUM_EPOCHS = [None, 50, None, None, None, 50]
BATCH_SIZE = [None, 100, None, None, None, 1000]

char2index = { 'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, '#': 6, ' ': 7 }

def process(data):
	indices1 = []
	indices2 = []
	labels = []
	for i in range(len(data[0])):
		part1, part2 = data[0][i].split(' ')
		indices1.append([ char2index[c] for c in part1 ])
		indices2.append([ char2index[c] for c in part2 ])
		labels.append(int(data[1][i]))
	X1 = torch.LongTensor(indices1)
	X2 = torch.LongTensor(indices2)
	y = torch.LongTensor(labels)
	return autograd.Variable(X1), autograd.Variable(X2), autograd.Variable(y)

def process2(dataset):
	indices1 = []
	indices2 = []
	labels = []
	for i in range(len(dataset)):
		lang = dataset[i][0].split(' ')
		indices1.append([ char2index[c] for c in lang[0] ])
		indices2.append([ char2index[c] for c in lang[1] ])
		labels.append(int(dataset[i][1]))
	X1 = torch.LongTensor(indices1)
	X2 = torch.LongTensor(indices2)
	y = torch.LongTensor(labels)
	return autograd.Variable(X1), autograd.Variable(X2), autograd.Variable(y)

def accuracy(predictions, labels):
	return ((predictions.max(1)[1] == labels).double().sum() / predictions.size(0)).item()

if (__name__ == '__main__'):
	ap = argparse.ArgumentParser()
	ap.add_argument('-o', '--output', default='results')
	ap.add_argument('-s', '--seed', type=int, default=0)
	ap.add_argument('-e', '--embedding-size', type=int, default=10)
	ap.add_argument('-l', '--learning-rate', type=float, default=0.1)
	ap.add_argument('-m', '--model', choices=['lstm', 'gru', 'lnlstm', 'lngru', 'attn_lstm', 'attn_gru', 'attn_lnlstm', 'attn_lngru'])
	ap.add_argument('-k', '--k-factors', type=int, default=5)
	ap.add_argument('-n', '--num-layers', type=int, default=1)
	ap.add_argument('-d', '--hidden-size', type=int)
	args = ap.parse_args()

	os.makedirs(args.output, exist_ok=True)
	train_dataset = LanguagesDataset('data/train{}.txt'.format(args.k_factors))
	data_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE[args.k_factors], shuffle = False)
	test_dataset = LanguagesDataset('data/test{}.txt'.format(args.k_factors))
	nll = nn.NLLLoss()
	X1_train, X2_train, y_train = process2(train_dataset)
	X1_test, X2_test, y_test = process2(test_dataset)
	torch.manual_seed(args.seed)
	
	if args.model == 'lstm':
		model = DoubleLSTM(VOCAB_SIZE, args.embedding_size, args.hidden_size, OUTPUT_SIZE, args.num_layers)
	if args.model == 'gru':
		model = DoubleGRU(VOCAB_SIZE, args.embedding_size, args.hidden_size, OUTPUT_SIZE, args.num_layers)
	if args.model == 'lnlstm':
		raise Exception('Not yet added')
	if args.model == 'lngru':
		raise Exception('Not yet added')
	if args.model == 'attn_lstm':
		model = AttentionDoubleLSTM(VOCAB_SIZE, args.embedding_size, args.hidden_size, OUTPUT_SIZE, args.num_layers, 4 * k_factors - 1)
	if args.model == 'attn_gru':
		model = AttentionDoubleGRU(VOCAB_SIZE, args.embedding_size, args.hidden_size, OUTPUT_SIZE, args.num_layers, 4 * k_factors - 1)
	if args.model == 'attn_lnlstm':
		raise Exception('Not yet added')
	if args.model == 'atnn_lngru':
		raise Exception('Not yet added')
		
	optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
	train_epoch_accuracy = []
	test_epoch_accuracy = []
	epochs = np.arange(1, NUM_EPOCHS[args.k_factors]+1)
	for epoch in tqdm(epochs):
		for batch in data_loader:
			X1_batch, X2_batch, y_batch = process(batch)
			model.zero_grad()
			logits = model(X1_batch, X2_batch)
			loss = nll(logits, y_batch)
			# print('Loss = ' + str(loss.data[0]))
			loss.backward()
			optimizer.step()
		train_accuracy = accuracy(model(X1_train, X2_train), y_train)
		test_accuracy = accuracy(model(X1_test, X2_test), y_test)
		tqdm.write("train_acc={}, test_acc={}\r".format(train_accuracy, test_accuracy))
		train_epoch_accuracy.append(100 * train_accuracy)
		test_epoch_accuracy.append(100 * test_accuracy)
		# print('Epoch: ' + str(epoch))
		# print('Train accuracy: ' + str(train_accuracy))
		# print('Test accuracy: ' + str(test_accuracy))
	results = open(os.path.join(args.output, '{}.txt'.format(args.seed)), 'w')
	results.write(' '.join([ str(round(x, 2)) for x in train_epoch_accuracy]))
	results.write('\n')
	results.write(' '.join([ str(round(x, 2)) for x in test_epoch_accuracy]))
	results.write('\n')
	# plt.plot(epochs, train_epoch_accuracy, 'b')
	# plt.plot(epochs, test_epoch_accuracy, 'k')
	# plt.show()
	results.close()
	
	fig = plt.figure(1)
	ax = fig.add_subplot(111)
	ax.plot(epochs, train_epoch_accuracy, 'b', label = 'Train')
	ax.plot(epochs, test_epoch_accuracy, 'g', label = 'Test')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy (%)')
	lgd = ax.legend()
	fig.savefig('plots/' + args.model + '_' + str(args.num_layers) + '_' + \
				str(args.hidden_size) + '_' + str(round(100 * max(test_epoch_accuracy))) + \
				'.svg', bbox_extra_artists = [lgd], bbox_inches = 'tight')