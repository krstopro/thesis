import argparse
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import *
from auto import *
from auto_pos import *
from manual import *
import numpy as np
import matplotlib.pyplot as plt

VOCAB_SIZE = 7
OUTPUT_SIZE = 2
NUM_LAYERS = 1
NUM_EPOCHS = 100
BATCH_SIZE = 100
K_FACTORS = 1

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
	return ((predictions.max(1)[1] == labels).double().sum() / predictions.size(0)).data[0]

if (__name__ == '__main__'):
	ap = argparse.ArgumentParser()
	ap.add_argument('-o', '--output', default='results')
	ap.add_argument('-s', '--seed', type=int, default=0)
	ap.add_argument('-e', '--embedding-size', type=int, default=10)
	ap.add_argument('-l', '--learning-rate', type=float, default=0.1)
	ap.add_argument('-m', '--model', choices=['manual', 'auto', 'auto_pos'])
	args = ap.parse_args()

	os.makedirs(args.output, exist_ok=True)
	results = open(os.path.join(args.output, '{}.txt'.format(args.seed)), 'w')
	train_dataset = LanguagesDataset('data/train1.txt')
	data_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
	test_dataset = LanguagesDataset('data/test1.txt')
	nll = nn.NLLLoss()
	X1_train, X2_train, y_train = process2(train_dataset)
	X1_test, X2_test, y_test = process2(test_dataset)
	torch.manual_seed(args.seed)
	if args.model == 'auto':
	    model = LookupTableAuto(VOCAB_SIZE, args.embedding_size, OUTPUT_SIZE)
	elif args.model == 'auto_pos':
	    model = LookupTablePos(VOCAB_SIZE, args.embedding_size, OUTPUT_SIZE)
	elif args.model == 'manual':
	    model = LookupTableManual(VOCAB_SIZE, args.embedding_size, OUTPUT_SIZE, K_FACTORS)
	optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
	train_epoch_accuracy = []
	test_epoch_accuracy = []
	epochs = np.arange(1, NUM_EPOCHS+1)
	for epoch in epochs:
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
		train_epoch_accuracy.append(train_accuracy)
		test_epoch_accuracy.append(test_accuracy)
		# print('Epoch: ' + str(epoch))
		# print('Train accuracy: ' + str(train_accuracy))
		# print('Test accuracy: ' + str(test_accuracy))
	results.write(' '.join([ str(round(100 * x, 2)) for x in train_epoch_accuracy]))
	results.write('\n')
	results.write(' '.join([ str(round(100 * x, 2)) for x in test_epoch_accuracy]))
	results.write('\n')
	# plt.plot(epochs, train_epoch_accuracy, 'b')
	# plt.plot(epochs, test_epoch_accuracy, 'k')
	# plt.show()
	results.close()
