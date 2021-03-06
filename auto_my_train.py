import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import *
from auto_my import *
import random
import numpy as np
import matplotlib.pyplot as plt

VOCAB_SIZE = 7
EMBEDDING_SIZE = 10
HIDDEN_SIZE = 50
OUTPUT_SIZE = 2
NUM_LAYERS = 1
NUM_EPOCHS = 50
BATCH_SIZE = 100
LEARNING_RATE = 0.01

BAR_WIDTH = 0.72

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
	return X1, X2, y

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
	#return autograd.Variable(X1), autograd.Variable(X2), autograd.Variable(y)
	return X1, X2, y

def accuracy(predictions, labels):
	return ((predictions.max(1)[1] == labels).double().sum() / predictions.size(0)).data[0].item()

if (__name__ == '__main__'):
	train_dataset = LanguagesDataset('data/train1.txt')
	data_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
	test_dataset = LanguagesDataset('data/test1.txt')
	while (True):
		model = LookupTableMy(VOCAB_SIZE, EMBEDDING_SIZE, OUTPUT_SIZE)
		nll = nn.NLLLoss()
		optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
		X1_train, X2_train, y_train = process2(train_dataset)
		X1_test, X2_test, y_test = process2(test_dataset)
		train_epoch_accuracy = []
		test_epoch_accuracy = []
		epochs = np.arange(1, NUM_EPOCHS+1)
		for epoch in epochs:
			for batch in data_loader:
				X1_batch, X2_batch, y_batch = process(batch)
				model.zero_grad()
				logits = model(X1_batch, X2_batch)
				loss = nll(logits, y_batch)
				print('Loss = ' + str(loss.data[0].item()))
				loss.backward()
				optimizer.step()
			train_accuracy = accuracy(model(X1_train, X2_train), y_train)
			test_accuracy = accuracy(model(X1_test, X2_test), y_test)
			train_epoch_accuracy.append(train_accuracy)
			test_epoch_accuracy.append(test_accuracy)
			print('Epoch: ' + str(epoch))
			print('Train accuracy: ' + str(train_accuracy))
			print('Test accuracy: ' + str(test_accuracy))
		if (test_accuracy > 0.99):
			plt.figure(1)
			plt.plot(epochs, train_epoch_accuracy, 'b')
			plt.plot(epochs, test_epoch_accuracy, 'k')
			plt.show()
			idx = random.sample(range(288), 3)
			for i in idx:
				model(X1_test[i].unsqueeze(0), X2_test[i].unsqueeze(0))
				fig, ax = plt.subplots(3, 1, sharex = True)
				ind = np.arange(1, 31)
				for t in range(3):
					input_gates = torch.squeeze(model.input_gates[t])
					forget_gates = torch.squeeze(model.forget_gates[t])
					output_gates = torch.squeeze(model.output_gates[t])
					igs = ax[t].bar(3*ind - BAR_WIDTH, input_gates, BAR_WIDTH, color = 'b')
					fgs = ax[t].bar(3*ind, forget_gates, BAR_WIDTH, color = 'r')
					ogs = ax[t].bar(3*ind + BAR_WIDTH, output_gates, BAR_WIDTH, color = 'tab:purple')
					ax[0].set_xticks(3*ind)
					ax[0].set_xticklabels(str(j) for j in ind)
				plt.show()
			break