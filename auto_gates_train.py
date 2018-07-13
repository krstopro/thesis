import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import *
from auto_pos_my import *
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

VOCAB_SIZE = 7
EMBEDDING_SIZE = 10
HIDDEN_SIZE = 50
OUTPUT_SIZE = 2
NUM_LAYERS = 1
NUM_EPOCHS = 100
BATCH_SIZE = 100
LEARNING_RATE = 0.01

#GAMMA = 0
GAMMA = 1/60
#GAMMA = 1/6

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
	return X1, X2, y

def accuracy(predictions, labels):
	return ((predictions.max(1)[1] == labels).double().sum() / predictions.size(0)).data[0].item()

if (__name__ == '__main__'):
	train_dataset = LanguagesDataset('data/train1.txt')
	data_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
	test_dataset = LanguagesDataset('data/test1.txt')
	nll = nn.NLLLoss()
	
	input_gates_targets = []
	for t in range(3):
		input_gates_target = torch.zeros(3 * EMBEDDING_SIZE)
		input_gates_target[(EMBEDDING_SIZE)*t : (EMBEDDING_SIZE*(t+1))] = torch.ones(EMBEDDING_SIZE)
		input_gates_targets.append(input_gates_target)
	forget_gates_targets = []
	for t in range(3):
		forget_gates_target = torch.zeros(3 * EMBEDDING_SIZE)
		forget_gates_target[:(EMBEDDING_SIZE*t)] = torch.ones(EMBEDDING_SIZE*t)
		forget_gates_targets.append(forget_gates_target)
	
	while (True):
		model = LookupTablePosMy(VOCAB_SIZE, EMBEDDING_SIZE, OUTPUT_SIZE, 1)
		
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
				prediction_loss = nll(logits, y_batch)
				gates_loss = 0
				for t in range(3):
					#print('input t = ' + str(t) + ': ' + str(F.binary_cross_entropy(model.input_gates[t], input_gates_targets[t].repeat(X1_batch.size(0), 1))))
					gates_loss += F.binary_cross_entropy(model.input_gates[t], input_gates_targets[t].repeat(X1_batch.size(0), 1))
				for t in range(3):
					#print('forget t = ' + str(t) + ': ' + str(F.binary_cross_entropy(model.forget_gates[t], forget_gates_targets[t].repeat(X1_batch.size(0), 1))))
					gates_loss += F.binary_cross_entropy(model.forget_gates[t], forget_gates_targets[t].repeat(X1_batch.size(0), 1))
				loss = prediction_loss + GAMMA * gates_loss
				print('Prediction loss = ' + str(prediction_loss.data[0].item()))
				print('Gates loss = ' + str(gates_loss.data[0].item()))
				print('Total loss = ' + str(loss.data[0].item()))
				print()
				loss.backward()
				"""
				for name, param in model.named_parameters():
					if (name == 'lstm.lstm_cell.wh.weight'):
						print('wh norm = ' + str(param.grad.norm().item()))
						if (param.grad.ne(param.grad).any()):
							raise Exception('Vanishing gradient')
					if (name == 'lstm.lstm_cell.wx.weight'):
						if (param.grad.ne(param.grad).any()):
							raise Exception('Vanishing gradient')
						print('wx norm = ' + str(param.grad.norm().item()))
				print()
				"""
				optimizer.step()
			
			train_accuracy = accuracy(model(X1_train, X2_train), y_train)
			test_accuracy = accuracy(model(X1_test, X2_test), y_test)
			train_epoch_accuracy.append(train_accuracy)
			test_epoch_accuracy.append(test_accuracy)
			print('Epoch: ' + str(epoch))
			print('Train accuracy: ' + str(train_accuracy))
			print('Test accuracy: ' + str(test_accuracy))
			print()
		#if (test_accuracy > 0.99):
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