from torch.utils.data import Dataset

class LanguagesDataset(Dataset):
	"""description"""
	
	def __init__(self, path):
		file = open(path)
		self.X = []
		self.y = []
		for line in file:
			s1, s2 = line.strip('\n').split('\t')
			self.X.append(s1)
			self.y.append(s2)
		file.close()
	
	def __len__(self):
		return len(self.X)
	
	def __getitem__(self, i):
		return self.X[i], self.y[i]