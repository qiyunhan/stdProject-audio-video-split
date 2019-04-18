import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import STFT
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import librosa
from numpy import linalg as la
#from sklearn.decomposition import NMF
from sklearn.decomposition import non_negative_factorization

base_length = 2049
M = 25 #NMF num of basis
K = 4
L = 8 #category
batch_size = 64
train_epoch = 1

learning_rate = 0.005


class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.fc_bn = nn.Sequential(
			nn.Conv2d(1,1024,kernel_size = (2049,1),stride = (1,1)),
			nn.BatchNorm2d(1024, eps=1e-05, momentum=None, affine=False))
		self.conv1 = nn.Sequential(
			nn.Conv2d(1024,K*L,kernel_size = 1),
			nn.BatchNorm2d(K*L))
		self.maxpool_1 = nn.MaxPool2d(kernel_size = (K,1),stride = (K,1))
		self.maxpool_2 = nn.MaxPool2d(kernel_size = (1,M),stride = (1,1))
		self.sfmax = nn.LogSoftmax()

	def forward(self,x):
		#print "bbb"
		x = x.transpose(1,2)#batch_size*2049*M
		out = torch.unsqueeze(x,1)#batch_size*1*2049*M
		out = self.fc_bn(out)#batch_size*1024*1*M
		out = F.relu(out)

		out_conv = self.conv1(out)#batch_size*(KL)*1*M
		out_conv = F.relu(out_conv)
		#print "out conv1",out_conv.size()
		out_conv = torch.squeeze(out_conv,2)#batch_size*KL*M
		out_conv = torch.unsqueeze(out_conv,1)#batch_size*1*KL*M

		
		out_pool_1 = self.maxpool_1(out_conv)#batch_size*1*L*M

		
		out_pool_2 = self.maxpool_2(out_pool_1)#batch_size*1*8*1
		#print "out pool 2:",out_pool_2.size()
		out_pool_2 = out_pool_2.reshape(batch_size,L)
		
		out_L = self.sfmax(out_pool_2)
		return out_L,out_pool_1

net = Net()
loss_func = nn.NLLLoss()
optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate)
base = [[],[],[],[],[],[],[],[]]
name = ["trumpet","saxophone","cello","xylophone","flute","violin","accordion","acoustic_guitar"]

def NMF_data(imdir):
	#print "begin NMF.."
	test,_,_ = STFT.get_NMF(imdir)
	test = torch.from_numpy(test)
	test = test.transpose(0,1)
	test = test.type(torch.FloatTensor)
	test = Variable(test,requires_grad = True)
	return test

def get_label(imdir):
	seed = np.random.randint(0,8)
	labels = Variable(torch.LongTensor([0]))
	return labels

def load_data(test_prb):
	data_load_path = "../data/NMF_data_solo/NMF_data.npy"
	labels_load_path = "../data/NMF_data_solo/labels.npy"
	data = np.load(data_load_path)
	#print len(data)
	#data = np.transpose(data,(0,2,1))
	labels = np.load(labels_load_path)
	#print labels.shape
	data_train, data_test,labels_train,labels_test = train_test_split(data,labels,test_size = test_prb, random_state = 60)
	#print data_train.shape
	return data_train,data_test,labels_train,labels_test


def solo_test(f,data_test,labels_test):
	#test accuracy
	flag = 0
	t = 0
	acc = 0.0
	while(flag + batch_size <= len(data_test)):
		t += 1
		test = data_test[flag:flag + batch_size]
		test = torch.from_numpy(test)
		test = test.type(torch.FloatTensor)
		test = Variable(test,requires_grad = True)#batch_size*M*base_length
		#print test.size()
		labels = labels_test[flag:flag + batch_size]
		#labels = Variable(torch.LongTensor(labels))
		#print labels
		flag = flag + batch_size
		outputs,base_cube = net(test)#batch_size*L,batch_size*1*L*M
		base_cube = torch.squeeze(base_cube,1)
		#print "base_cube:",base_cube.size()
		#base_cube = base_cube.reshape(batch_size,M,L,K)
		#base_cube = base_cube.transpose(1,2)#b*L*M*K
		predict = torch.argmax(outputs,dim = 1)
		test = test.double()
		test = test.data.numpy()
		#print predict
		for g in range(len(outputs)):
			if predict[g] == labels[g]:
				acc += 1
				if f == 1:
					#print base_cube[g]
					M_long = base_cube[g][predict[g]]
					M_long = M_long.data.numpy()
					#print M_long
					M_long_sort = np.sort(M_long)
					if M_long_sort[M-1] == 0:
						print base_cube[g]
					index = np.where(M_long == M_long_sort[M-1])
					
					base_t = np.array(test[g][index])#1*base_length
					
					if len(base_t) == 1:
						u = M_long_sort[M-1]
						u = np.array([[u]])
						base_t = np.concatenate((base_t,u),axis = 1)#1*(base_length+1)
						'''
						l = np.linspace(0,len(base_cube[g])-1,len(base_cube[g]))
						m = np.linspace(0,len(base_cube[g][0])-1,len(base_cube[g][0]))
						d = base_cube[g].data.numpy()
						plt.figure()
						plt.pcolormesh(m,l,d,vmin = 0,vmax = 2.5)
						plt.colorbar()
						plt.title('bases-kinds relation map')
						plt.show()
						exit()
						'''
						base[predict[g]].append(base_t)
	acc = acc/t/batch_size
	return acc


def train(train_loss,data,labels_):
	data_train,_,labels_train,_ = train_test_split(data,labels_,test_size = 0.0, random_state = 70)
	labels_train = labels_train.reshape(len(labels_train))
	num_batch = (len(data_train)/batch_size)
	flag = 0
	while(flag + batch_size <= len(data_train)):
		test = data_train[flag:flag + batch_size]
		test = torch.from_numpy(test)
		test = test.type(torch.FloatTensor)
		test = Variable(test,requires_grad = True)
		#print test.size()
		labels = labels_train[flag:flag + batch_size]
		labels = Variable(torch.LongTensor(labels))
		#print labels
		flag = flag + batch_size
		outputs,_ = net(test)#batch_size*L
		loss = loss_func(outputs,labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_loss.append(loss.data)
	return train_loss


def main():
	data ,data_test,labels_,labels_test = load_data(0.15)
	#duet_data,duet_label = load_duet_data()
	#duet_data_train,duet_data_test,duet_labels_train,duet_labels_test = train_test_split(data,labels_,test_size = 0.2, random_state = 40)
	#labels_ = labels.reshape(len(labels),1)
	#print data.shape
	#print labels.shape
	for k in range(train_epoch):
		f = 0
		train_loss = []
		#train_loss_mean = np.mean(np.array(train_loss))
		train_loss = train(train_loss,data,labels_)
		#train_loss = train(train_loss,duet_data_train,duet_labels_train)
		train_loss_mean = np.mean(np.array(train_loss))
		#solo test
		acc = solo_test(0,data_test,labels_test)
		#duet test
		print "train epoch:",k,"loss:",train_loss_mean,"accuracy solo:",acc
	#clean data,data_test,labels,labels_test
	acc = solo_test(1,data,labels_)
	print "final acc:",acc
	'''
	f = []
	min_len = 100000
	for i in range(8):
		if len(base[i]) < min_len:
			print len(base[i])
			min_len = len(base[i])
	print "min base length:",min_len
	for i in range(8):
		print len(base[i])
		t = np.array(base[i])
		print t.shape
		t = np.squeeze(t)
		#print t.shape
		t = t[0:min_len]
		print t.shape
		f.append(t)
	f = np.array(f)
	print f.shape
	np.save("bases.npy",f)
	'''
	for i in range(8):
		t = np.array(base[i])
		t = np.squeeze(t)
		t = t[np.argsort(-t[:,len(t)])]#sort
		print t[:,len(t)]
		t = t[:,0:len(t[0])-1]
		print i,t.shape
		np.save("../data/base/"+str(i)+".npy",t)

if __name__=='__main__':
    main()

















