import os
import numpy as np
import math
import librosa
from sklearn.decomposition import NMF
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf


def get_NMF(kind,path,dir_,save_dir):
	save = []
	for i in range(len(dir_)):
		p = os.path.join(path,dir_[i])
		print p
		audio,Fs = sf.read(p)
		#print Fs
		#audio ,Fs = librosa.core.load(p)
		z = librosa.core.stft(audio,n_fft = 4096, hop_length = 2048)
		z = abs(z)
		'''
		t = np.linspace(0,len(z[0])-1,len(z[0]))
		f = np.linspace(0,)
		'''
		D = librosa.amplitude_to_db(z,ref = np.max)
		plt.figure()
		librosa.display.specshow(D,y_axis = 'linear',x_axis = 'time',sr = 44100,hop_length = 2048)
		plt.colorbar(format = '%+2.0f dB')
		plt.title('Linear-frequency power spectrogram')
		plt.show()
		#print z.shape
		#print len(z[0,:])
		#print Fs
		delt = 100
		num = len(z[0])/delt
		#print num
		flag = 0
		for j in range(num):
			tmp = z[:,(0 + delt*j):(delt+delt*j)]
			model = NMF(n_components = 25)
			W = model.fit_transform(tmp)
			W = np.transpose(W,(1,0))
			#print W.shape
			save.append(W)
		print "completing:",kind,"file:",dir_[i]
		#break
		#print save
		'''
		plt.figure()
		t = []
		for j in range(len(W[:,0])):
			t.append(j)
		for j in range(len(W[0])):
			plt.plot(t,W[:,j])
		plt.show()
		'''
	#name = dir_[i].split('.')
	#s = kind + ".npy"
	#s = os.path.join(save_dir,s)
	#print s
	#np.save(s,W)
	save = np.array(save)
	print save.shape
	n = kind + ".npy"
	save_path = os.path.join(save_dir,n)
	np.save(save_path,save)
	

	print "////////completing kind:",kind,"///////////"

	#return W,len(W[:,0]),8

def main():
	path = "../data/dataset/audios/solo/"
	result = "../data/NMF_data_solo/"
	dir_1 = os.listdir(path)#kind
	#result_dir_1 = os.listdir(result)
	for i in range(len(dir_1)):
		#if i == 6 or i == 7 or i == 8:
		di = os.path.join(path,dir_1[i])#one kind path
		#re = os.path.join(result)
		dir_2 = os.listdir(di)#.wav
		get_NMF(dir_1[i],di,dir_2,result)



if __name__=='__main__':
    main()