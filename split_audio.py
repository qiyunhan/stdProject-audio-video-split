import os
import numpy as np
import librosa
import wave
from scipy.io import wavfile
import soundfile as sf
import json
from sklearn.decomposition import non_negative_factorization
from collections import OrderedDict


def split_sound(path_read_file,label,path_save_file,name):
	base_num = 400
	for i in range(len(path_read_file)):
		#if i != 0 and i != 1 and i != 15 and i != 14:
		#	continue
		print path_read_file[i]
		print label[i][0],label[i][1]

		audio,sample_rate = sf.read(path_read_file[i])
		print audio.shape
		z1 = librosa.core.stft(audio,n_fft = 4096,hop_length = 2048)
		print len(z1[0])
		r = 100
		N = len(z1[0])/r
		#if N <= 10:
		#	r = 1000
		#	N = len(z1[0])/r
		print N
		d1 = []
		d2 = []
		for s in range(2049):
			d1.append([])
			d2.append([])
		d1 = np.array(d1)
		d2 = np.array(d2)
		for j in range(N+1):
			#print j
			if j == N:
				z = z1[:,r*j:len(z1[0])]
			else:
				z = z1[:,0+r*j:r*(j+1)]
			#print z.shape
			#print z.dtype
			Fs = sample_rate

			V = np.abs(z)

			W_1 = np.load("../data/base/"+str(label[i][0])+".npy")
			#print W_1.shape
			#W_1.dtype = float
			if base_num <= len(W_1):
				W_1 = W_1[0:base_num]
			W_1 = W_1.T

			W_2 = np.load("../data/base/"+str(label[i][1])+".npy")
			#print "W_2:",W_2.shape
			if base_num <= len(W_2):
				W_2 = W_2[0:base_num]

			W_2 = W_2.T
			#print W_2.shape

			W_ = np.concatenate((W_1,W_2),axis = 1)
			#print "W:",W_
			#print "W shape",W_.shape
			#print W_.shape#length*8
			#print "begin NMF"
			H,_,_ = non_negative_factorization(V.T,H = W_.T,n_components = (len(W_[0])),update_H = False)
			#print "completing NMF"
			H = H.T
			#print H
			#print H.shape
			H_1 = H[0:len(W_1[0]),:]
			H_2 = H[len(W_1[0]):len(W_[0]),:]
			V_1 = np.matmul(W_1,H_1)
			#print "V_1",V_1.shape
			V_2 = np.matmul(W_2,H_2)
			#print "V_2",V_2.shape
			#delt = np.ones((len(V),len(V[0])))*1e-50
			V_1_q = V_1/(V_1 + V_2 + 1e-50)*z
			V_2_q = V_2/(V_1 + V_2 + 1e-50)*z

			d1 = np.concatenate((d1,V_1_q),axis = 1)
			d2 = np.concatenate((d2,V_2_q),axis = 1)
			#print d1.shape
			#print d2.shape

		
		#print "begin istft"
		audio_pre_1 = librosa.core.istft(d1, hop_length = 2048)
		#print audio_pre_1.max()
		audio_pre_2 = librosa.core.istft(d2, hop_length = 2048)
		#print "complete istft"
		#print audio_pre_2.max()
		print audio_pre_1.shape
		if len(audio_pre_1) < len(audio):
			zero = np.zeros(len(audio)-len(audio_pre_1))
			audio_pre_1 = np.append(audio_pre_1,zero)
			audio_pre_2 = np.append(audio_pre_2,zero)
		print audio_pre_1.shape
		print audio_pre_2.shape
		sf.write(os.path.join(path_save_file,name[i]+"_seg1.wav"),audio_pre_1,Fs)
		sf.write(os.path.join(path_save_file,name[i]+"_seg2.wav"),audio_pre_2,Fs)
		print "completing:",path_read_file[i]

def main():
	#base = np.load("bases.npy")
	#print "load bases successfully..."
	with open(os.path.join("../data/testset25/result_json","gt.json"),"r") as f:
		gt=json.load(f,object_pairs_hook = OrderedDict)
	result = {}

	read_file_path = []
	name = []
	for keys in gt:
		#print(keys)
		#generate result.json

		file_prefix=keys.split('.')[0]
		#print(file_prefix)
		file_key = file_prefix + '.mp4'
		#print(file_key)
		result[file_key] = []
		tmp = {}
		tmp['audio'] = file_prefix + '_seg1.wav'
		tmp['position'] = 0
		result[file_key].append(tmp)
		tmp = {}
		tmp['audio'] = file_prefix + '_seg2.wav'
		tmp['position'] = 1
		result[file_key].append(tmp)

		#print(file_prefix)
		read_file_path.append("../data/testset25/gt_audio/"+file_prefix+".wav")
		name.append(file_prefix)

	with open("../data/testset25/result_json/result.json","w") as f:
		json.dump(result,f,indent = 4)
	print(read_file_path)
	save_file_path = "../data/testset25/result_audio/"
	#print read_file_path
	#print name
	#label_test = [[4,4],[4,6],[7,4],[2,4],[4,0],[6,2],[7,4],[6,2],[2,4],[7,5],[6,7],[6,1],[6,1],[4,3],[6,4],[4,5],[4,0],[4,1],[7,5],[1,5],[6,7],[4,0],[7,5],[1,5],[4,1]]
	label_test = np.load("testset25_left_and_right.npy")
	print label_test
	split_sound(read_file_path,label_test,save_file_path,name)

if __name__=='__main__':
    main()
