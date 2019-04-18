import os
import numpy as np

connection = "connection.txt"
f = open(connection,"w")
labels = []
flag = 0
data_path = "../data/NMF_data_solo/"
file = os.listdir(data_path)
x = np.load(os.path.join(data_path,file[0]))
print x.shape
print "processing:",file[0]
for i in range(len(x)):
	labels.append(flag)
print file[0],"---->",flag
f.write(file[0]+"->"+str(flag))
f.write('\n')
flag = flag + 1

for i in range(len(file)):
	if i == 0:
		continue
	print "processing:",file[i]
	tmp = np.load(os.path.join(data_path,file[i]))
	print tmp.shape
	if len(tmp) > 1300:
		tmp = tmp[0:1300]
	for j in range(len(tmp)):
		labels.append(flag)
	print file[i],"->",flag
	f.write(file[i]+"---->"+str(flag))
	f.write('\n')
	flag = flag + 1
	x = np.concatenate((x,tmp),axis = 0)
print x.shape
np.save(os.path.join(data_path,"NMF_data.npy"),x)
labels = np.array(labels)
print "label shape:",labels.shape
np.save(os.path.join(data_path,"labels.npy"),labels)
f.close()
'''
f = open('duet_path.txt','w')
#flag = 0
duet = []
duet_label = []
data_path_duet = "../data/dataset/audios/duet/"
midir = os.listdir(data_path_duet)
#x = np.load("../data/NMF_data_duet/celloacoustic_guitar/4.npy")
#f.write("../data/dataset/audios/duet/celloacoustic_guitar/4.wav")
#f.write('\n')
#for k in range(len(x)):
#	duet_label.append([2,7])

for i in range(len(midir)):
	
	flag = []
	if midir[i] == "acoustic_guitarviolin":
		flag.append(5)
		flag.append(7)
	if midir[i] == "celloacoustic_guitar":
		flag.append(2)
		flag.append(7)
	if midir[i] == "flutetrumpet":
		flag.append(0)
		flag.append(4)
	if midir[i] == "fluteviolin":
		flag.append(4)
		flag.append(5)
	if midir[i] == "saxophoneacoustic_guitar":
		flag.append(1)
		flag.append(7)
	if midir[i] == "xylophoneacoustic_guitar":
		flag.append(3)
		flag.append(7)
	if midir[i] == "xylophoneflute":
		flag.append(3)
		flag.append(4)
	
	file = os.listdir(os.path.join(data_path_duet,midir[i]))
	for j in range(len(file)):
		s = file[j].split('.')
		print os.path.join(data_path_duet,midir[i],file[j])
		f.write(os.path.join("../data/dataset/audios/duet/",midir[i],s[0]+".wav"))
		f.write('\n')
		print os.path.join("../data/dataset/audios/duet/",midir[i],s[0]+".wav")
		#tmp = np.load(os.path.join(data_path_duet,midir[i],file[j]))
		#x = np.concatenate((x,tmp),axis = 0)
		#print x.shape
		print flag
		duet_label.append(flag)

#duet = np.array(duet)
#np.save(os.path.join(data_path_duet,"duet_data.npy"),x)
duet_label = np.array(duet_label)
np.save(os.path.join("../data/NMF_data_duet/","duet_label.npy"),duet_label)
#print x.shape
print duet_label.shape
print duet_label[18]
'''