一、 demo文件组织方式
demo
——Net.py
——nmf_data_process.py
——split_audio.py
——STFT_NMF.py
——feat_extractor.py
——connection.txt
——testset25_left_and_right.npy
——testset7_left_and_right.npy
——labels.json
——base（文件夹）
    0.npy~7.npy
——NMF_data_solo（文件夹）
——testset25（文件夹）
    dataHelper
    gt_audio
    result_audio
    result_json
    testimage
    testvideo
    Evaluate.py

二、各文件解析
1. STFT_NMF.py
从dataset/audios/solo中读取文件，进行STFT后再分段NMF，得到训练数据集
分类别将乐器的训练数据保存到NMF_data_solo文件中

2. nmf_data_process.py和connection.txt
从NMF_data_solo文件中读取各个乐器的训练数据，保存为NMF_data_solo/NMF_data.npy
并标定标签，保存为：NMF_data_solo/labels.npy
将乐器和标签对应关系保存到connection.txt中

3. Net.py
从NMF_data_solo文件中读取NMF_data.npy和labels.npy
经过神经网络训练后，挑选出基，保存到bases文件夹下

4. split_audio.py
生成result.json并保存到testset25/result_json中，
分离25个音频，并保存到testset25/result_audio中

5. testset25_left_and_right.npy
是通过图像提取得到的testset25中，每一个视频左边和右边的label，是一个shape=（25,2）的矩阵，每一行第一个数是左边的label，第二个数是右边的label

6. testset7_left_and_right.npy
与testset25_left_and_right.npy作用类似，是通过testset7真实场景中图片得到的乐器标签

7. testset25/Evaluate.py
计算acc和sdr指标

8. feat_extractor.py
从testset25/testimage或testset7/testimage文件中读取文件，得到每组图片左边和右边的label
读取的顺序是从gt.json中读取，该文件中只给出了testset25的例子，助教如果要跑testset7的例子，只需要吧25改成7即可。

三、运行注意事项
1. NMF_data_solo文件夹中的内容太大，有7G，因此上传清华网盘以供下载使用，助教也可以运行STFT_NMF.py自己训练，大约运行1.5-2h。网盘链接为：https://cloud.tsinghua.edu.cn/f/50932e1b3ada44a58028/

2. testset25文件夹下，gt_audio、testimage、testvideo是空白的，需要助教把内容拷贝进去。

3. 运行demo流程
运行STFT_NMF.py、nmf_data_process.py（结果（训练）数据可以从网盘上下载，可略过此步）

运行Net.py（请确保Net.py与dataset在同一文件夹下，具体的读取数据路径在load_data函数中，助教可以自行修改。Net.py结果保存在bases文件夹中，该文件夹中已经有训练好的基，可以直接使用，否则，可以运行Net.py重新训练）

运行feat_extractor.py（该文件夹中已经有运行后的结果，为文件testset25_left_and_right.npy，如果需要验证该结果，可以运行该程序，可略过此步）

运行split_audio.py（分离testset25中的音频，保存在testset25/result_audio/中，分离的result_audio打包放在了网盘上，链接为：https://cloud.tsinghua.edu.cn/f/11a9c42dbf2545fe978c/，助教可以直接下载，或者运行split_audio.py重新分离一遍）

运行Evaluate.py（在/testset25/目录下运行）

四、python2库版本
scikit-learn == 0.20.1
SoundFile == 0.10.2
torch == 0.4.1
torchvisioin == 0.2.1
librosa == 0.6.2

注：如果重新运行Net.py，则提取的基可能是不同的，因此最终sdr也会不一样，不过不会有太大的误差。









