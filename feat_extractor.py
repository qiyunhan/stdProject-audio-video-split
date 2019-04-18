import os
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
from collections import OrderedDict
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

global net
global normalize
global preprocess
global features_blobs
global classes
global weight_softmax
labels_path='labels.json'
idxs=[401,402,486,513,558,642,776,889]
names=['accordion','acoustic_guitar','cello','trumpet','flute','xylophone','saxophone','violin']
number=[6, 7, 2, 0, 4, 3, 1, 5]

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def hook_feature(module, input, output):
    global features_blobs
    features_blobs=output.data.cpu().numpy()

def load_model():
    global net
    global normalize
    global preprocess
    global features_blobs
    global classes
    global weight_softmax
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'
    net.eval()
    net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
       transforms.Resize((224,224)),
       transforms.ToTensor(),
       normalize
    ])
    classes = {int(key):value for (key, value)
              in json.load(open(labels_path,'r')).items()}
    if torch.cuda.is_available():
        net=net.cuda()

def get_CAM(imdir, imname):
    probs = [[], []]
    img = cv2.imread(os.path.join(imdir,imname))
    height, width, _ = img.shape
    mid = int(width / 2)
    img_pil = Image.open(os.path.join(imdir,imname))
    coordinate_1 = (0, 0, mid, height)
    coordinate_2 = (mid, 0, width, height)
    img_pil_1 = img_pil.crop(coordinate_1)
    img_pil_2 = img_pil.crop(coordinate_2)
    img_1 = [img_pil_1, img_pil_2]
    for i in range(2):
        img_tensor = preprocess(img_1[i])
        img_variable = Variable(img_tensor.unsqueeze(0))
        if torch.cuda.is_available():
            img_variable=img_variable.cuda()
        logit = net(img_variable)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        if torch.cuda.is_available():
            h_x=h_x.cpu()
        probs1 = h_x.numpy()
        for j in range(0, 8):
            #print('{:.3f} -> {}'.format(probs1[idxs[j]], names[j]))
            probs[i].append(probs1[idxs[j]])
    return probs[0], probs[1]

def get_CAM_right(imdir, imname):
    img = cv2.imread(os.path.join(imdir,imname))
    height, width, _ = img.shape
    quarter = int(width / 4)
    img_pil = Image.open(os.path.join(imdir,imname))
    coordinate = (width - quarter, 0, width, height)
    img_pil_1 = img_pil.crop(coordinate)
    probs = []
    img_tensor = preprocess(img_pil_1)
    img_variable = Variable(img_tensor.unsqueeze(0))
    if torch.cuda.is_available():
        img_variable=img_variable.cuda()
    logit = net(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    if torch.cuda.is_available():
        h_x=h_x.cpu()
    probs1 = h_x.numpy()
    for i in range(0, 8):
        #print('{:.3f} -> {}'.format(probs1[idxs[i]], names[i]))
        probs.append(probs1[idxs[i]])
    return probs

def get_CAM_left(imdir, imname):
    img = cv2.imread(os.path.join(imdir,imname))
    height, width, _ = img.shape
    quarter = int(width / 4)
    img_pil = Image.open(os.path.join(imdir,imname))
    coordinate = (0, 0, quarter, height)
    img_pil_1 = img_pil.crop(coordinate)
    probs = []
    img_tensor = preprocess(img_pil_1)
    img_variable = Variable(img_tensor.unsqueeze(0))
    if torch.cuda.is_available():
        img_variable=img_variable.cuda()
    logit = net(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    if torch.cuda.is_available():
        h_x=h_x.cpu()
    probs1 = h_x.numpy()
    for i in range(0, 8):
        #print('{:.3f} -> {}'.format(probs1[idxs[i]], names[i]))
        probs.append(probs1[idxs[i]])
    return probs

def main():
    with open(os.path.join("testset25/result_json", "gt.json"), "r") as f:
        gt = json.load(f, object_pairs_hook = OrderedDict)                      
    file='testset25/testimage'
    #orderfile = 'testset7/gt_audio'                                           
    load_model()
    #gt=os.listdir(orderfile)
    #correct = 0
    index = []
    for fi in gt:
        file_prefix = fi.split('.')[0]
        imdir = file + '/' + file_prefix
        print(imdir)
        imlist=os.listdir(imdir)
        probs_left=np.zeros([8])
        probs_right=np.zeros([8])
        count = 0
        for im in imlist:
            count = count + 1
        print(count)
        if count <= 200:
            for im in imlist:
                print(im)
                probs1, probs2=get_CAM(imdir, im)
                probs_left=probs_left+np.array(probs1)
                probs_right=probs_right+np.array(probs2)
                print(probs_left)
                print(probs_right)
            index_left = np.argmax(probs_left)
            index_right = np.argmax(probs_right)
            if index_left == index_right and probs_left[index_left] > probs_right[index_right]:
                probs_right = np.zeros([8])
                for im in imlist:
                    print(im)
                    probs2 = get_CAM_right(imdir, im)
                    probs_right = probs_right + np.array(probs2)
                    print(probs_left)
                    print(probs_right)
                index_right = np.argmax(probs_right)
            if index_left == index_right:
                probs_right[index_right] = 0
                index_right = np.argmax(probs_right)
            if index_left == index_right and probs_left[index_left] < probs_right[index_right]:
                probs_left = np.zeros([8])
                for im in imlist:
                    print(im)
                    probs1 = get_CAM_left(imdir, im)
                    probs_left = probs_left + np.array(probs1)
                    print(probs_left)
                    print(probs_right)
                index_left = np.argmax(probs_left)
            if index_left == index_right:
                probs_left[index_left] = 0
                index_left = np.argmax(probs_left)
        if count > 200:
            distance = int(count / 200)
            a = 0
            for i in range(0, 200):
                print(a)
                im = imlist[a]
                print(im)
                probs1, probs2=get_CAM(imdir, im)
                probs_left=probs_left+np.array(probs1)
                probs_right=probs_right+np.array(probs2)
                print(probs_left)
                print(probs_right)
                a = a + distance
            index_left = np.argmax(probs_left)
            index_right = np.argmax(probs_right)
            print(index_left)
            print(index_right)
            print(probs_left[index_left])
            print(probs_right[index_right])
            if index_left == index_right and probs_left[index_left] > probs_right[index_right]:
                a = 0
                probs_right = np.zeros([8])
                for i in range(0, 200):
                    print(a)
                    im = imlist[a]
                    print(im)
                    probs2 = get_CAM_right(imdir, im)
                    probs_right = probs_right + np.array(probs2)
                    print(probs_left)
                    print(probs_right)
                    a = a + distance
                index_right = np.argmax(probs_right)
                if index_left == index_right:
                    probs_right[index_right] = 0
                    index_right = np.argmax(probs_right)
            if index_left == index_right and probs_left[index_left] < probs_right[index_right]:
                a = 0
                probs_left = np.zeros([8])
                for i in range(0, 200):
                    print(a)
                    im = imlist[a]
                    print(im)
                    probs1 = get_CAM_left(imdir, im)
                    probs_left = probs_left + np.array(probs1)
                    print(probs_left)
                    print(probs_right)
                    a = a + distance
                index_left = np.argmax(probs_left)
                if index_left == index_right:
                    probs_left[index_left] = 0
                    index_left = np.argmax(probs_left)
        finame = fi.split("_")   
        if names[index_left][0:4] == finame[0][0:4] and names[index_right][0:4] == finame[2][0:4]:
            correct = correct + 1
        elif names[index_left][0:4] == finame[0][0:4] and names[index_right][0:4] == finame[3][0:4]:
            correct = correct + 1
        index.append(number[index_left])
        index.append(number[index_right])
        print('{} {}'.format(number[index_left], number[index_right]))
    index_array = np.array(index)
    index_array.shape = 25, 2                                                   
    np.save('testset25_left_and_right.npy', index_array)
    #print('{} {:.3f}'.format(correct, float(correct) / 25))                     

if __name__=='__main__':
    main()
            

