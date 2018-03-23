from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import pickle
import nltk
import cv2 , os, time

import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn as nn
import torch.autograd as autograd
from torch.nn import ReLU

import torchvision
from torchvision import transforms, datasets, models

import scipy.misc
import matplotlib.pyplot as plt
import json

def grayscale(image):
    image = torch.sum(image, dim=0)
    image = torch.div(image, image.shape[0])
    return image

def normalize(image):
    normalize = transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])

    image = Variable(preprocess(image).unsqueeze(0))

    return image

def predict(image):
    _, index = model(image).data[0].max(0)
    return str(index[0]), labels[str(index)][1]

def deprocess(image):
    return image * torch.Tensor([0.229,0.224,0.225]) + torch.Tensor([0.485,0.456,0.406])

def load_image(path):
    image = Image.open(path)
    image = image.convert('RGB')
    plt.imshow(image)
    plt.title("Image loaded successfully")
    return image

kitten_1 = load_image("../images/index.png")

model = models.vgg16(pretrained=True)

labels = json.load(open("../labels/imagenet_class_index.json"))

kitten_2 = normalize(kitten_1)

modulelist = list(model.features.modules())


def layer_outputs(image):
    outputs = []
    names = []
    for layer in modulelist[1:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))

    output_im = []
    for i in outputs:
        i = i.squeeze(0)
        temp = grayscale(i)
        output_im.append(temp.data.cpu().numpy())

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (100, 100)

    for i in range(len(output_im)):
        a = fig.add_subplot(8, 4, i + 1)
        imgplot = plt.imshow(output_im[i])
        plt.axis('off')
        a.set_title(names[i].partition('(')[0], fontsize=15)

    plt.savefig('layer_outputs.jpg', bbox_inches='tight')


layer_outputs(kitten_2)