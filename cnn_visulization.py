from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import pickle
import nltk
import cv2 , os, time
import numpy as np

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


class cnn_layer_visualization():

    def __init__(self,model,imagenet_class_index_file_path="",output_path=""):
        self.model = model
        self.output_path = output_path
        self.modulelist = list(self.model.features.modules())
        self.labels = json.load(open(imagenet_class_index_file_path))

    def grayscale(self,image):
        image = torch.sum(image, dim=0)
        image = torch.div(image, image.shape[0])
        return image

    def normalize(self,image):
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

    def predict(self,image):
        _, index = self.model(image).data[0].max(0)
        return str(index[0]), self.labels[str(index)][1]

    def deprocess(self,image):
        return image * torch.Tensor([0.229,0.224,0.225]) + torch.Tensor([0.485,0.456,0.406])



    def layer_outputs(self,image):
        outputs = []
        names = []
        for layer in self.modulelist[1:]:
            image = layer(image)
            outputs.append(image)
            names.append(str(layer))

        output_im = []
        for i in outputs:
            i = i.squeeze(0)
            temp = self.grayscale(i)
            output_im.append(temp.data.cpu().numpy())

        fig = plt.figure()
        plt.rcParams["figure.figsize"] = (100, 100)

        for i in range(len(output_im)):
            a = fig.add_subplot(8, 4, i + 1)
            imgplot = plt.imshow(output_im[i])
            plt.axis('off')
            a.set_title(names[i].partition('(')[0], fontsize=15)

        plt.savefig(join(self.output_path,'layer_outputs.jpg'), bbox_inches='tight')
        print("File saved at {} with file name {}.".format(self.output_path,"layer_outputs.jpg"))

    def get_all_layers(self,image):
        image = self.normalize(image)

        self.layer_outputs(image)

    def one_layer_output(self,image,layer_num = 0):
        outputs = []
        names = []
        for en,layer in enumerate(self.modulelist[1:]):
            image = layer(image)
            outputs.append(image)
            names.append(str(layer))
            if en == layer_num:
                break

        output_im = []
        out = outputs[-1].squeeze(0)
        temp = self.grayscale(out)
        output_im.append(temp.data.cpu().numpy())

        fig = plt.figure()
        plt.rcParams["figure.figsize"] = (100, 100)

        for i in range(len(output_im)):
            a = fig.add_subplot(8, 4, i + 1)
            imgplot = plt.imshow(output_im[i])
            plt.axis('off')
            a.set_title(names[i].partition('(')[0], fontsize=15)

        plt.savefig(join(self.output_path,'one_layer_output.jpg'), bbox_inches='tight')
        print("File saved at {} with file name {}.".format(self.output_path,"one_layer_output.jpg"))

    def get_one_layer(self,image,layer_num=0):
        image = self.normalize(image)

        self.one_layer_output(image, layer_num)

    def get_one_layer_live(self,layer_num=0):
        cv2.namedWindow("Input")
        cv2.namedWindow("Layer number {}".format(layer_num))

        cap = cv2.VideoCapture(0)
        count = 0
        start_time = time.time()


        while cap.isOpened():
            rval, frame = cap.read()
            frame_PIL = Image.fromarray(np.uint8(frame*255))
            prep_img = self.normalize(frame_PIL)

            image = prep_img

            outputs = []
            names = []
            for en, layer in enumerate(self.modulelist[1:]):
                image = layer(image)
                outputs.append(image)
                names.append(str(layer))
                if en == layer_num:
                    break

            output_im = []
            out = outputs[-1].squeeze(0)
            temp = self.grayscale(out)
            output_im.append(temp.data.cpu().numpy())

            output = output_im[0]

            cv2.imshow("Input", frame)
            cv2.imshow("Colored Guided Backpropagation", output)

            count += 1

            if count == 10:
                print("Frame per sec:{}".format(10 / (time.time() - start_time)))
                count = 0
                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def get_all_layers_live(self):
        cv2.namedWindow("Input")
        for i in range(len(self.modulelist[1:])):
            cv2.namedWindow("Layer number {}".format(i))

        cap = cv2.VideoCapture(0)
        count = 0
        start_time = time.time()


        while cap.isOpened():
            rval, frame = cap.read()
            frame_PIL = Image.fromarray(np.uint8(frame*255))
            prep_img = self.normalize(frame_PIL)

            image = prep_img

            outputs = []
            names = []
            for layer in self.modulelist[1:]:
                image = layer(image)
                outputs.append(image)
                names.append(str(layer))

            output_im = []
            for i in outputs:
                i = i.squeeze(0)
                temp = self.grayscale(i)
                output_im.append(temp.data.cpu().numpy())



            cv2.imshow("Input", frame)
            for i in range(len(output_im)):
                cv2.imshow("Layer number {}".format(i), output_im[i])

            count += 1

            if count == 10:
                print("Frame per sec:{}".format(10 / (time.time() - start_time)))
                count = 0
                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def filter_outputs(self,image, layer_to_visualize):
        if layer_to_visualize < 0:
            layer_to_visualize += 31
        output = None
        name = None
        for count, layer in enumerate(self.modulelist[1:]):
            image = layer(image)
            if count == layer_to_visualize:
                output = image
                name = str(layer)

        filters = []
        output = output.data.squeeze()
        for i in range(output.shape[0]):
            filters.append(output[i, :, :])

        fig = plt.figure()
        plt.rcParams["figure.figsize"] = (10, 10)

        for i in range(int(np.sqrt(len(filters))) * int(np.sqrt(len(filters)))):
            fig.add_subplot(np.sqrt(len(filters)), np.sqrt(len(filters)), i + 1)
            imgplot = plt.imshow(filters[i])
            plt.axis('off')

        plt.savefig(join(self.output_path, 'filter_outputs.jpg'), bbox_inches='tight')
        print("File saved at {} with file name {}.".format(self.output_path, "filter_outputs.jpg"))

    def get_all_filters_of_one_layer(self,image,layer_num):
        image = self.normalize(image)

        self.filter_outputs(image, layer_num)

    def get_all_filters_live(self,layer_to_visualize):
        cv2.namedWindow("Input")
        for i in range(len(self.modulelist[1:])):
            cv2.namedWindow("filter number {}".format(i))

        cap = cv2.VideoCapture(0)
        count = 0
        start_time = time.time()


        while cap.isOpened():
            rval, frame = cap.read()
            frame_PIL = Image.fromarray(np.uint8(frame*255))
            prep_img = self.normalize(frame_PIL)

            image = prep_img

            if layer_to_visualize < 0:
                layer_to_visualize += 31
            output = None
            name = None
            for count, layer in enumerate(self.modulelist[1:]):
                image = layer(image)
                if count == layer_to_visualize:
                    output = image
                    name = str(layer)

            filters = []
            output = output.data.squeeze().numpy()
            for i in range(output.shape[0]):
                filters.append(output[i, :, :])



            cv2.imshow("Input", frame)
            for i in range(len(filters)):
                cv2.imshow("filter number {}".format(i), filters[i])

            count += 1

            if count == 10:
                print("Frame per sec:{}".format(10 / (time.time() - start_time)))
                count = 0
                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def get_one_filter_live(self,layer_to_visualize,filter_num):
        cv2.namedWindow("Input")
        cv2.namedWindow("filter number {}".format(filter_num))

        cap = cv2.VideoCapture(0)
        count = 0
        start_time = time.time()


        while cap.isOpened():
            rval, frame = cap.read()
            frame_PIL = Image.fromarray(np.uint8(frame*255))
            prep_img = self.normalize(frame_PIL)

            image = prep_img

            if layer_to_visualize < 0:
                layer_to_visualize += 31
            output = None
            name = None
            for count, layer in enumerate(self.modulelist[1:]):
                image = layer(image)
                if count == layer_to_visualize:
                    output = image
                    name = str(layer)

            filters = []
            output = output.data.squeeze().numpy()

            filters.append(output[filter_num, :, :])



            cv2.imshow("Input", frame)
            cv2.imshow("filter number {}".format(filter_num), filters[0])

            count += 1

            if count == 10:
                print("Frame per sec:{}".format(10 / (time.time() - start_time)))
                count = 0
                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


def load_image( path):
    image = Image.open(path)
    image = image.convert('RGB')
    return image



# kitten_1 = load_image("../images/index.png")
#
# model = models.vgg16(pretrained=True)
#
# CLV = cnn_layer_visualization(model,output_path="",imagenet_class_index_file_path="../labels/imagenet_class_index.json")
# CLV.get_one_filter_live(0,0)