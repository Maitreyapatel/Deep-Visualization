import numpy as np
import pandas as pd
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


def preprocess_image(cv2im, resize_im=True):

    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def convert_to_grayscale(cv2im):
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im




def get_positive_negative_saliency(gradient):

    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency



class GuidedBackprop():
    def __init__(self, model, target_class):
        self.model = model

        self.target_class = target_class
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):

        def relu_hook_function(module, grad_in, grad_out):

            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)

    def generate_gradients(self, prep_img):
        # Forward pass
        self.input_image = prep_img
        model_output = self.model(self.input_image)
        temp = model_output

        _, predicted = torch.max(temp.data, 1)

        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][self.target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr




class Visualize_Guided_Backpropagation():

    def __init__(self,model,target_class):
        self.model = model
        self.target_class = target_class
        self.GBP = GuidedBackprop(model, target_class)

    def Colored_Guided_Backpropagation(self,image):
        prep_img = preprocess_image(image)
        guided_grads = self.GBP.generate_gradients(prep_img)

        output = self.convert_image(guided_grads)

        return output

    def Colored_Guided_Backpropagation_Live(self):
        cv2.namedWindow("Input")
        cv2.namedWindow("Colored Guided Backpropagation")

        cap = cv2.VideoCapture(0)
        count = 0
        start_time = time.time()

        # pretrained_model = models.alexnet(pretrained=True)


        while cap.isOpened():
            rval, frame = cap.read()
            prep_img = preprocess_image(frame)

            output = self.Colored_Guided_Backpropagation(frame)

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






    def Guided_Backpropagation_Saliency(self,image):
        prep_img = preprocess_image(image)
        guided_grads = self.GBP.generate_gradients(prep_img)

        grayscale_guided_grads = convert_to_grayscale(guided_grads)

        output = self.convert_image(grayscale_guided_grads)

        return output


    def Guided_Backpropagation_Saliency_Live(self):
        cv2.namedWindow("Input")
        cv2.namedWindow("Guided Backpropagation Saliency")

        cap = cv2.VideoCapture(0)
        count = 0
        start_time = time.time()


        while cap.isOpened():
            rval, frame = cap.read()
            prep_img = preprocess_image(frame)

            output = self.Guided_Backpropagation_Saliency(frame)

            cv2.imshow("Input", frame)
            cv2.imshow("Guided Backpropagation Saliency", output)

            count += 1

            if count == 10:
                print("Frame per sec:{}".format(10 / (time.time() - start_time)))
                count = 0
                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def Guided_Backpropagation_Positive_Negative_Saliency(self,image):
        prep_img = preprocess_image(image)
        guided_grads = self.GBP.generate_gradients(prep_img)

        pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)

        pos_image, neg_image = self.convert_image(pos_sal), self.convert_image(neg_sal)

        return pos_image, neg_image



    def Guided_Backpropagation_Positive_Negative_Saliency_Live(self):
        cv2.namedWindow("Input")
        cv2.namedWindow("Positive Saliency")
        cv2.namedWindow("Negative Saliency")

        cap = cv2.VideoCapture(0)
        count = 0
        start_time = time.time()


        while cap.isOpened():
            rval, frame = cap.read()
            prep_img = preprocess_image(frame)

            pos_output, neg_output = self.Guided_Backpropagation_Positive_Negative_Saliency(frame)

            cv2.imshow("Input", frame)
            cv2.imshow("Positive Saliency", pos_output)
            cv2.imshow("Negative Saliency", neg_output)

            count += 1

            if count == 10:
                print("Frame per sec:{}".format(10 / (time.time() - start_time)))
                count = 0
                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def convert_image(self,gradient):
        gradient = gradient - gradient.min()
        gradient /= gradient.max()
        gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
        gradient = gradient[..., ::-1]

        return gradient


# pretrained_model = models.alexnet(pretrained=True)
#
# VGP = Visualize_Guided_Backpropagation(pretrained_model,target_class=0)
#
# VGP.Guided_Backpropagation_Positive_Negative_Saliency_Live()