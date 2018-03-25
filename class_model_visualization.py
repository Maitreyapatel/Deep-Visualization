import os
import cv2
import numpy as np
import copy
from os.path import join

from torch.optim import SGD
from torchvision import models
from torch.autograd import Variable
import torch


def preprocess_image(cv2im, resize_im=True):

    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)

    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]

    im_as_ten = torch.from_numpy(im_as_arr).float()

    im_as_ten.unsqueeze_(0)

    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def recreate_image(im_as_var):
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)

    recreated_im = recreated_im[..., ::-1]
    return recreated_im

class ClassSpecificImageGeneration():
    def __init__(self, model, target_class,path):
        self.mean = [-0.485, -0.456, -0.406]
        self.std = [1/0.229, 1/0.224, 1/0.225]
        self.model = model
        self.path = path
        self.model.eval()
        self.target_class = target_class

        self.created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def generate(self,initial_learning_rate,itr):
        # initial_learning_rate = 6
        for i in range(1, itr):

            self.processed_image = preprocess_image(self.created_image)

            optimizer = SGD([self.processed_image], lr=initial_learning_rate)

            output = self.model(self.processed_image)

            class_loss = -output[0, self.target_class]
            print('Iteration:', str(i), 'Loss', "{0:.2f}".format(class_loss.data.numpy()[0]))

            self.model.zero_grad()

            class_loss.backward()

            optimizer.step()

            self.created_image = recreate_image(self.processed_image)

            cv2.imwrite(join(self.path,'c_specific_iteration_'+str(i)+'.jpg'), self.created_image)
        return self.processed_image


pretrained_model = models.alexnet(pretrained=True)
csig = ClassSpecificImageGeneration(pretrained_model, target_class=52,path="../generated/")
csig.generate(6,150)