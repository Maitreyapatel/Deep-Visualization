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


class CNNLayerVisualization():
    def __init__(self, model, selected_layer, selected_filter,path):
        self.model = model
        self.model.eval()
        self.path = path
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0

        self.created_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            self.conv_output = grad_out[0, self.selected_filter]

        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):

        self.hook_layer()

        self.processed_image = preprocess_image(self.created_image)

        optimizer = SGD([self.processed_image], lr=200, weight_decay=1e-6)
        for i in range(1, 51):
            optimizer.zero_grad()

            x = self.processed_image
            for index, layer in enumerate(self.model):

                x = layer(x)

                if index == self.selected_layer:
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()[0]))

            loss.backward()
            optimizer.step()

            self.created_image = recreate_image(self.processed_image)

            if i % 5 == 0:
                cv2.imwrite(join(self.path,'layer_vis_l' + str(self.selected_layer) +
                            '_f' + str(self.selected_filter) + '_iter'+str(i)+'.jpg'),
                            self.created_image)



cnn_layer = 2
filter_pos = 0
# Fully connected layer is not needed
pretrained_model = models.vgg16(pretrained=True).features
layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos,path='../genereted')
layer_vis.visualise_layer_with_hooks()
