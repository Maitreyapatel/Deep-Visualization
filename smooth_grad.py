import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms, utils

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
# %matplotlib inline
from PIL import Image
import json
from os.path import isfile, join, abspath, exists, isdir, expanduser
import os




class smooth_grad():
    def __init__(self,model,output_path=""):
        self.normalise = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalise
        ])
        self.model = model
        self.output_path = output_path

        if output_path!="" and not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def deprocess(self,image):
        return image * torch.Tensor([0.229, 0.224, 0.225]) + torch.Tensor([0.485, 0.456, 0.406])

    def normalize(self,image):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
        image = Variable(preprocess(image).unsqueeze(0))
        return image

    def get_smooth_grad_output(self,input, label, x=10, percent_noise=10):

        tensor_input = torch.from_numpy(np.array(input)).type(torch.FloatTensor)

        final_grad = torch.zeros((1, 3, 224, 224))
        for i in range(x):
            print('Sample:', i + 1)
            temp_input = tensor_input

            noise = torch.from_numpy(
                np.random.normal(loc=0, scale=(percent_noise / 100) * (tensor_input.max() - tensor_input.min()),
                                 size=temp_input.shape)).type(torch.FloatTensor)
            temp_input = (temp_input + noise).cpu().numpy()
            temp_input = Image.fromarray(temp_input.astype(np.uint8))
            temp_input = Variable(self.preprocess(temp_input).unsqueeze(0), requires_grad=True)

            output = self.model.forward(temp_input)
            output[0][label].backward()
            final_grad += temp_input.grad.data

        grads = final_grad / x
        grads = grads.clamp(min=0)
        grads.squeeze_()
        grads.transpose_(0, 1)
        grads.transpose_(1, 2)
        grads = np.amax(grads.cpu().numpy(), axis=2)

        true_image = self.normalize(input)
        true_image = true_image.squeeze()
        true_image = true_image.transpose(0, 1)
        true_image = true_image.transpose(1, 2)
        true_image = self.deprocess(true_image.data)

        fig = plt.figure()
        plt.rcParams["figure.figsize"] = (20, 20)

        a = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(true_image)
        plt.title('Original Image')
        plt.axis('off')

        a = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(grads)
        plt.axis('off')
        plt.title('SmoothGrad, Noise: ' + str(percent_noise) + '%, ' + 'Samples: ' + str(x))
        plt.show()
        plt.savefig(join(self.output_path, 'smooth_dog.jpg'), bbox_inches='tight')

        return grads

def load_image(path):
    image = Image.open(path)
    image = image.convert('RGB')
    return image


# vgg = models.vgg16(pretrained=True)
# SM = smooth_grad(vgg)
# dog_sg = load_image('../images/dog.jpg')
# dog_sg_sal = SM.get_smooth_grad_output(dog_sg, 207, 30, 10)