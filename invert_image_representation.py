import cv2
import torch
from torch.autograd import Variable
from torch.optim import SGD
from torchvision import models

import os,copy
from os.path import join
import numpy as np

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


class InvertedRepresentation():
    def __init__(self, model, output_path=""):
        self.model = model
        self.model.eval()
        self.output_path = output_path


    def alpha_norm(self, input_matrix, alpha):
        alpha_norm = ((input_matrix.view(-1))**alpha).sum()
        return alpha_norm

    def total_variation_norm(self, input_matrix, beta):
        to_check = input_matrix[:, :-1, :-1]  # Trimmed: right - bottom
        one_bottom = input_matrix[:, 1:, :-1]  # Trimmed: top - right
        one_right = input_matrix[:, :-1, 1:]  # Trimmed: top - right
        total_variation = (((to_check - one_bottom)**2 +
                            (to_check - one_right)**2)**(beta/2)).sum()
        return total_variation

    def euclidian_loss(self, org_matrix, target_matrix):
        distance_matrix = target_matrix - org_matrix
        euclidian_distance = self.alpha_norm(distance_matrix, 2)
        normalized_euclidian_distance = euclidian_distance / self.alpha_norm(org_matrix, 2)
        return normalized_euclidian_distance

    def get_output_from_specific_layer(self, x, layer_id):
        layer_output = None
        for index, layer in enumerate(self.model.features):
            x = layer(x)
            if str(index) == str(layer_id):
                layer_output = x[0]
                break
        return layer_output

    def generate_inverted_image_specific_layer(self, input_image, img_size, target_layer=3):

        input_image = preprocess_image(input_image)

        opt_img = Variable(1e-1 * torch.randn(1, 3, img_size, img_size), requires_grad=True)

        optimizer = SGD([opt_img], lr=1e4, momentum=0.9)

        input_image_layer_output = \
            self.get_output_from_specific_layer(input_image, target_layer)


        alpha_reg_alpha = 6

        alpha_reg_lambda = 1e-7

        tv_reg_beta = 2

        tv_reg_lambda = 1e-8

        for i in range(201):
            optimizer.zero_grad()

            output = self.get_output_from_specific_layer(opt_img, target_layer)

            euc_loss = 1e-1 * self.euclidian_loss(input_image_layer_output.detach(), output)

            reg_alpha = alpha_reg_lambda * self.alpha_norm(opt_img, alpha_reg_alpha)

            reg_total_variation = tv_reg_lambda * self.total_variation_norm(opt_img,
                                                                            tv_reg_beta)

            loss = euc_loss + reg_alpha + reg_total_variation

            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                print('Iteration:', str(i), 'Loss:', loss.data.numpy()[0])


            if i % 40 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 1/10

        x = recreate_image(opt_img)
        cv2.imwrite(join(self.output_path,'Inv_Image_Layer_') + str(target_layer) +
                    '_Iteration_' + str(i) + '.jpg', x)

        print("Image file saved at {} with name Inv_Image_Layer_{}_Iteration_{}.jpg".format(self.output_path,str(target_layer),x))




# pretrained_model = models.alexnet(pretrained=True)
# original_image = cv2.imread("../images/dog.jpg", 1)
# # Process image
#
# inverted_representation = InvertedRepresentation(pretrained_model)
# image_size = 224  # width & height
# target_layer = 2
# inverted_representation.generate_inverted_image_specific_layer(original_image,image_size,target_layer)
