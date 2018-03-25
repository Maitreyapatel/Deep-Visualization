import numpy as np
import cv2 , os
import torch
from torch.autograd import Variable


def preprocess_image(cv2im, resize_im=True):


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







def get_positive_negative_saliency(gradient):

    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


class VanillaBackprop():
    def __init__(self, model, target_class):
        self.model = model
        self.target_class = target_class
        self.gradients = None

        self.model.eval()

        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]


        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self,image,file_name,output_path=""):
        prep_img = preprocess_image(image)
        model_output = self.model(prep_img)

        self.model.zero_grad()

        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][self.target_class] = 1

        model_output.backward(gradient=one_hot_output)

        gradients_as_arr = self.gradients.data.numpy()[0]

        output = self.convert_image(gradients_as_arr)

        self.save_iamge(output,file_name,output_path)

        return gradients_as_arr


    def convert_image(self,gradient):
        gradient = gradient - gradient.min()
        gradient /= gradient.max()
        gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
        gradient = gradient[..., ::-1]

        return gradient

    def save_iamge(self,gradient,file_name,output_path=""):
        path_to_file = os.path.join(output_path, file_name + '.jpg')
        cv2.imwrite(path_to_file, gradient)

    def convert_to_grayscale(self,cv2im):
        grayscale_im = np.sum(np.abs(cv2im), axis=0)
        im_max = np.percentile(grayscale_im, 99)
        im_min = np.min(grayscale_im)
        grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
        grayscale_im = np.expand_dims(grayscale_im, axis=0)
        return grayscale_im



# target_class = 1
#
# original_image = cv2.imread("../images/dog.jpg", 1)
#
# pretrained_model = models.alexnet(pretrained=True)
#
# VBP = VanillaBackprop(pretrained_model, target_class)
#
# vanilla_grads = VBP.generate_gradients(original_image,"colored_fake")
#
# grayscale_vanilla_grads = VBP.convert_to_grayscale(vanilla_grads)
#
# output = VBP.convert_image(grayscale_vanilla_grads)
# VBP.save_iamge(output,"gray_fake")
#
# print("It's done see it..!")