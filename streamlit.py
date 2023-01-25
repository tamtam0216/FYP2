import sys


import streamlit as st
import matplotlib.pyplot as plt

import io

from PIL import Image

import timm
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np

import json
from functools import partial


class Mish_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        result = x * torch.tanh(F.softplus(x))
        ctx.save_for_backward(x)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        v = 1. + x.exp()
        h = v.log()
        grad_gh = 1. / h.cosh().pow_(2)

        # Note that grad_hv * grad_vx = sigmoid(x)
        # grad_hv = 1./v
        # grad_vx = i.exp()

        grad_hx = x.sigmoid()
        grad_gx = grad_gh * grad_hx
        grad_f = torch.tanh(F.softplus(x)) + x * grad_gx

        return grad_output * grad_f


class Mish(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(self, x):
        return Mish_func.apply(x)


#global pooling
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
    # apply the formula of GeM pooling
    def gem(self, x: torch.Tensor, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)
    # print this method's detail when call this method
    def __repr__(self):
        return self.__class__.__name__ + (f'p={self.p.data.tolist()[0]:.4f}, eps={self.eps}')


# 选一个最标准的答案，然后以他做标准
class ArcMarginProduct_subcenter(nn.Module):

    def __init__(self, in_features, out_features, k=3):  # k = number of sub-centers
        '''
        Function to initialize the ArcMarginProduct
        '''
        super().__init__()

        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        '''
        Function to initialize the weights to (-std_dev, +std_dev)
        '''
        # use the weight of the image to get the stdv
        stdv = 1. / math.sqrt(self.weight.size(1))
        # make a random distribution that fit the weight of image into a range of number (can be said normalize it)
        self.weight.data.uniform_(-stdv, stdv)  # Uniform sampling from -stddev to +stddev

    def forward(self, features):
        '''
        Apply forward propagation with linear function y = m*X + C
        Calculate and find maximum cosine distance
        '''
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))  # shape = (N, out_features)
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        # Find the maximum cosine distance
        # find angular distance (to find the maximum distance between two subcenter/ feature)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


    # change the number of neuron after the global pooling
class LeafModel(nn.Module):
    def __init__(self, model_name, pretrained=True, in_chans=3):
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=in_chans)
        # out_features is to get the number of neuron in last fully connected layer
        # in features = output of global pooling, then pass to the classifier
        if 'regnet' in model_name:
            self.out_features = self.model.head.fc.in_features
        elif 'vit' in model_name or 'swin' in model_name:
            self.out_features = self.model.head.in_features
        elif 'deit' in model_name:
            if 'base' in model_name:
                self.out_features = 768
            elif 'small' in model_name:
                self.out_features = 384
            elif 'tiny' in model_name:
                self.out_features = 192
        elif 'nfnet' in model_name:
            self.out_features = self.model.head.fc.in_features
        elif 'rexnet' in model_name:
            self.out_features = self.model.head.fc.in_features
        elif 'csp' in model_name:
            self.out_features = self.model.head.fc.in_features
        elif 'res' in model_name:  # works also for resnest
            self.out_features = self.model.fc.in_features
        elif 'efficientnet' in model_name:
            self.out_features = self.model.classifier.in_features

        elif 'densenet' in model_name:
            self.out_features = self.model.classifier.in_features
        elif 'senet' in model_name:
            self.out_features = self.model.fc.in_features
        elif 'inception' in model_name:
            self.out_features = self.model.last_linear.in_features
        else:
            self.out_features = self.model.classifier.in_features
        # to remove the head of timm (change neuron)
        self.model.reset_classifier(4)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.gem = GeM(p=3)
        self.embed_size = 512

        self.neck = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.out_features, self.embed_size),
            nn.BatchNorm1d(self.embed_size),
            Mish()
        )

        self.head = nn.Linear(512, 5)
        self.mish = Mish()
        self.metric_classify = ArcMarginProduct_subcenter(self.embed_size, 5, k=3)

    def forward(self, x):
        # x = x.unsqueeze(1)
        batch_size = x.shape[0]
        # to get the unpooled data (means that we remove the ori pooling)
        output = self.model.forward_features(x)
        # output = output.unsqueeze(-1).unsqueeze(-1)
        # output = self.triple_att(output)
        # output = self.gem(output).view(batch_size, -1)
        # output = output[:,:,0,0]

        # # call the GeM pooling, use view because we sure column but not sure the number of row
        output = self.gem(output).view(batch_size, -1)
        output = self.neck(output)
        # out_cos = self.metric_classify(output)
        out_cos = self.head(output)
        # out_cos = self.model(x)

        return out_cos

# Loading model from checkpoint
def load_model(model, checkpoint):
    state_dict = torch.load(checkpoint,map_location=torch.device('cpu'))
    if "model_state_dict" in state_dict.keys():
        state_dict = state_dict['model_state_dict']
        
    state_dict = {k[7:] if k.startswith('module.') else k : state_dict[k] for k in state_dict.keys()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

# Define transformations for the image
def get_transforms(input_size, resize_im=True):
    t = []
    if resize_im:
        size = int((1.0) * input_size) #int((256 / 224) * args.input_size) (deit crop ratio (256 / 224), deit III crop ratio 1.0)
        t.append(
            T.Resize(input_size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        # t.append(T.CenterCrop(input_size))

    t.append(T.ToTensor())
    #t.append(T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return T.Compose(t)


# Specify title
st.title('Bone Resorption Severity Classification')
st.markdown("Our system allows you to identify the severity of your bone resorption from the dental panoramic radiograph images.")
# Create file uploader
uploaded_data = st.file_uploader(
    'Upload your image.', 
    type=['jpg', 'png', 'jpeg'],
    accept_multiple_files=False
)

# Check file is present or not
if uploaded_data is not None:
    data = uploaded_data.getvalue()
    # Convert data to bytes
    data = io.BytesIO(data)
    # Convert bytes to image
    data = Image.open(data)
    # Display image
    st.image(data, caption=uploaded_data.name)


# Specify model checkpoint
MODEL_CHECKPOINT = 'C:\\Users\\user\\Desktop\\best_model_optimizer_fold_3.pth'

CLASSES = {
    0: 'normal',
    1: 'early',
    2: 'moderate',
    3: 'severe'

}
# Create model
#model = ReLUViT(
   # patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=False,
  #  norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1000
#)

model = LeafModel('tf_efficientnet_b5_ns', pretrained=False)

# Load model
model = load_model(model,  MODEL_CHECKPOINT)

# Specify GPU



transforms = get_transforms(456)

if uploaded_data is not None:

   data = transforms(data)
   data = data / 255.
   #st.image(T.ToPILImage()(data))
   data = data.float().unsqueeze(0)

   with torch.no_grad():
       output = model(data)

   output_idx = output.softmax(-1).argmax(-1).cpu().tolist()[0]
   #st.write(f'Prediction: {output_idx}')
   st.write(f'Prediction: {CLASSES[output_idx]}')

st.uploaded_data = None



