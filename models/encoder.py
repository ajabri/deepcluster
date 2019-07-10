
import math
import numpy as np
import time
import sys


import torch
from torch import nn
import torchvision
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import scipy
import scipy.ndimage

from . import snail
from . import resnet

class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size=7, sigma=3):
        super(GaussianSmoothing, self).__init__()
        self.size = kernel_size
        self.channels = channels
        self.sigma = sigma
 
        conv = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=0, bias=None, groups=channels)
        self.weight = conv.weight
        self.weights_init()

    def forward(self, x):
        return F.conv2d(F.pad(x, (2, 2, 2, 2), mode='reflect'),  weight=self.weight, groups=self.channels)

    def weights_init(self):
        n= np.zeros((self.size, self.size))
        ss = self.size//2
        n[ss,ss] = 1

        k = scipy.ndimage.gaussian_filter(n,sigma=self.sigma)
        self.weight.data.copy_(torch.from_numpy(k))

def add_sobel(net, sobel=True):
    if sobel:
        grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        grayscale.weight.data.fill_(1.0 / 3.0)
        grayscale.bias.data.zero_()
        sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        sobel_filter.weight.data[0, 0].copy_(
            torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        )
        sobel_filter.weight.data[1, 0].copy_(
            torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        )
        sobel_filter.bias.data.zero_()
        net.sobel = nn.Sequential(grayscale, sobel_filter)
        for p in net.sobel.parameters():
            p.requires_grad = False
    else:
        net.sobel = None
        
class Encoder(nn.Module):
    def __init__(self, sobel=False, blur=False, n_out=1, pretrained=False, 
            frame_size=128, traj_enc='bow', traj_length='1', use_affinity=True):

        super(Encoder, self).__init__()

        self.traj_enc = traj_enc
        self.traj_len = traj_length
        self.use_affinity = use_affinity

        self.frame_size = frame_size
        self.n_inp_chan = 2 if sobel else 3
        
        resnet_init = resnet.resnet10 if frame_size <= 128 else resnet.resnet18
        self.resnet = resnet_init(in_chan=self.n_inp_chan, pretrained=pretrained)

        self.features = [
            self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool,
            self.resnet.layer1, self.resnet.layer2, self.resnet.layer3
        ]

        # if sum(self.resnet.layers) >= 8:
        #     self.features += [self.resnet.layer4]

        if not (self.use_affinity and self.traj_len > 1):
            self.features += [self.resnet.avgpool]

        self.features = nn.Sequential(*(self.features))

        dummy_out = self.features(torch.zeros(1, self.n_inp_chan, self.frame_size, self.frame_size))
        self.map_shape = dummy_out[0].shape     
        self.feat_dim = self.map_shape[0] * self.map_shape[1] * self.map_shape[2]
        print(self.map_shape)

        if self.use_affinity and self.traj_len > 1:
            self.feat_dim = self.map_shape[0]
            self.affinity = Affinity(
                spatial_dim=7 if self.map_shape[2] < 10 else (14 if self.map_shape[2] < 20 else 28),
                inp_nchan=self.feat_dim,
                corr_mlp=True, T=None)

        if self.traj_enc == 'bow':
            self.tc = None
            self.classifier = nn.Sequential(
                                # nn.Linear(fcdim, fcdim),
                                # nn.ReLU(inplace=True)
                                )
            self.top_layer = nn.Linear(self.feat_dim, n_out)  ## dummy

        else:
            self.classifier = nn.Sequential(
                                # nn.Linear(self.tc.channel_count, fcdim),
                                # nn.ReLU(inplace=True)
                                )

            self.tc = snail.TCBlock(self.feat_dim, self.traj_len - (1 if hasattr(self, 'affinity') else 0), 32)
            self.top_layer = nn.Linear(self.tc.channel_count, n_out)  ## dummy


        self.printed = False

        self.blur = None if not blur else GaussianSmoothing(3, 7, 5)
        add_sobel(self, sobel=sobel)

    def forward(self, x):
        # import visdom
        # vis = visdom.Visdom(env='main2', port=8095)
        _x = x
        B, T, C, H, W = x.shape

        x = x.view(B*T, *x.shape[-3:])

        if self.blur is not None:
            x = self.blur(x)
            print('blurred')

        if self.sobel is not None:
            x = self.sobel(x)
            print('sobelled')

        # import pdb; pdb.set_trace()
        x = self.features(x)

        if not self.printed:
            print(x.shape)
            self.printed = True

        # x = self.avgpool(x)
        x = x.view(B, T, *x.shape[1:])

        flow = False

        # APPLY AFFINITY HERE
        if self.use_affinity and self.traj_len > 1:
            corr_out, corr_proj, corr, flows = self.affinity(x, avg_pool=True, flow=flow)
            x = corr_out
            
            # if flow:
                # import visdom
                # vis = visdom.Visdom(port=8095)
                
                # for i in range(_x.shape[0]):
                #     import cv2
                #     import pdb; pdb.set_trace()
                #     vis.images((_x[i] - _x[i].min())/(_x[i] - _x[i].min()).max())
                #     vis.image(flows[i].transpose(2, 1, 0))
                #     vis.image(cv2.resize(flows[i],None,fx=9,fy=9, interpolation=cv2.INTER_NEAREST).transpose(2, 1, 0))
                #     vis.text('', opts=dict(width=10000, height=5))

        # x = x.view(B, T, -1)
        # x = torch.tanh(x[:, 1:] - x[:, :-1])

        if self.traj_enc == 'temp_conv':
            import pdb; pdb.set_trace()
            x = self.tc(x.transpose(1,2)).transpose(1,2)
            x = x[:, -1]
        else:
            x = x.mean(1)
    
        # import pdb; pdb.set_trace()
        # in_channels, seq_len, filters

        # if self.classifier is None:
        #     self.classifier = nn.Sequential(
        #             nn.Linear(x.shape[-1], 1024),
        #             # nn.Linear(256 * block.expansion, 1024),
        #             nn.ReLU(inplace=True))

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        self.feats = x.data.cpu()

        if self.top_layer:
            x = self.top_layer(x)

        return x


class Affinity(nn.Module):

    def __init__(self, spatial_dim=14, inp_nchan=512, corr_mlp=False, T=None):
        super(Affinity, self).__init__()

        self.spatial_dim = spatial_dim
        self.corr_nchan = self.spatial_dim * self.spatial_dim

        self.div_num = self.inp_chan = inp_nchan
        self.T = self.div_num**-.5 if T is None else T
        print('self.T:', self.T)

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)

        if corr_mlp:
            print(self.corr_nchan, self.spatial_dim)
            self.corr_mlp = nn.Sequential(*[
                nn.Conv2d(spatial_dim*spatial_dim, 128, kernel_size=1, bias=True),
                self.relu,
                nn.Conv2d(128, 64, kernel_size=1, bias=True)]
            )

            nn.init.kaiming_normal_(self.corr_mlp[0].weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.corr_mlp[2].weight, mode='fan_out', nonlinearity='relu')
            self.corr_nchan = 64


        self.feat_proj1 = nn.Conv2d(self.inp_chan, 256, kernel_size=1, bias=False)
        self.corr_conv1 = nn.Conv2d(self.corr_nchan, self.inp_chan, kernel_size=2, padding=0, bias=True)
        # self.corr_conv2 = nn.Conv2d(128, 64, kernel_size=2, padding=0, bias=True)


        # initialization
        nn.init.kaiming_normal_(self.feat_proj1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.corr_conv1.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.corr_conv2.weight, mode='fan_out', nonlinearity='relu')


    def compute_corr_softmax(self, feat1, feat2, finput_num, spatial_dim, out_d=3):
        
        feat2 = feat2.transpose(3, 4) # for the inlier counter
        feat2 = feat2.contiguous()
        feat2_vec = feat2.view(feat2.size(0), feat2.size(1), -1)
        feat2_vec = feat2_vec.transpose(1, 2)

        feat1_vec = feat1.view(feat1.size(0), feat1.size(1), -1)
        corrfeat = torch.matmul(feat2_vec, feat1_vec)

        # if self.use_l2norm is False:
        corrfeat = torch.div(corrfeat, self.div_num**-.5)

        if out_d == 3:
            corrfeat = corrfeat.view(corrfeat.size(0), finput_num, spatial_dim * spatial_dim, spatial_dim, spatial_dim)
        elif out_d == 4:
            corrfeat = corrfeat.view(corrfeat.size(0), finput_num, spatial_dim, spatial_dim, spatial_dim, spatial_dim)

        corrfeat = F.softmax(corrfeat, dim=2)

        # corrfeat  = corrfeat.view(corrfeat.size(0), finput_num * spatial_dim * spatial_dim, spatial_dim, spatial_dim)

        return corrfeat

    def vis_flow(self, u, v):
        import cv2
        flows = []
        u, v = u.data.cpu().numpy().astype(np.float32), v.data.cpu().numpy().astype(np.float32)
        hsv = np.zeros((u.shape[1], u.shape[2], 3))
        
        for i in range(u.shape[0]):
            mag, ang = cv2.cartToPolar(u[i], v[i])
            hsv[...,1] = 255
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2BGR)
            flows.append(bgr)

        return flows

    def compute_flow(self, corr):
        nnf = corr.argmax(dim=1)
        nnf = nnf.transpose(-1, -2)
 
        u = nnf % nnf.shape[-1]
        v = nnf / nnf.shape[-2] # nnf is an IntTensor so rounds automatically

        rr = torch.arange(u.shape[-1])[None].long().cuda()

        for i in range(u.shape[-1]):
            u[:, i] -= rr

        for i in range(v.shape[-1]):
            v[:, :, i] -= rr

        flows = self.vis_flow(u, v)

        return flows, u, v

    def forward(self, x, avg_pool=False, flow=False): #, ximg1, img2):
        '''
        Takes `x` of shape B x L x C x H x W
        Computes the affinity between adjacent frames for each video of length L in batch
            whereby similarity is measure in the C dimension.
            (x is assumed to be a batch of sequences of 2D feature maps)
        
        Affinity is reshaped to be 3D -> h*w x h x w
            and optionally projected in lower-dim (yielding a soft representation of flow)

        TODO 
            corr_conv as 4D conv!
        '''

        # x is B x L x C x H x W
        B, L, C, H, W = x.shape

        x = x.view(B * L, *x.shape[2:]) 

        x_proj = self.feat_proj1(x)             # B*L x C x H x W
        x_proj_relu = self.lrelu(x_proj)

        # x = x.transpose(1, 2)      # B x C x L x H x W

        x_proj_relu_norm = F.normalize(x_proj_relu, p=2, dim=1)    # consider softmax instead here

        spatial_dim = x_proj_relu_norm.size(3)

        xx = x_proj_relu_norm.view(B, L, *x_proj_relu_norm.shape[1:])   # B x L x C' x H x W

        x1 = xx[:, :-1].contiguous().view(B*(L-1), *xx.shape[2:])     # B*(L-1) x C' x H x W
        x2 = xx[:,  1:].contiguous().view(B*(L-1), *xx.shape[2:])

        x1, x2 = x1.unsqueeze(2), x2.unsqueeze(2)     # B*(L-1) x C' x 1 x H x W

        # B*(L-1) x 1 x H*W x H x W   (with softmax along H*W dim)
        corr = self.compute_corr_softmax(x1, x2, 1, spatial_dim).squeeze(1)

        flows = None
        if flow:
            flows, u, v = self.compute_flow(corr)

        corr_proj = self.corr_mlp(corr)

        corr_conved = self.corr_conv1(self.lrelu(corr_proj))
        # corr_conved = self.corr_conv2(self.lrelu(corr_conved))
        
        if avg_pool:
            corr_conved = torch.nn.functional.avg_pool2d(corr_conved, kernel_size=corr_conved.shape[-1])
            corr_conved = corr_conved.squeeze(-1).squeeze(-1)

        corr_out = corr_conved.view(B, L-1, *corr_conved.shape[1:]).contiguous()  # B x L-1 x D x H x W

        return corr_out, corr_proj, corr, flows