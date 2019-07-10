
import math
import numpy as np

import torch
from torch import nn
import torchvision
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx.cpu().numpy()].dot(
        feature_conv.reshape((nc, h * w)))
    # cam = weight_fc[class_idx.cpu().numpy()].dot(feature_conv.reshape(-1))

    # import pdb; pdb.set_trace()

    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]


class SaveFeatures():
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()

    def remove(self):
        self.hook.remove()


def CAM(model, img):
    model.eval()
    final_layer = model.layer4
    activated_features = SaveFeatures(final_layer)
    x = Variable((img.unsqueeze(0)).cuda(), requires_grad=True)

    y = model(x)
    yp = F.softmax(y).data.squeeze()
    activated_features.remove()

    weight_softmax_params = list(model.fc.parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    # weight_softmax_params
    class_idx = torch.topk(yp, 1)[1].int()
    overlay = getCAM(activated_features.features, weight_softmax, class_idx)

    from skimage import transform
    oo = transform.resize(overlay[0], img.shape[1:3])
    model.train()
    img = img.cpu().numpy()
    cmap = plt.cm.jet

    # import pdb; pdb.set_trace()
    return np.array(cmap(oo)).transpose([2, 0, 1
                                         ])[:3] * 0.5 + img / img.max() * 0.5
    # prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)


