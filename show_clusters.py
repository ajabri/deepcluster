# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle
import time
import debug

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset

import clustering
import models
from util import AverageMeter, Logger, UnifLabelSampler


parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['alexnet', 'vgg16', 'resnet18'], default='alexnet',
                    help='CNN architecture (default: alexnet)')
parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                    default='Kmeans', help='clustering algorithm (default: Kmeans)')
parser.add_argument('--sobel', action='store_true', default=False, help='Sobel filtering')
parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                    help='number of cluster for k-means (default: 10000)')
parser.add_argument('--lr', default=0.05, type=float,
                    help='learning rate (default: 0.05)')
parser.add_argument('--wd', default=-5, type=float,
                    help='weight decay pow (default: -5)')
parser.add_argument('--reassign', type=float, default=1.,
                    help="""how many epochs of training between two consecutive
                    reassignments of clusters (default: 1)""")
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--batch', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: None)')
parser.add_argument('--checkpoints', type=int, default=25000,
                    help='how many iterations between two checkpoints (default: 25000)')
parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
parser.add_argument('--exp', type=str, default='', help='path to exp folder')
parser.add_argument('--verbose', action='store_true', help='chatty')


def main():
    global args
    args = parser.parse_args()


    import visdom 
    vis = visdom.Visdom(port=8095, env=args.resume.split('/')[-2])

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    
    model = models.__dict__[args.arch](sobel=args.sobel)
    
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            for key in list(checkpoint['state_dict'].keys()):
                if 'num_batches' in key:
                    del checkpoint['state_dict'][key]

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # creating checkpoint repo
    # exp_check = os.path.join(args.exp, 'checkpoints')
    # if not os.path.isdir(exp_check):
    #     os.makedirs(exp_check)

    # # creating cluster assignments log
    # cluster_log = Logger(os.path.join(args.exp, 'clusters'))

    # preprocessing of data
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # tra = [transforms.Resize(256),
    #        transforms.CenterCrop(224),
    #        transforms.ToTensor(),
    #        normalize]

    # load the data
    end = time.time()
    if 'vizdoom' in args.data:
        # import pdb; pdb.set_trace()
        # data_tensor = torch.from_numpy(np.load(args.data))
        # dataset = TensorDataset(data_tensor)

        if '/data/' in args.data:
        # preprocessing of data
            mean=[0.29501004, 0.34140844, 0.3667595 ]
            std=[0.16179572, 0.1323428 , 0.1213659 ]

        elif '/data3/' in args.data:
            std=[0.12303435, 0.13653513, 0.16653976]
            mean=[0.4091152 , 0.38996586, 0.35839223]
        else:
            assert False, 'which normalization?'

        normalize = transforms.Normalize(mean=mean,
                                        std=std)
        unnormalize = transforms.Normalize(mean=[(-mean[i] / std[i]) for i in range(3)],
                                std=[1.0 / std[i] for i in range(3)])

        tra = [
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
            normalize]
        
        if 'bottom' in args.resume:
            tra = [transforms.Resize([128, 128]),
                transforms.Lambda(
                    # lambda crops: torch.stack([ToTensor()(crop) for crop in crops])
                lambda x: transforms.functional.crop(x, 128/2, 0, 128/2, 128)
            )] + tra
        dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))
        # import pdb; pdb.set_trace()

    else:
        # preprocessing of data
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        tra = [transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]
        dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))

    if args.verbose: print('Load dataset: {0:.2f} s'.format(time.time() - end))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)


    # remove head
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    # get the features for the whole dataset
    features = compute_features(dataloader, model, len(dataset))

    # cluster the features
    clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

    centroids = faiss.vector_float_to_array(deepcluster.clus.centroids)
    centroids = centroids.reshape(args.nmb_cluster, 256)

    faiss.write_VectorTransform(deepcluster.mat, args.resume + '.pca')
    
    model.features = model.features.module
    torch.save({
        'state_dict': model.state_dict(), 'centroids': centroids,
        'pca_path': args.resume + '.pca',
        'mean': mean, 'std': std},
        args.resume + '.clus')

    import random
    # import pdb; pdb.set_trace()
    sorted_lists = sorted(deepcluster.images_lists, key=len)#[::-1] 
    sorted_lists = [s for s in sorted_lists if len(s) > 7]

    for c, l in enumerate(sorted_lists):
        ll = random.sample(l, min(10, len(l)))
        imgs = torch.stack([unnormalize(dataset[i][0])*255. for i in ll])
        vis.images(imgs, opts=dict(title=f"{c} of length {len(l)}"))
        import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()

        
def compute_features(dataloader, model, N):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_tensor.cuda())

        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1])).astype('float32')

        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux.astype('float32')
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux.astype('float32')

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features


if __name__ == '__main__':
    main()
