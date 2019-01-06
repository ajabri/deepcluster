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
import folder

import clustering
import models
import util
from util import AverageMeter, Logger, UnifLabelSampler


import sys
sys.path.append('../meta-vizdoom/')
sys.path.append('../meta-vizdoom/ppo/')
import env # VizDoom env

sys.path.append('./html/PyHTMLWriter/src')
from Element import Element
from TableRow import TableRow
from Table import Table
from TableWriter import TableWriter
import vis_utils
import imageio
import cv2

def main(args):
    # global args
    # args = parser.parse_args()

    resume = args.resume
    batch = args.batch
    data_path = args.data
    sobel = args.sobel
    clustering_type = args.clustering
    verbose = True #args.verbose
    nmb_cluster = args.nmb_cluster
    arch = 'resnet18'
    workers = args.workers

    seed = 31

    import visdom 
    vis = visdom.Visdom(port=8095, env=resume.split('/')[-2])

    # fix random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # CNN
    if verbose:
        print('Architecture: {}'.format(arch))
    
    model = models.__dict__[args.arch](sobel=args.sobel, traj_enc=args.traj_enc)
    
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True


    # optionally resume from a checkpoint
    if resume:
        util.resume_model(resume, model)

    end = time.time()

    # smoother = models.mini_models.GaussianSmoothing(3, 5, 1)
    tra, (mean, std), (m1, std1), (norm, unnorm) = vis_utils.make_transform(data_path)

    dataset = folder.ImageFolder(data_path, transform=transforms.Compose(tra),
            args=[args.ep_length, args.traj_length])

    if verbose: print('Load dataset: {0:.2f} s'.format(time.time() - end))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch,
                                             num_workers=workers,
                                             pin_memory=True)


    export(args, model, dataloader, dataset)

def export(args, model, dataloader, dataset):


    # remove head
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    # get the features for the whole dataset
    features, idxs, poses = compute_features(dataloader, model, len(dataset), args)

    # idxs = idxs[np.argsort(idxs)]
    # features = features[np.argsort(idxs)]
    
    if args.group > 1:
        args.group = args.ep_length - args.traj_length + 1

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster, group=args.group)

    # cluster the features
    clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

    centroids = deepcluster.clus.centroids
    
    # centroids = faiss.vector_float_to_array(deepcluster.clus.get_means_and_variances)
    # centroids = centroids.reshape(nmb_cluster, 256)

    # import pdb; pdb.set_trace()
    
    # self_index = faiss.IndexFlatL2(centroids.shape[1])   # build the index
    # self_index.add(centroids)         
    # self_dists = self_index.search(centroids, centroids.shape[0])

    _, (mean, std), _, _ = vis_utils.make_transform(args.data)

    model.features = model.features.module

    c_mean, c_cov, c_var = get_means_and_variances(deepcluster, features)

    if args.export > 0:
        faiss.write_VectorTransform(deepcluster.mat, resume + '.pca')
        torch.save({
            'state_dict': model.state_dict(), 'centroids': centroids,
            'pca_path': resume + '.pca',
            'mean': mean, 'std': std,
            'cluster_mean': c_mean, 'cluster_cov': c_cov
            },
            resume + '.clus')

    T = args.traj_length

    import random
    
    sorted_lists = sorted(deepcluster.images_dists, key=len)[::-1] 
    sorted_lists = [s for s in sorted_lists if len(s) > 7]


    meta = torch.load(args.data + '/meta.dict')
    pos = np.array(meta['pos'])

    pos_idx = np.arange(pos.shape[0]*pos.shape[1])
    pos_idx = pos_idx.reshape(pos.shape[0], pos.shape[1])[:, T-1:]
    pos_idx = pos_idx.reshape(pos_idx.shape[0] * pos_idx.shape[1])

    pos = pos.reshape(pos.shape[0]*pos.shape[1], pos.shape[2])

    sz = 30

    x0, y0 = pos.min(0)[:2]
    pos -= pos.min(0)[None]

    x1, y1 = pos.max(0)[:2]
    pos /= pos.max(0)[None]

    pos[:, :2] *= sz - 1e-3

    from scipy.ndimage.filters import gaussian_filter

    def get_obj_masks(objs):
        out = np.zeros((3, sz, sz))
        for o in objs[0]:
            # import pdb; pdb.set_trace()
            x, y = o
            x, y = int((x - x0)/x1 *sz), int((y-y0)/y1 * sz)
            out[:, x:x+1, y:y+1] = 1

        return out        

    def get_mask_from_coord(coord):
        import matplotlib.cm as cm

        x, y, a = coord
        x, y = int(x), int(y)
        out = np.zeros((3, sz, sz))
        out[:, x, y] = cm.jet(a)[:3]

        return out

    # import pdb; pdb.set_trace()

    # sorted_self_dists = np.argsort(self_dists[0][:, 1])[::-1]
    # sorted_self_dists = np.argsort(self_dists[0].sum(axis=-1))[::-1]
    
    smoother1 = models.mini_models.GaussianSmoothing(3, 5, 5)
    smoother2 = models.mini_models.GaussianSmoothing(3, 7, 5)
    smoother3 = models.mini_models.GaussianSmoothing(3, 7, 7)
    smoother4 = models.mini_models.GaussianSmoothing(3, 9, 7)


    if args.expname == '':
        exp_name = args.resume.split('/')[-2]
    else:
        exp_name = args.expname

    out_root = '/home/ajabri/clones/deepcluster-ajabri/html/%s' % exp_name

    # import pdb; pdb.set_trace()
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    table = Table()

    num_show = 8

    sorted_variance = np.argsort(c_var)[::-1]
    
    # import pdb; pdb.set_trace()
    
    for c, clus_idx in enumerate(sorted_variance):
    # for c, clus_idx in enumerate(sorted_self_dists):
        l = deepcluster.images_dists[clus_idx]
        
        # ll = random.sample(l, min(8, len(l)))
        ll = [ii[0] for ii in sorted(l, key=lambda x: x[1])[::-1]][:num_show//2]
        ll += [ii[0] for ii in random.sample(l, min(num_show//2, len(l)))]

        l = [ii[0] for ii in l]

        row = TableRow(rno=c)

        e = Element()
        e.addTxt('size: %s <br>variance: %s' % (len(deepcluster.images_dists[clus_idx]), c_var[clus_idx]))
        row.addElement(e)

        # import pdb; pdb.set_trace()

        ## MAP
        poo = []
        for t in range(T):
            poo += [get_mask_from_coord(ppp) for ppp in pos[pos_idx[l] - t]]
            # po1 = [get_mask_from_coord(ppp) for ppp in pos[pos_idx[l]]]
            # po2 = [get_mask_from_coord(ppp) for ppp in pos[pos_idx[l]]]

        posum = sum(poo)
        objs_mask = get_obj_masks(meta['objs'])

        posum /= posum.max()


        # import pdb; pdb.set_trace()
        posum += objs_mask
        posum = np.clip(posum, a_min=0, a_max=1)

        # posum *= 255.0
        # vis.image((posum*255.).astype(np.uint8), opts=dict(width=300, height=300))
        # vis.image(gaussian_filter((posum*255.), sigma=1).astype(np.uint8), opts=dict(width=300, height=300))


        # gifname = '%s/%s_%s.png' % (exp_name, c, 'map')
        gifname = '%s_%s.png' % (c, 'map')
        gifpath = '%s/%s' % (out_root, gifname)

        imageio.imwrite(gifpath,
            cv2.resize((posum*255.).astype(np.uint8).transpose(1, 2, 0), 
                (0,0), fx=5, fy=5, interpolation = cv2.INTER_AREA))

        e = Element()
        e.addImg(gifname, width=180)
        row.addElement(e)


        # ## EXEMPLARS
        for iii, i in enumerate(ll):
            # import pdb; pdb.set_trace()
            imgs = vis_utils.unnormalize_batch(dataset[i][0], mean, std)
            # vis.images(imgs, opts=dict(title=f"{c} of length {len(l)}"))
            # vis.images(smoother1(torch.Tensor(imgs)).numpy(), opts=dict(title=f"{c} of length {len(l)}"))
            # vis.images(smoother2(torch.Tensor(imgs)).numpy(), opts=dict(title=f"{c} of length {len(l)}"))
            # vis.images(smoother3(torch.Tensor(imgs)).numpy(),  opts=dict(title=f"{c} of length {len(l)}"))
            # vis.images(smoother4(torch.Tensor(imgs)).numpy(), opts=dict(title=f"{c} of length {len(l)}"))

            # gifname = '%s/%s_%s.gif' % (exp_name, c, i)
            gifname = '%s_%s.gif' % (c, i)
            gifpath = '%s/%s' % (out_root, gifname)

            vis_utils.make_gif_from_tensor(imgs.astype(np.uint8), gifpath)
            e = Element()
            if iii < num_show // 2:
                e.addTxt('rank %i<br>' % iii)
            else:
                e.addTxt('random<br>')

            e.addImg(gifname, width=128)
            row.addElement(e)


        ## EXEMPLARS
        # gl = np.array(l).reshape(-1, args.group)

        # for iii, i in enumerate(gl[:3]):
        #     # import pdb; pdb.set_trace()
        #     # imgs = vis_utils.unnormalize_batch(dataset[i][0], mean, std)
        #     imgs = np.stack([dataset[_idx][0][0] for _idx in i])
        #     imgs = vis_utils.unnormalize_batch(imgs, mean, std)
        #     # import pdb; pdb.set_trace()
    
        #     # imgs = vis_utils.unnormalize_batch(dataset[i][0], mean, std)


        #     # vis.images(imgs, opts=dict(title=f"{c} of length {len(l)}"))
        #     # vis.images(smoother1(torch.Tensor(imgs)).numpy(), opts=dict(title=f"{c} of length {len(l)}"))
        #     # vis.images(smoother2(torch.Tensor(imgs)).numpy(), opts=dict(title=f"{c} of length {len(l)}"))
        #     # vis.images(smoother3(torch.Tensor(imgs)).numpy(),  opts=dict(title=f"{c} of length {len(l)}"))
        #     # vis.images(smoother4(torch.Tensor(imgs)).numpy(), opts=dict(title=f"{c} of length {len(l)}"))

        #     # gifname = '%s/%s_%s.gif' % (exp_name, c, i)
        #     gifname = '%s_%s.gif' % (c, i[0])
        #     gifpath = '%s/%s' % (out_root, gifname)

        #     vis_utils.make_gif_from_tensor(imgs.astype(np.uint8), gifpath)
        #     e = Element()
        #     if iii < num_show // 2:
        #         e.addTxt('rank %i<br>' % iii)
        #     else:
        #         e.addTxt('random<br>')

        #     e.addImg(gifname, width=128)
        #     row.addElement(e)

        table.addRow(row)

        # vis.text('', opts=dict(width=10000, height=2))
        if (c+1) % 10 == 0:
            # import pdb; pdb.set_trace()
            tw = TableWriter(table, '/home/ajabri/clones/deepcluster-ajabri/html/%s' % exp_name, rowsPerPage=min(args.nmb_cluster,100))
            tw.write()

    # import pdb; pdb.set_trace()



def get_means_and_variances(dc, features):
    m = []
    cv = []
    v = []
    for i in range(args.nmb_cluster):
        feats = preprocess_features(dc.mat, features[dc.images_lists[i]])
        mm = feats.mean(0)
        xx = feats - mm

        cov = (xx.transpose() @ xx) / len(dc.images_lists[i])
        m.append(mm )
        cv.append(cov)

        v.append((((xx)**2).sum(-1) ** 0.5).mean())

    return m, cv, v

def preprocess_features(mat, npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """

    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata
    
def compute_features(dataloader, model, N, args):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, idx, pose) in enumerate(dataloader):
        # print(i)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_tensor.cuda())

        aux = model(input_var).data.cpu().numpy()
        idx = idx.data.cpu().numpy()
        pose = pose.data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1])).astype('float32')
            poses = np.zeros((N, pose.shape[1])).astype('float32')
            idxs = np.zeros((N)).astype(np.int)

        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux.astype('float32')
            poses[i * args.batch: (i + 1) * args.batch] = pose.astype(np.int)            
            idxs[i * args.batch: (i + 1) * args.batch] = idx.astype(np.int)
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux.astype('float32')
            poses[i * args.batch:] = pose.astype(np.int)            
            idxs[i * args.batch:] = idx.astype(np.int)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 50) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features, idxs, poses



if __name__ == '__main__':
    parser = util.get_argparse()
    args = parser.parse_args()
    main(args)
