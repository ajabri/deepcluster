import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys
import numpy as np
import cv2
import time

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, class_to_idx, extensions, stride, shingle, N=-1, frame_skip=1):
    '''
    Walks through

    root/class/xxx.jpg

    assume class to be index of a video

    if stride < 0, then will identify automatically
    '''

    images = []
    iimages = []
    dir = os.path.expanduser(dir)

    for c_idx, target in enumerate(sorted(class_to_idx.keys())):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            n_imgs =  sum([has_file_allowed_extension(fname, extensions) for fname in fnames])
            length = n_imgs if stride < 0 else stride

            for idx, fname in enumerate(sorted(fnames, key=lambda x: int(x.split('.')[0]))[::frame_skip]):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)

                    group_idx = c_idx if stride < 0 else idx // stride
                    item = (path, group_idx)

                    images.append(item)
            
                    if idx % length >= shingle - 1:
                        iimages.append(([im[0] for im in images[-shingle:]], images[-1][1]))
                        if N > 0 and len(iimages) >= N:
                            break
            
    # import pdb; pdb.set_trace()

    return iimages


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None, 
        stride=-1, shingle=1, samples=None, N=-1, frame_skip=1, flow=0):
        if samples is None:
            classes, class_to_idx = self._find_classes(root)
            samples = make_dataset(root, class_to_idx, extensions,
                stride=stride, shingle=shingle, N=N, frame_skip=frame_skip)

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.flow = flow

        # self.classes = classes
        # self.class_to_idx = class_to_idx

        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """

        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        out = self.samples[index]
        if len(out) == 2:
            (path, target), pos = out, np.array([0, 0, 0])
        else:
            path, target, pos = out
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        def _sample(path):
            if isinstance(path, str):
                sample = self.loader(path)
            else:
                sample = path
            
            return sample
        
        samples = [_sample(p) for p in path]

        if self.flow > 0:

            assert len(samples) > 1, "need more than one frame per sample for flow"
            gray_samples = [cv2.cvtColor(np.array(s), cv2.COLOR_BGR2GRAY) for s in samples]
            
            for i in range(len(samples)-1):
                ff = None

                # try to reload flow
                if isinstance(path[0], str):
                    # get flow path
                    flow_dir = os.path.dirname(path[i]).replace('frames', 'flow')
                    p1 = os.path.basename(path[i])
                    p2 = os.path.basename(path[i+1])
                    flow_p1p2 = os.path.join([flow_dir, p1, p2])

                    if not os.path.exists(flow_p1p1):
                        os.makedirs(flow_dir, exist_ok=True)

                        ff = Image.fromarray(np.uint8(
                            self.compute_flow(gray_samples[i],
                            gray_samples[i+1]) * 255.0))
                        np.save(flow_p1p2, ff)
                    else:
                        print('loaded', p1p2)
                        ff = np.load(flow_p1p2)

                if ff is None:
                    ff = Image.fromarray(np.uint8(
                            self.compute_flow(gray_samples[i],
                            gray_samples[i+1]) * 255.0))
                
                flows.append(ff)

            if self.flow == 1:
                samples = flows
            elif self.flow == 2:
                samples = [np.concatenate([s, f], axis=0) for (s,f) in zip(samples, flows)]

            # HACK, only flow?

        # vis.image(np.array(sample).transpose(2, 0,1))
        if self.transform is not None:
            samples = [self.transform(s) for s in samples]

        sample = np.stack(samples)
        
        return sample, target, pos
    
    def compute_flow(self, x, y):
        hsv = np.zeros((*x.shape, 3))
        hsv[...,1] = 255

        t0 = time.time()
        flow = cv2.calcOpticalFlowFarneback(x, y, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        t1 = time.time()

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        t2 = time.time()

        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        t3 = time.time()
        bgr = cv2.cvtColor(hsv.astype(np.float32),cv2.COLOR_HSV2BGR).clip(min=0)/255.0 #.transpose(2, 0, 1)
        t4 = time.time()

        # print(t2-t1, t1-t0)
        # print(t4-t3, t3-t2, t2-t1, t1-t0)
        # import visdom; vis = visdom.Visdom(port=8095, env='main2')
        # vis.image(x)
        # vis.image(y)
        # vis.image(bgr.transpose(2, 0, 1).clip(min=0))
        # vis.text('', opts=dict(width=10000, height=5))
        # time.sleep(1)

        # import pdb; pdb.set_trace()
        # return np.concatenate([mag[None], ang[None], bgr])
        return bgr

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples


    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, stride=-1, shingle=1, samples=None, N=-1,
                 frame_skip=1, flow=0):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform,
                                          stride=stride, shingle=shingle, samples=samples,
                                          N=N, frame_skip=frame_skip, flow=flow)
        self.imgs = self.samples