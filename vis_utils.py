import imageio
import torchvision.transforms as transforms
import numpy as np

def make_gif_from_paths(paths, outname, duration=0.2):
    images = []
    for path in paths:
        image = imageio.imread(path)
        images.append(image)

    imageio.mimsave(outname, images, 'GIF', duration=duration)

def make_gif_from_tensor(tensor, outname, duration=0.2):
    images = []
    for tt in tensor:
        if tt.shape[0] == 3:
            tt = tt.transpose(1, 2, 0)
        images.append(tt)

    imageio.mimsave(outname, images, 'GIF', duration=duration)

def make_transform(data_path, sz=None):
    if sz is None:
        sz = 128
   
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
   
    if 'vizdoom' in data_path:
        # import pdb; pdb.set_trace()
        # data_tensor = torch.from_numpy(np.load(data_path))
        # dataset = TensorDataset(data_tensor)

        # if '/data/' in data_path:
        # # preprocessing of data
        #     mean = [0.29501004, 0.34140844, 0.3667595 ]
        #     std = [0.16179572, 0.1323428 , 0.1213659 ]

        # elif '/data3/' in data_path:
        std = [0.12303435, 0.13653513, 0.16653976]
        mean = [0.4091152 , 0.38996586, 0.35839223]
        # else:
        #     assert False, 'which normalization?'

        normalize = transforms.Normalize(mean=mean,
                                        std=std)

        tra = [
            transforms.Resize([sz, sz]),
            transforms.ToTensor(),
            normalize,
        ]

    else:
        # preprocessing of data
        normalize = transforms.Normalize(mean=mean,
                                        std=std)
        tra = [transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]

    m1, std1 = [(-mean[i] / std[i]) for i in range(3)], [1.0 / std[i] for i in range(3)]
    unnormalize = transforms.Normalize(mean=[(-mean[i] / std[i]) for i in range(3)],
                            std=[1.0 / std[i] for i in range(3)])

    return tra, (mean, std), (m1, std1), (normalize, unnormalize)

def unnormalize_batch(batch, m1, s1, inplace=False):
    if 'torch' in str(type(batch)):
        mean = torch.zeros(batch.shape)
        std = torch.zeros(batch.shape)
    else:
        mean = np.zeros(batch.shape)
        std = np.zeros(batch.shape)

    mean[:, 0, :, :] = m1[0]
    mean[:, 1, :, :] = m1[1]
    mean[:, 2, :, :] = m1[2]

    std[:, 0, :, :] = s1[0]
    std[:, 1, :, :] = s1[1]
    std[:, 2, :, :] = s1[2]
    
    # out = batch / 255.0
    if inplace:
        out = batch
        out *= std
        out += mean
        out *= 255.0
    else:
        out = batch
        out = out * std
        out += mean
        out *= 255.0

    return out

