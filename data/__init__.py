from .mytransforms import MyRandomCrop, GasussNoise, UpDownCrop, RandomBlue, RandomWhite, StopLine, RandomYellowShort, RandomYellowLong, RoadShoulder, RoadBreak, RandomResize, Canny, Histogram, HistNorm
from torchvision import transforms
import numpy as np
from .mydataset import RoadAllData, RoadAllDataTest
import warnings
import torch




def build_special_transform():
    return {
        'blue': RandomBlue(probability=1.0), 
        'white': RandomWhite(probability=1.0),
        'stop': StopLine(probability=1.0),
        'yelshort': RandomYellowShort(probability=1.0),
        'yellong': RandomYellowLong(probability=1.0),
        'roadsh': RoadShoulder(probability=1.0),
        'roadbr': RoadBreak(probability=1.0)
    }

def build_transform(size=(496, 224), rate_for_crop=None, rand_filp=False, to='Tensor', norm=True, updown_rand_crop=False, translate=False, 
    canny=False, hist=False, hist_level=16, colorjittor=[0,0,0,0], gasuss_noise=False):

    assert (rate_for_crop == None) or (rate_for_crop >= 1)
    assert to in ['Tensor', 'ndarray']
    assert not (to == 'ndarray' and  norm)
    if canny and norm:  
        norm = False
        warnings.warn('if use canny, norm will be disabled.')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # normalize = transforms.Normalize(mean=[0.714, 0.711, 0.678], std=[0.333, 0.345, 0.302])

    transform_list = []
    if gasuss_noise:
        transform_list.append(GasussNoise(mean=0, var=0.001))
    if updown_rand_crop and (updown_rand_crop != [0,0,0]):
        transform_list.append(UpDownCrop(probability=1.0, crop_up1=updown_rand_crop[0], crop_up2=updown_rand_crop[1], crop_down=updown_rand_crop[2]))
    if translate:
        transform_list.append(transforms.RandomAffine(degrees=0, translate=(0,0.1), fill=(0,0,0)))
    if canny:
        transform_list.append(Canny())
    if hist:
        hist_mean = np.load('data/hist_mean_{}.npy'.format(hist_level))
        hist_std = np.load('data/hist_std_{}.npy'.format(hist_level))

        transform_list.append(Histogram(hist_level))
        transform_list.append(HistNorm(hist_mean, hist_std))
        transform_list.append(torch.Tensor)
        return transforms.Compose(transform_list)
        
    if rate_for_crop and (rate_for_crop > 1):
        short = min(size)
        transform_list.append(RandomResize(short, rate_for_crop))
    if rand_filp:
        if 'h' in rand_filp:
            transform_list.append(transforms.RandomHorizontalFlip())
        if 'v' in rand_filp:
            transform_list.append(transforms.RandomVerticalFlip())
    if rate_for_crop and (rate_for_crop > 1):
        transform_list.append(MyRandomCrop(short))

    transform_list.append(transforms.Resize(size))
    if colorjittor != [0,0,0,0]:
        cj = colorjittor
        transform_list.append(transforms.ColorJitter(cj[0], cj[1], cj[2], cj[3]))
    if to == 'Tensor':
        transform_list.append(transforms.ToTensor())
    elif to == 'ndarray':
        transform_list.append(np.array)
    
    if norm:
        transform_list.append(normalize)
     
    return transforms.Compose(transform_list)



