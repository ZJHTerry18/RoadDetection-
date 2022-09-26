import torch
import numpy as np
from torchvision import transforms
import random
from PIL import Image
import math
import cv2


class Histogram(torch.nn.Module):
    def __init__(self, levels=16):
        self.levels = levels
        self.max_v = levels ** 3
        self.v_num = (256 // levels)
        
    def __call__(self, img):
        img = np.array(img)
        img = img // self.v_num
        img = img.astype(int)
        values = img[:, :, 2] * (self.levels * self.levels)
        values += img[:, :, 1] * self.levels
        values += img[:, :, 0]
        values = values.reshape(-1)
        values = np.insert(values, len(values), self.max_v)
        return np.bincount(values)

class HistNorm(torch.nn.Module):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.mask = (self.std > 0)
    
    def __call__(self, img):
        return (img[self.mask] - self.mean[self.mask]) / self.std[self.mask]


class RandomResize(torch.nn.Module):
    def __init__(self, size, enlarge_rate):
        self.size = size
        self.max_size = max(int(size * enlarge_rate + 0.5), size+1)
        self.resizeGroup = [transforms.Resize(sz) for sz in range(size, self.max_size)]
    def __call__(self, img):
        sz = np.random.randint(self.size, self.max_size)
        return self.resizeGroup[sz-self.size](img)

class GasussNoise(object):
    def __init__(self, mean=0, var=0.001):
        self.mean = mean
        self.var = var
        self.probability = 0.2

    def __call__(self, img):
        if random.uniform(0,1) > self.probability:
            return img
        img = np.array(img)
        image = np.array(img / 255, dtype=float)
        noise = np.random.normal(self.mean, self.var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out * 255)
        return out


class Canny(torch.nn.Module):
    def __call__(self, img):
        img = np.array(img)
        r, g, b = cv2.split(img)
        r = cv2.Canny(r, 128, 200)
        g = cv2.Canny(g, 128, 200)
        b = cv2.Canny(b, 128, 200)
        img_ = cv2.merge([r, g, b])
        img_ = cv2.dilate(img_, np.ones((3,3), dtype=np.uint8))
        return Image.fromarray(img_)



class MyRandomCrop(torch.nn.Module):
    def __init__(self, size=224):
        super().__init__()
        self.size = size
    def forward(self, img):
        w, h = img.size
        if h > w:
            nh = int(h * self.size / w + 0.5)
            nw = self.size
        else:
            nw = int(w * self.size / h + 0.5)
            nh = self.size
        dh = np.random.randint(0, h-nh+1)
        dw = np.random.randint(0, w-nw+1)
        return img.crop((dw, dh, dw+nw, dh+nh))

class UpDownCrop(object):
    def __init__(self, probability=1.0, crop_up1=0.9, crop_up2=0.9, crop_down=0.9, 
    upcolor1=[0,0,0], upcolor2=[244,121,153], downcolor=[24,24,24], fillcolor=[0,0,0]):
        self.probability = probability
        self.crop_up1 = crop_up1
        self.crop_up2 = crop_up2
        self.crop_down = crop_down
        self.upcolor1 = upcolor1
        self.upcolor2 = upcolor2
        self.downcolor = downcolor
        self.fillcolor = fillcolor
    
    def __call__(self, img):
        if random.uniform(0,1) > self.probability:
            return img
        img = np.array(img)

        ## randomly crop upper black bar and upper red bar
        if random.uniform(0,1) < self.crop_up1:
            res = np.ones((img.shape[0], img.shape[1])).astype(bool)
            value = self.upcolor1
            for i in range(3):
                res = res & (img[:, :, i] >= int(value[i]))
                res = res & (img[:, :, i] <= int(value[i] + 1))
            x_index = np.nonzero(res)[0]
            if len(x_index) > 0:
                x_index[x_index > 130] = 130
                x_bound_1 = np.max(x_index) + 20
                img[:x_bound_1, :, :] = self.fillcolor

            if random.uniform(0,1) < self.crop_up2:
                res = np.ones((img.shape[0], img.shape[1])).astype(bool)
                value = self.upcolor2
                for i in range(3):
                    res = res & (img[:, :, i] >= int(value[i]))
                    res = res & (img[:, :, i] <= int(value[i] + 1))
                x_index = np.nonzero(res)[0]
                if len(x_index) > 0:
                    x_index[x_index > 350] = 350
                    x_bound_2 = np.max(x_index) + 20
                    img[x_bound_1:x_bound_2, :, :] = self.fillcolor

        
        ## randomly crop bottom black bar
        if random.uniform(0,1) < self.crop_down:
            res = np.ones((img.shape[0], img.shape[1])).astype(bool)
            value = self.downcolor
            for i in range(3):
                res = res & (img[:, :, i] >= int(value[i]))
                res = res & (img[:, :, i] <= int(value[i] + 1))
            x_index = np.nonzero(res)[0]
            if len(x_index) > 0:
                x_index[x_index < 2100] = 2100
                x_bound = np.min(x_index) - 20
                img[x_bound:, :, :] = self.downcolor
        
        return Image.fromarray(img)

class RandomBlue(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=1, sl=0.02, sh=0.05, r1=0.3, mean=[0.431372549019607, 0.623529411764705, 1]):
        self.probability = probability
        self.mean = np.array(mean) * 255
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        img = np.array(img)
        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(self.sl, self.sh) * area
        aspect_ratio = random.uniform(self.r1, 1 / self.r1)
        res = np.ones((img.shape[0], img.shape[1])).astype(bool)
        value = np.array([0.14901960784, 0.172549019607843, 0.207843137254901]) * 255
        for i in range(3):
            res = res & (img[:, :, i] >= int(value[i]))
            res = res & (img[:, :, i] <= int(value[i] + 1))
        index = np.nonzero(res)  # 返回索引
        if len(index[0]) > 0:
            choice = random.randint(0, len(index[0]) - 1)
            # print(choice)
            position = [index[0][choice], index[1][choice]]
            h = int(round(math.sqrt(target_area * aspect_ratio)))  # round返回四舍五入的值
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            x1 = position[0]
            y1 = position[1]
            x2 = min(x1 + h, img.shape[0])
            y2 = min(y1 + w, img.shape[1])
            img[x1:x2, y1:y2, :] = self.mean * 0.4 + img[x1:x2, y1:y2, :] * 0.6
            return Image.fromarray(img)
        return Image.fromarray(img)

'''
使用：
transforms.Compose([
    RandomWhite(probability=1),
])
probability代表使用该增广的概率
'''
class RandomWhite(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=1, sl=0.00083, sh=0.039444, r1=0.3, mean=[1., 1., 1.]):
        self.probability = probability
        self.mean = np.array(mean) * 255
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.pattern = self.getPattern().type(torch.long)

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        img = np.array(img)
        area = img.shape[0] * img.shape[1]
        # ifroad = torch.ones(1, 2340, 1080) * 0.14901960784
        # ifroad1 = torch.ones(1, 2340, 1080) * 0.172549019607843
        # ifroad2 = torch.ones(1, 2340, 1080) * 0.207843137254901
        # bool = ifroad.eq(img[0])
        # bool1 = ifroad1.eq(img[1])
        # bool2 = ifroad2.eq(img[2])
        # boolsum = bool & bool1 & bool2
        # index = torch.nonzero(boolsum)  # 返回索引
        res = np.ones((img.shape[0], img.shape[1])).astype(bool)
        value = np.array([0.14901960784, 0.172549019607843, 0.207843137254901]) * 255
        for i in range(3):
            res = res & (img[:, :, i] >= int(value[i]))
            res = res & (img[:, :, i] <= int(value[i] + 1))
        index = np.nonzero(res)  # 返回索引
        if len(index[0] > 0):
            choice = random.randint(0, len(index[0]) - 1)
            choice1 = random.randint(0, 15)
            # print(choice)
            position = [index[0][choice], index[1][choice]]
            pattern1 = self.pattern[choice1].view(300, 2)
            xr = position[0] + pattern1[:,1]
            yr = position[1] + pattern1[:,0]
            ind = torch.where(xr < img.shape[0], torch.ones_like(xr, dtype=bool), torch.zeros_like(xr, dtype=bool)) & \
                    torch.where(yr < img.shape[1], torch.ones_like(xr, dtype=bool), torch.zeros_like(xr, dtype=bool))
            img[xr[ind], yr[ind], :] = self.mean
            return Image.fromarray(img)
        return Image.fromarray(img)
    
    def getPattern(self):
        pattern = torch.zeros(20,75,4,2)
        order1 = torch.arange(0, 4, 1)
        order2 = torch.ones(4)
        for i in range(75):
            pattern[0, i, :, :] = torch.stack([order1, i * order2], 1)
            pattern[1, i, :, :] = torch.stack([order1 + i/75, i * order2], 1)
            pattern[8, i, :, :] = torch.stack([order1 + i / 2, i * order2], 1)
            pattern[9, i, :, :] = torch.stack([order1 - i / 2, i * order2], 1)
            pattern[10, i, :, :] = torch.stack([order1 + i / 3, i * order2], 1)
            pattern[11, i, :, :] = torch.stack([order1 - i / 3, i * order2], 1)
            pattern[12, i, :, :] = torch.stack([order1 + i / 4, i * order2], 1)
            pattern[13, i, :, :] = torch.stack([order1 - i / 4, i * order2], 1)
            pattern[14, i, :, :] = torch.stack([order1 + i / 5, i * order2], 1)
            pattern[15, i, :, :] = torch.stack([order1 - i / 5, i * order2], 1)
        for i in range(50):#斜着的话像素值要少一些，否则太长了
            pattern[2, i, :, :] = torch.stack([order1 + i, i * order2], 1)
            pattern[3, i, :, :] = torch.stack([order1 - i, i * order2], 1)
        for i in range(35):
            pattern[4, i, :, :] = torch.stack([order1 + 2 * i, i * order2], 1)
            pattern[5, i, :, :] = torch.stack([order1 - 2 * i, i * order2], 1)
        for i in range(25):
            pattern[6, i, :, :] = torch.stack([order1 + 3 * i, i * order2], 1)
            pattern[7, i, :, :] = torch.stack([order1 - 3 * i, i * order2], 1)
            pattern[6, i+30, :, :] = torch.stack([order1 + 3 * i, i * order2+1], 1)
            pattern[7, i+30, :, :] = torch.stack([order1 - 3 * i, i * order2+1], 1)#多给加一行，不然线太细了
        return pattern


'''
用于模拟停止线
'''
class StopLine(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """
    def __init__(self, probability=1, sl=0.00083, sh=0.039444, r1=0.3, mean=[1, 1, 1]):
        self.probability = probability
        self.mean = np.array(mean) * 255
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.pattern = self.getPattern().type(torch.long)

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        img = np.array(img)
        res = np.ones((img.shape[0], img.shape[1])).astype(bool)
        value = np.array([0.14901960784, 0.172549019607843, 0.207843137254901]) * 255
        for i in range(3):
            res = res & (img[:, :, i] >= int(value[i]))
            res = res & (img[:, :, i] <= int(value[i] + 1))
        index = np.nonzero(res)  # 返回索引
        if len(index[0]) > 0:
            choice = random.randint(0, len(index[0]) - 1)
            choice1 = random.randint(0, 15)
            # print(choice)
            position = [index[0][choice], index[1][choice]]
            #print (position)
            #print('choice1:',choice1)
            #print(img.size())
            pattern1 = self.pattern[choice1].view(375, 2)
            xr = position[0] + pattern1[:,1]
            yr = position[1] + pattern1[:,0]
            ind = torch.where(xr < img.shape[0], torch.ones_like(xr, dtype=bool), torch.zeros_like(xr, dtype=bool)) & \
                    torch.where(yr < img.shape[1], torch.ones_like(xr, dtype=bool), torch.zeros_like(xr, dtype=bool))
            img[xr[ind], yr[ind], :] = self.mean
            return Image.fromarray(img)
        return Image.fromarray(img)
    
    def getPattern(self):
        pattern = torch.zeros(20,75,5,2)
        order1 = torch.arange(0, 5, 1)
        order2 = torch.ones(5)
        for i in range(75):
            pattern[0, i, :, :] = torch.stack([order1, i * order2], 1)
            pattern[1, i, :, :] = torch.stack([order1 + i/75, i * order2], 1)
            pattern[8, i, :, :] = torch.stack([order1 + i / 2, i * order2], 1)
            pattern[9, i, :, :] = torch.stack([order1 - i / 2, i * order2], 1)
            pattern[10, i, :, :] = torch.stack([order1 + i / 3, i * order2], 1)
            pattern[11, i, :, :] = torch.stack([order1 - i / 3, i * order2], 1)
            pattern[12, i, :, :] = torch.stack([order1 + i / 4, i * order2], 1)
            pattern[13, i, :, :] = torch.stack([order1 - i / 4, i * order2], 1)
            pattern[14, i, :, :] = torch.stack([order1 + i / 5, i * order2], 1)
            pattern[15, i, :, :] = torch.stack([order1 - i / 5, i * order2], 1)
        for i in range(50):#斜着的话像素值要少一些，否则太长了
            pattern[2, i, :, :] = torch.stack([order1 + i, i * order2], 1)
            pattern[3, i, :, :] = torch.stack([order1 - i, i * order2], 1)
        for i in range(35):
            pattern[4, i, :, :] = torch.stack([order1 + 2 * i, i * order2], 1)
            pattern[5, i, :, :] = torch.stack([order1 - 2 * i, i * order2], 1)
        for i in range(25):
            pattern[6, i, :, :] = torch.stack([order1 + 3 * i, i * order2], 1)
            pattern[7, i, :, :] = torch.stack([order1 - 3 * i, i * order2], 1)
            pattern[6, i+30, :, :] = torch.stack([order1 + 3 * i, i * order2+1], 1)
            pattern[7, i+30, :, :] = torch.stack([order1 - 3 * i, i * order2+1], 1)#多给加一行，不然线太细了
        return pattern

'''
用于模拟黄色车道线
'''
class RandomYellowShort(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """
    def __init__(self, probability=1, sl=0.00083, sh=0.039444, r1=0.3, mean=[0.956862745098039, 0.847058823529411, 0.109803921568627]):
        self.probability = probability
        self.mean = np.array(mean) * 255
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.pattern = self.getPattern().type(torch.long)

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        img = np.array(img)
        res = np.ones((img.shape[0], img.shape[1])).astype(bool)
        value = np.array([0.14901960784, 0.172549019607843, 0.207843137254901]) * 255
        for i in range(3):
            res = res & (img[:, :, i] >= int(value[i]))
            res = res & (img[:, :, i] <= int(value[i] + 1))
        index = np.nonzero(res)  # 返回索引
        if len(index[0]) > 0:
            choice = random.randint(0, len(index[0]) - 1)
            choice1 = random.randint(0, 15)
            # print(cho[ice)
            position = [index[0][choice], index[1][choice]]
            #print (position)
            #print('choice1:',choice1)
            #print(img.size())
            pattern1 = self.pattern[choice1].view(300, 2)
            xr = position[0] + pattern1[:,1]
            yr = position[1] + pattern1[:,0]
            ind = torch.where(xr < img.shape[0], torch.ones_like(xr, dtype=bool), torch.zeros_like(xr, dtype=bool)) & \
                    torch.where(yr < img.shape[1], torch.ones_like(xr, dtype=bool), torch.zeros_like(xr, dtype=bool))
            img[xr[ind], yr[ind], :] = self.mean
        return Image.fromarray(img)
    
    def getPattern(self):
        pattern = torch.zeros(20,75,4,2)
        order1 = torch.arange(0, 4, 1)
        order2 = torch.ones(4)
        for i in range(75):
            pattern[0, i, :, :] = torch.stack([order1, i * order2], 1)
            pattern[1, i, :, :] = torch.stack([order1 + i/75, i * order2], 1)
            pattern[8, i, :, :] = torch.stack([order1 + i / 2, i * order2], 1)
            pattern[9, i, :, :] = torch.stack([order1 - i / 2, i * order2], 1)
            pattern[10, i, :, :] = torch.stack([order1 + i / 3, i * order2], 1)
            pattern[11, i, :, :] = torch.stack([order1 - i / 3, i * order2], 1)
            pattern[12, i, :, :] = torch.stack([order1 + i / 4, i * order2], 1)
            pattern[13, i, :, :] = torch.stack([order1 - i / 4, i * order2], 1)
            pattern[14, i, :, :] = torch.stack([order1 + i / 5, i * order2], 1)
            pattern[15, i, :, :] = torch.stack([order1 - i / 5, i * order2], 1)
        for i in range(50):#斜着的话像素值要少一些，否则太长了
            pattern[2, i, :, :] = torch.stack([order1 + i, i * order2], 1)
            pattern[3, i, :, :] = torch.stack([order1 - i, i * order2], 1)
        for i in range(35):
            pattern[4, i, :, :] = torch.stack([order1 + 2 * i, i * order2], 1)
            pattern[5, i, :, :] = torch.stack([order1 - 2 * i, i * order2], 1)
        for i in range(25):
            pattern[6, i, :, :] = torch.stack([order1 + 3 * i, i * order2], 1)
            pattern[7, i, :, :] = torch.stack([order1 - 3 * i, i * order2], 1)
            pattern[6, i+30, :, :] = torch.stack([order1 + 3 * i, i * order2+1], 1)
            pattern[7, i+30, :, :] = torch.stack([order1 - 3 * i, i * order2+1], 1)#多给加一行，不然线太细了
        return pattern

'''
用于模拟黄色中心线
'''
class RandomYellowLong(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """
    def __init__(self, probability=1, sl=0.00083, sh=0.039444, r1=0.3, mean=[0.956862745098039, 0.847058823529411, 0.109803921568627]):
        self.probability = probability
        self.mean = np.array(mean) * 255
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.pattern = torch.stack([self.getPattern(1).type(torch.long),self.getPattern(2).type(torch.long),self.getPattern(3).type(torch.long),self.getPattern(4).type(torch.long),self.getPattern(5).type(torch.long)])

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        img = np.array(img)
        res = np.ones((img.shape[0], img.shape[1])).astype(bool)
        value = np.array([0.14901960784, 0.172549019607843, 0.207843137254901]) * 255
        for i in range(3):
            res = res & (img[:, :, i] >= int(value[i]))
            res = res & (img[:, :, i] <= int(value[i] + 1))
        index = np.nonzero(res)  # 返回索引
        if len(index[0]) > 0:
            choice = random.randint(0, len(index[0]) - 1)
            choice1 = random.randint(0, 15)
            length = random.randint(1, 5)
            # print(choice)
            position = [index[0][choice], index[1][choice]]
            #print (position)
            #print('length:',length)
            #print('choice1:', choice1)
            #print(img.size())
            pattern1 = self.pattern[length-1][choice1].view(300 * 5, 2)
            xr = position[0] + pattern1[0:300 * length,1]
            yr = position[1] + pattern1[0:300 * length,0]
            ind = torch.where(xr < img.shape[0], torch.ones_like(xr, dtype=bool), torch.zeros_like(xr, dtype=bool)) & \
                    torch.where(yr < img.shape[1], torch.ones_like(xr, dtype=bool), torch.zeros_like(xr, dtype=bool))
            img[xr[ind], yr[ind], :] = self.mean
            return Image.fromarray(img)
        return Image.fromarray(img)
    
    def getPattern(self,length):
        pattern = torch.zeros(20, 75 * 5, 4 , 2)
        order1 = torch.arange(0, 4, 1)
        order2 = torch.ones(4)
        for i in range(75 * length):
            pattern[0, i, :, :] = torch.stack([order1, i * order2], 1)
            pattern[1, i, :, :] = torch.stack([order1 + i/(75 * length), i * order2], 1)
            pattern[8, i, :, :] = torch.stack([order1 + i / 2, i * order2], 1)
            pattern[9, i, :, :] = torch.stack([order1 - i / 2, i * order2], 1)
            pattern[10, i, :, :] = torch.stack([order1 + i / 3, i * order2], 1)
            pattern[11, i, :, :] = torch.stack([order1 - i / 3, i * order2], 1)
            pattern[12, i, :, :] = torch.stack([order1 + i / 4, i * order2], 1)
            pattern[13, i, :, :] = torch.stack([order1 - i / 4, i * order2], 1)
            pattern[14, i, :, :] = torch.stack([order1 + i / 5, i * order2], 1)
            pattern[15, i, :, :] = torch.stack([order1 - i / 5, i * order2], 1)
        for i in range(50 * length):#斜着的话像素值要少一些，否则太长了
            pattern[2, i, :, :] = torch.stack([order1 + i, i * order2], 1)
            pattern[3, i, :, :] = torch.stack([order1 - i, i * order2], 1)
        for i in range(35 * length):
            pattern[4, i, :, :] = torch.stack([order1 + 2 * i, i * order2], 1)
            pattern[5, i, :, :] = torch.stack([order1 - 2 * i, i * order2], 1)
        for i in range(25 * length):
            pattern[6, i, :, :] = torch.stack([order1 + 3 * i, i * order2], 1)
            pattern[7, i, :, :] = torch.stack([order1 - 3 * i, i * order2], 1)
            pattern[6, i + (25+5) * length, :, :] = torch.stack([order1 + 3 * i, i * order2+1], 1)
            pattern[7, i + (25+5) * length, :, :] = torch.stack([order1 - 3 * i, i * order2+1], 1)#多给加一行，不然线太细了
        return pattern

'''
用于模拟路肩问题
'''
class RoadShoulder(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=1, sl=0.001, sh=0.002, r1=0.3, mean=[0.14901960784, 0.172549019607843, 0.207843137254901]):
        self.probability = probability
        self.mean = np.array(mean) * 255
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean2 = torch.ones(3) * 255
        self.mean1 = torch.tensor([0.576470588235294,0.615686274509803, 0.741176470588235]) * 255

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img
        img = np.array(img)
        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(self.sl, self.sh) * area
        aspect_ratio = random.uniform(self.r1, 1 / self.r1)
        res = np.ones((img.shape[0], img.shape[1])).astype(bool)
        value = np.array([0.14901960784, 0.172549019607843, 0.207843137254901]) * 255
        for i in range(3):
            res = res & (img[:, :, i] >= int(value[i]))
            res = res & (img[:, :, i] <= int(value[i] + 1))
        index = np.nonzero(res)  # 返回索引
        if len(index[0]) > 0:
            choice = random.randint(0, len(index[0]) - 1)
            # print(choice)
            position = [index[0][choice], index[1][choice]]
            h = max(int(round(math.sqrt(target_area * aspect_ratio))), 22) # round返回四舍五入的值
            w = max(int(round(math.sqrt(target_area / aspect_ratio))), 22)
            x1 = position[0]
            y1 = position[1]
            x2 = min(x1 + h, img.shape[0])
            y2 = min(y1 + w, img.shape[1])
            img[x1:x2, y1:y2, :] = self.mean1
            x1 = x1 + 8
            y1 = y1 + 8
            img[x1:x2 - 16, y1:y2 - 16, :] = self.mean2
            x1 = x1 + 3
            y1 = y1 + 3
            img[x1:x2 - 22, y1:y2 - 22, :] = self.mean
            return Image.fromarray(img)
        return Image.fromarray(img)

'''
用于模拟路面断裂问题
'''
class RoadBreak(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=1, sl=0.02, sh=0.05, r1=0.3, mean=[0.909803921568627, 0.949019607843137, 0.8666666666666]):
        self.probability = probability
        self.mean = np.array(mean) * 255
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean1 = torch.ones(3) * 255
        self.mean2 = torch.tensor([0.576470588235294,0.615686274509803, 0.741176470588235]) * 255

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img
        img = np.array(img)
        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(self.sl, self.sh) * area
        aspect_ratio = random.uniform(self.r1, 1 / self.r1)
        res = np.ones((img.shape[0], img.shape[1])).astype(bool)
        value = np.array([0.14901960784, 0.172549019607843, 0.207843137254901]) * 255
        for i in range(3):
            res = res & (img[:, :, i] >= int(value[i]))
            res = res & (img[:, :, i] <= int(value[i] + 1))
        index = np.nonzero(res)  # 返回索引
        if len(index[0]) > 0:
            choice = random.randint(0, len(index[0]) - 1)
            # print(choice)
            position = [index[0][choice], index[1][choice]]
            h = max(int(round(math.sqrt(target_area * aspect_ratio))), 22) # round返回四舍五入的值
            w = max((round(math.sqrt(target_area / aspect_ratio))), 22)
            x1 = position[0]
            y1 = position[1]
            x2 = min(x1 + h, img.shape[0])
            y2 = min(y1 + w, img.shape[1])
            img[x1:x2, y1:y2, :] = self.mean1
            x1 = x1 + 3
            y1 = y1 + 3
            img[x1:x2 - 6, y1:y2 - 6, :] = self.mean2
            x1 = x1 + 8
            y1 = y1 + 8
            img[x1:x2 - 22, y1:y2 - 22, :] = self.mean
            return Image.fromarray(img)
        return Image.fromarray(img)