from PIL import Image
import os
import numpy as np

data_path = r'/data/zhaojiahe/HUAWEIAI/dataset/train_image'

if __name__ == '__main__':
    imglist = os.listdir(os.path.join(data_path, 'labeled_data'))
    for imname in imglist:
        img = Image.open(os.path.join(data_path, imname)).convert('RGB')
        img = np.array(img)
        print(img)