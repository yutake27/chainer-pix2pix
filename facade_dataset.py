import os
from PIL import Image
import numpy as np
from pathlib import Path

from chainer.dataset import dataset_mixin

class FacadeDataset(dataset_mixin.DatasetMixin):
    def __init__(self, imgDir='../../Image', contourDir='../../Image_Contour', data_num = 400):
        print("load dataset start")
        print("Original Image from: {}".format(imgDir))
        print("Contour  Image from: {}".format(contourDir))
        print("Data num: {}".format(data_num))
        imgDir = Path(imgDir)
        contourDir = Path(contourDir)
        self.dataset = []
        imgs = [img for img in imgDir.iterdir()][:data_num]
        for img_path in imgs:
            img = Image.open(img_path)
            label = Image.open(contourDir/img_path.name)
            label = label.convert(mode='RGB')
            w_in = 512
            img = img.resize((w_in, w_in), Image.BILINEAR)
            label = label.resize((w_in, w_in), Image.NEAREST)

            img = np.asarray(img).astype('f').transpose(2,0,1)/128.0-1.0
            label = np.asarray(label).astype('f').transpose(2,0,1)/128.0-1.0

            self.dataset.append((img,label))
        print("load dataset done")

    def __len__(self):
        return len(self.dataset)

    # return (label, img)
    def get_example(self, i, crop_width=256):
        return self.dataset[i][1], self.dataset[i][0]
