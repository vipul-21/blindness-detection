# dump old large dataset into another directory
from config import *
from data_aug import *

from torch.utils.data import Subset, ConcatDataset
from PIL import Image
import os
from multiprocessing import Pool

img_shape = (256, 256)
transform = CircleCrop(img_shape)

dataset1 = BD_Dataset(OLD_TRAIN_CSV, OLD_TRAIN_DIR,
                      transform, True, False, False, True)
dataset2 = BD_Dataset(OLD_TEST_CSV, OLD_TEST_DIR,
                      transform, True, False, False, True)
dataset2.data = dataset2.data.drop(["Usage"], axis=1)
dataset = ConcatDataset([dataset1, dataset2])
# dataset = BD_Dataset(TRAIN_CSV, TRAIN_DIR, transform, True, False, True, True)

if not os.path.isdir(OLD_PREPROCESSED_DIR):
    os.mkdir(OLD_PREPROCESSED_DIR)


def dumpImage(ind):
    img, id = dataset[ind]
    img.save(OLD_PREPROCESSED_DIR+id+".jpeg")

p = Pool(16)
p.map(dumpImage, [i for i in range(len(dataset))])