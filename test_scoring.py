import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from structured_data.scoring import compute_intra_object_deviation, compute_inter_object_deviation
from structured_data.unstructured_image import UnstructuredImage, GroupOfUnstructruedImages
from structured_data.augmentations import Augmentations

path = "/Users/fmuenke/datasets/second_dataset/train/im1"

augment = Augmentations(flip=0.5, color_shift=0.5, noise=0.9)

goi = GroupOfUnstructruedImages([])
for img_f in os.listdir(path):
    u_img = UnstructuredImage(img_f, time_stamp=1, coordinates=(1, 1), path_to_image=os.path.join(path, img_f), augment=augment)
    goi.add(u_img)

print(len(goi))

list_of_clust = goi.synth_clusters(augment, n_clusters=32)
print(len(list_of_clust))