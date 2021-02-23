from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch


def findPeak(old_mask, label, num_k=30):
    res = []
    topval = []
    mask = np.zeros((old_mask.shape[0]+2, old_mask.shape[1]+2))
    # print(old_mask.shape)
    h, w = mask.shape[:2]
    mask[1:w-1, 1:h-1] = old_mask
    for i in range(1, w-1):
        for j in range(1, h-1):
            if mask[i, j] == np.max(mask[i-1:i+2, j-1:j+2]):
                res.append([label, j-1, i-1])
                topval.append(mask[i, j])
    res = np.array(res)
    topval = np.array(topval)

    return res[np.argsort(topval)[-num_k:]]


def get_center_from_center_mask(center_mask):
    height, width, depth = center_mask.shape[1:]
    center = []
    for i in range(depth):
        if (center_mask[0, :, :, i]).max() > 0.7:
            center.extend(findPeak(center_mask[0, :, :, i], i))
    return center


def draw(img, centers, offset_mask, size_mask):
    # centers: list of center point coordinate in the strided image
    # offset_mask: the offset mask showing how off the coordinate in strided image
    # is to the one in original image
    # size_mask: the w and h of the boxes
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    img = img.numpy().astype('uint8')
    for center in centers:
        clr = colors[center[0]]
        # print(clr)
        center_x = center[1]*4 + offset_mask[center[1], center[2], 0]
        center_y = center[2]*4 + offset_mask[center[1], center[2], 1]
        #print(center_x, center_y)
        w = size_mask[center[1], center[2], 0]
        h = size_mask[center[1], center[2], 1]
        x0, x1, y0, y1 = center_x-w//2, center_x+w//2, center_y-h//2, center_y+h//2

        #print(x0, x1, y0, y1)

        img = cv2.rectangle(img, (int(x0), int(y0)),
                            (int(x1), int(y1)), clr, 2)
    return img
