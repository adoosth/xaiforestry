import numpy as np
import os
import random
from PIL import Image
import matplotlib.image as mpimg
import cv2

def grabcut(file_path):
    img = np.array(cv2.imread(file_path)).astype('uint8')
    mask=np.zeros(img.shape[:2],dtype="uint8")    
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    size=len(img)
    rect=(0,0,size-1,size-1)
    (mask, bgModel, fgModel) = cv2.grabCut(img=img, mask=mask, rect=rect,bgdModel=bgModel,fgdModel=fgModel,iterCount=50, mode=cv2.GC_INIT_WITH_RECT)
    threshold = np.mean(mask)
    for x in range(size):
        for y in range(size):
            if(mask[x][y] >= threshold):
                mask[x][y] = 0
            else:
                mask[x][y] = 255
    return mask

def minimum_enclosing_rectangle(file_path):
    img = cv2.imread(file_path)
    mask = grabcut(img)
    mincol = np.min(np.where(~np.min(mask == 255, axis=0))[0])
    maxcol = np.max(np.where(~np.min(mask == 255, axis=0))[0])
    minrow = np.min(np.where(~np.min(mask == 255, axis=1))[0])
    maxrow = np.max(np.where(~np.min(mask == 255, axis=1))[0])
    mask[minrow:maxrow+1, mincol:maxcol+1] = 0
    return mask

def load_from_file(file_path, label, mask_folder):
    img = cv2.imread(file_path)
    mask = np.zeros(img.shape[:2],dtype="uint8")
    mask_file_path = os.path.join(mask_folder, 'mask.' + os.path.split(file_path)[1])
    if (label == 1 and os.path.isfile(mask_file_path)):
        mask = np.mean(cv2.imread(mask_file_path), axis=2).astype('uint8')
    return mask