import cv2
import os
import random
import numpy as np

def random_circle(height, width, radiusRange):
    """
    Creates a random circle

    This function returns the center and radius of a randomly
    generated circle within the given dimensions and range.

    Parameters
    ----------
    height : int
        Height of the image
    width : int
        Width of the image
    radiusRange : int
        Possible range for the circle's radius. The random is uniformly distributed.

    Returns
    -------
    (int, int, int)
        x: the x coordinates of the center point
        y: the y coordinates of the center point
        r: the circle's radius

    Examples
    --------
    >>> random_circle(10, 10, (0, 5))
    (3, 7, 2)
    """
    cx, cy = random.randint(0, width-1), random.randint(0, height-1)
    cr = random.randint(radiusRange[0], radiusRange[1])
    return cx, cy, cr



def gen_images(src_dir, dst_dir, filenames, radiusRange = (0, 50), num_circles = 1, copies = 1, color = (255, 255, 255), opacity = 1.0, save_mask = True):
    """
    Generates augmented images by adding a random circle to each image.

    This function reads all given images and adds a random circle to each image
    and saves them in the destination folder

    Parameters
    ----------
    src_dir : string
        Folder of source images
    dst_dir : int
        Folder of destination images
    filenames : list[string]
        List of filenames within src_dir to be processed
    radiusRange : (int, int)
        Range of possible radius values
    num_circles : int
        number of random circles per each generated image
    copies : int
        number of generated images per source image
    val_frac : float
        Fraction of data pertaining to validation set
    test_frac : float
        Fraction of data pertaining to test set
    save_mask : boolean
        Whether to save mask files

    Returns
    -------
    True
        if successfully completed process
    False
        otherwise
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    if type(filenames) is not list:
        print("Incorrect parameter for filenames. Should be list.")
        return False
    for filename in filenames:
        img = cv2.imread(src_dir + filename, cv2.IMREAD_UNCHANGED)
        for i in range(copies):
            random_circles = [random_circle(img.shape[0], img.shape[1], radiusRange=radiusRange) for k in range(num_circles)]
            overlay = img.copy()
            ### Top left pixel for color
            #for (cx, cy, cr) in random_circles: overlay = cv2.circle(overlay, (cx, cy), cr, [int(i) for i in tuple(img[0,0])], -1)
            ### Argument for color
            for (cx, cy, cr) in random_circles: overlay = cv2.circle(overlay, (cx, cy), cr, color, -1)
            img = cv2.addWeighted(img, 1 - opacity, overlay, opacity, 0)
            cv2.imwrite(dst_dir + filename, img)
            if save_mask:
                mask = np.ones(img.shape, np.uint8)*255
                for (cx, cy, cr) in random_circles: mask = cv2.circle(mask, (cx, cy), cr, (0, 0, 0), -1)
                cv2.imwrite(dst_dir + 'mask.' + filename, mask)
    return True

def gen_all_images(src_dir, dst_dir, radiusRange = (0, 50), num_circles = 1, copies = 1, save_mask = True):
    """
    Generates augmented images by adding a random circle to each image in a database.

    This function reads all given images and adds a random circle to each image
    in a databse (folder/subfolders) and saves them in the destination folder

    Parameters
    ----------
    src_dir : string
        Folder of source images with format "folder/subfolders"
    dst_dir : int
        Folder of destination images
    radiusRange : (int, int)
        Range of possible radius values
    num_circles : int
        number of random circles per each generated image
    copies : int
        number of generated images per source image
    """
    subdirs = os.listdir(src_dir)
    for subdir in subdirs:
        dta_dir = src_dir + '/' + subdir + '/'
        gen_dir = dst_dir + '/' + subdir + '/'
        os.makedirs(gen_dir, exist_ok=True)
        filenames = os.listdir(dta_dir)
        gen_images(dta_dir, gen_dir, filenames, radiusRange, num_circles, copies, save_mask)

#src_dir = '../../PlantVillage-Dataset/raw/color'
#dst_dir = 'gen'
#gen_all_images(src_dir, dst_dir)

src_dir = '../../adoosth/PepperSetHF/Healthy/'
dst_dir = '../../adoosth/PepperSetHF/Fake3C30/'

gen_images(src_dir, dst_dir, os.listdir(src_dir), radiusRange = (0, 20), num_circles=3, copies=1, opacity=0.3, save_mask=True)

os.chdir