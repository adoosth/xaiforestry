
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def explanable_MSE(gradcam_img, mask_raw):
    mask = 1.0 - np.mean(mask_raw, axis=0)
    #fig, axes = plt.subplots(1, 2)
    #axes[0].imshow(gradcam_img, cmap='gray')
    #axes[1].imshow(mask, cmap='gray')
    #plt.show()
    ans = np.mean((mask - gradcam_img)**2)
    return ans

def explanable_RMSE(gradcam_img, mask_raw):
    mask = 1.0 - np.mean(mask_raw, axis=0)
    return np.sqrt(np.mean((mask - gradcam_img)**2))

def explanable_jaccard(gradcam_img, mask_raw, show_plots = False):
    mask = (1.0 - np.mean(mask_raw, axis=0)) > 0.5
    num_pixels = int(np.round(np.sum(mask)))
    if num_pixels < 1:
        num_pixels = 1
    orig_shape = gradcam_img.shape
    gradcam_img = gradcam_img.reshape(-1)
    gradcam_img[np.argsort(gradcam_img)[:-num_pixels]] = 0
    gradcam_img[np.argsort(gradcam_img)[-num_pixels:]] = 1
    gradcam_img = gradcam_img.reshape(orig_shape)
    gradcam_img = gradcam_img > 0.5
    num_intersection = np.sum(np.logical_and(gradcam_img, mask))
    num_union = np.sum(np.logical_or(gradcam_img, mask))
    ans = num_intersection / num_union
    if show_plots:
        fig, axes = plt.subplots(1, 2)
        fig.suptitle("JI: %.6f" % ans)
        axes[0].imshow(gradcam_img*1, cmap='gray')
        axes[1].imshow(mask*1, cmap='gray')
        plt.show()
    return ans

def explanable_cosine(gradcam_img, mask_raw):
    mask = 1.0 - np.mean(mask_raw, axis = 0)
    mask = mask.reshape(-1)
    gradcam_img = gradcam_img.reshape(-1)
    return (np.dot(mask, gradcam_img))/(np.linalg.norm(mask)*np.linalg.norm(gradcam_img))

def explanable_top_x_percent(gradcam_img, mask_raw, frac):
    mask = (1.0 - np.mean(mask_raw, axis=0))
    mask, gradcam_img = mask.reshape(-1), gradcam_img.reshape(-1)
    pixels_in_mask = int(np.round(np.sum(mask)))
    num_pixels = int(np.math.ceil(frac * len(gradcam_img)))
    return np.sum(mask[np.argsort(-gradcam_img)[:num_pixels]])/num_pixels

def explanable_top1p(gradcam_img, mask_raw):
    return explanable_top_x_percent(gradcam_img, mask_raw, 0.01)

def explanable_top5p(gradcam_img, mask_raw):
    return explanable_top_x_percent(gradcam_img, mask_raw, 0.05)

def explanable_top10p(gradcam_img, mask_raw):
    return explanable_top_x_percent(gradcam_img, mask_raw, 0.10)

def explanable_avgdst(gradcam_img, mask_raw):
    mask = (1.0 - np.mean(mask_raw, axis=0)) > 0.5
    colnums = np.array([[i for i in range(224)] for j in range(224)])
    rownums = np.array([[j]*224 for j in range(224)])
    msk_center = np.array([np.mean(mask * rownums), np.mean(mask*colnums)])
    img_center = np.array([np.mean(gradcam_img * rownums), np.mean(gradcam_img * colnums)])
    ans = np.linalg.norm(msk_center - img_center)
    return ans

def run_explanable(net, loaders, folderpath, explanable_functions, save_results = True):
    train_loader, validation_loader, test_loader = loaders
    folderpath = os.path.join(folderpath, "")
    layers=[net.features[-1]]
    a=0
    b=0
    explanable_scores = [[] for f in explanable_functions]
    for images, labels, masks, names in test_loader:
        images, labels, masks= images.cuda(), labels.cuda(), masks.cuda()
        cam=GradCAM(model=net, target_layers=layers,use_cuda=True)
        #cam=EigenCAM(model=net, target_layer=layer,use_cuda=True)
        
        grayscale_cam = cam(input_tensor=images, target_category=None)

        #grayscale_cam = grayscale_cam[0, :]
        images=images.detach().cpu().numpy()
        for i in range(len(images)):
            rgbimage=images[i].transpose((1,2,0))
            mean=np.array([0.485, 0.456, 0.406])
            std=np.array([0.22, 0.22, 0.22])
            rgbimage=(rgbimage*std)+mean
            if(np.isnan(np.sum(grayscale_cam[i]))):
                visualization = show_cam_on_image(rgbimage,np.zeros((224,224)),use_rgb=True)
            else:
                visualization = show_cam_on_image(rgbimage,grayscale_cam[i],use_rgb=True)
            tmpmask=masks[i].detach().cpu().numpy()
            if not np.isnan(np.sum(grayscale_cam[i])):
                for f_n,f in enumerate(explanable_functions):
                    score = f(grayscale_cam[i], tmpmask)
                    if not np.isnan(score):
                        explanable_scores[f_n].append(score)
            if save_results:
                matplotlib.image.imsave(folderpath+names[i]+'_input.jpg',rgbimage)
                matplotlib.image.imsave(folderpath+names[i]+'_result.jpg',visualization)
    for i,f in enumerate(explanable_functions):
        print("Average %s: %.6f" % (f.__name__, np.mean(explanable_scores[i])))