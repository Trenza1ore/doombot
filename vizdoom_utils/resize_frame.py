import cv2
import skimage.transform

def resize_cv_linear(img, resolution):
    return cv2.resize(img.transpose(1,2,0), resolution).transpose(2,0,1)

def resize_ski(img, resolution):
    return (skimage.transform.resize(img, resolution)*256).astype('uint8')

def resize_cv_nearest(img, resolution):
    return cv2.resize(img.transpose(1,2,0), resolution, interpolation=cv2.INTER_NEAREST).transpose(2,0,1)

def resize_cv_cubic(img, resolution):
    return cv2.resize(img.transpose(1,2,0), resolution, interpolation=cv2.INTER_CUBIC).transpose(2,0,1)