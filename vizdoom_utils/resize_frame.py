import cv2
import numpy as np
# import skimage.transform

def resize_cv_linear(img: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Resize an image to target resolution via bilinear interpolation

    Args:
        img (np.ndarray): source image
        resolution (tuple[int, int]): target resolution

    Returns:
        np.ndarray: resized image
    """
    return cv2.resize(img.transpose(1,2,0), resolution).transpose(2,0,1)

# This performance is unacceptable, remove support for skimage
# def resize_ski(img, resolution):
#     return (skimage.transform.resize(img, resolution)*256).astype('uint8')

def resize_cv_nearest(img: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Resize an image to target resolution via nearest neighbour interpolation

    Args:
        img (np.ndarray): source image
        resolution (tuple[int, int]): target resolution

    Returns:
        np.ndarray: resized image
    """
    return cv2.resize(img.transpose(1,2,0), resolution, interpolation=cv2.INTER_NEAREST).transpose(2,0,1)

def resize_cv_cubic(img: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Resize an image to target resolution via bicubic interpolation

    Args:
        img (np.ndarray): source image
        resolution (tuple[int, int]): target resolution

    Returns:
        np.ndarray: resized image
    """
    return cv2.resize(img.transpose(1,2,0), resolution, interpolation=cv2.INTER_CUBIC).transpose(2,0,1)