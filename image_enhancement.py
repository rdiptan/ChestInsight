import cv2
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow, show, subplot, title, get_cmap, hist
def clahe_image_enhance(input_image:str):
    # dimensions of input images
    image = cv2.imread(input_image)
    widthImg = image.shape[0]
    heightImg = image.shape[1]
    scale = max([widthImg, heightImg])

    # resizing image
    #image = cv2.resize(image, (scale, scale))

    # image processing (contrast limited adaptive histogram equalization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Histogram Equalization
    equalized = clahe.apply(gray)

    return equalized

def increase_brightness(input_image: str ,value):
    image = cv2.imread(input_image)
    value = value
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img