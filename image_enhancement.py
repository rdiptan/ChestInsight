import cv2
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow, show, subplot, title, get_cmap, hist
def clahe_image_enhance(input_image:str, mode='HE'):
    """
    Enhances the input image using Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        input_image (str): Path to the input image file.

    Returns:
        numpy.ndarray: The enhanced image.
    """
    # Read the input image
    image = cv2.imread(input_image)
    
    # Get the dimensions of the input image
    widthImg = image.shape[0]
    heightImg = image.shape[1]
    scale = max([widthImg, heightImg])

    # resizing image
    #image = cv2.resize(image, (scale, scale))

    # image processing (contrast limited adaptive histogram equalization
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mode=='HE':
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
    else:
        # Create a Contrast Limited Adaptive Histogram Equalization (CLAHE) object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Apply CLAHE to the equalized image
        equalized = clahe.apply(gray)

    return equalized

def increase_brightness(input_image: str ,value):
    """
    Increases the brightness of the input image.

    Args:
        input_image (str): Path to the input image file.
        value (int): The amount by which to increase the brightness.

    Returns:
        numpy.ndarray: The image with increased brightness.
    """
    # Read the input image
    image = cv2.imread(input_image)
    
    # Set the brightness value
    value = value
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Increase the brightness of the image
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    # Merge the modified channels back into the HSV image
    final_hsv = cv2.merge((h, s, v))
    
    # Convert the HSV image back to BGR color space
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img