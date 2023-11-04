import cv2
import numpy as np
import math
from PIL import Image
from matplotlib.pyplot import imshow, show, subplot, title, get_cmap, hist
import torch
import torchvision.transforms as transforms
import monai.transforms as monai_transforms


def clahe_image_enhance(input_image: str, mode="HE"):
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
    # image = cv2.resize(image, (scale, scale))

    # image processing (contrast limited adaptive histogram equalization
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mode == "HE":
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
    else:
        # Create a Contrast Limited Adaptive Histogram Equalization (CLAHE) object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Apply CLAHE to the equalized image
        equalized = clahe.apply(gray)

    return equalized


def increase_brightness(input_image: str, value):
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


def gamma(input_image: str):
    # dimensions of input images
    image = cv2.imread(input_image)
    # first method
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(val)
    gamma = math.log(mid * 255) / math.log(mean)
    # print(gamma)

    # do gamma correction on value channel
    val_gamma = np.power(val, gamma).clip(0, 255).astype(np.uint8)

    # combine new value channel with original hue and sat channels
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    img_gamma1 = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)
    # 2nd method
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(gray)
    gamma = math.log(mid * 255) / math.log(mean)
    # print(gamma)

    # do gamma correction
    img_gamma2 = np.power(image, gamma).clip(0, 255).astype(np.uint8)

    return img_gamma2


# Function to perform Super-Resolution
def super_resolution(image_path, upscale_factor=4):
    # Load the image
    image = Image.open(image_path).convert("RGB")  # Convert to RGB
    image = transforms.ToTensor()(image).unsqueeze(
        0
    )  # Convert to tensor and add batch dimension

    # Upscale the image using interpolation
    super_res_image = torch.nn.functional.interpolate(
        image, scale_factor=upscale_factor, mode="bicubic", align_corners=False
    )

    # Convert the enhanced image tensor back to a PIL image
    enhanced_image_pil = transforms.ToPILImage()(super_res_image.squeeze(0))
    return enhanced_image_pil


# Function to perform Noise Reduction using Gaussian Smoothing
def noise_reduction(image_path, sigma=1.0):
    # Load the image
    image = Image.open(image_path).convert("RGB")  # Convert to RGB

    # Convert the Pillow image to a NumPy array
    image_array = np.array(image)

    # Apply Gaussian Smoothing to reduce noise
    smoothed_image_array = monai_transforms.GaussianSmooth(sigma=sigma)(image_array)

    # Convert the smoothed NumPy array back to a Pillow image
    smoothed_image_pil = Image.fromarray(smoothed_image_array.astype(np.uint8))
    return smoothed_image_pil
