import cv2
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
def canny_enhance(input_image: str):
    image = cv2.imread(input_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(gray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)
    return imgCanny

def threshold_enhance(input_image: str):
    image = cv2.imread(input_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 115, 255, 0)  # 160
    return thresh