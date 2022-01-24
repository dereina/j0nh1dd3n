# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot

# load image as pixel array
image1 = image.imread('images/visumhsi_10_1635154646.png')
# summarize shape of the pixel array
print(image1.dtype)
print(image1.shape)
# display the array of pixels as an image
pyplot.imshow(image1)
pyplot.show()


# load and show an image with Pillow
from PIL import Image
from numpy import asarray
import numpy as np
# Open the image form working directory
image2 = Image.open('images/visumhsi_10_1635154646.png').convert('RGB')
# summarize some details about the image
image3 = asarray(image2)
print(type(image3))
image3 = np.array(image2)
print(type(image3))

print(image2.format)
print(image2.size)
print(image2.mode)
# show the image
image2.show()


import cv2

im = cv2.imread('images/visumhsi_10_1635154646.png')
img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)   # BGR -> RGB
cv2.imwrite('opncv_kolala.png', img) 
print (type(img))