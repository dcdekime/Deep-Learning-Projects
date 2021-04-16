import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torchvision import datasets
from PIL import Image
from glob import glob
from tqdm import tqdm


# load filenames for human and dog images
human_files = np.array(glob("./data/lfw/*/*"))
dog_files = np.array(glob("./data/dogImages/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x, y, w, h) in faces:
    # add bounding box to color image
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
#plt.imshow(cv_rgb)
#plt.show()

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

#-#-# Do NOT modify the code above this line. #-#-#

## TODO: Test the performance of the face_detector algorithm
## on the images in human_files_short and dog_files_short.
# humanDetection = tqdm(total=len(human_files_short), desc='Human Faces Detected in Human Files', position=0)
# dogDetection = tqdm(total=len(dog_files_short), desc='Human Faces Detected in Dog Files', position=1)
#
# for file in human_files_short:
#     if face_detector(file):
#         humanDetection.update(1)
#
# for file in dog_files_short:
#     if face_detector(file):
#         dogDetection.update(1)


# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()


def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to
    predicted ImageNet class for image at specified path

    Args:
        img_path: path to an image

    Returns:
        Index corresponding to VGG-16 model's prediction
    '''

    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image

    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.ToTensor()])
    img = Image.open(img_path)
    img = data_transform(img)
    img = img.unsqueeze(0)
    output = VGG16(img)
    imageInd = int(torch.argmax(output))

    return imageInd  # predicted class index


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    index = VGG16_predict(img_path)

    return (index >= 151 and index <= 268) # true/false


### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
humanDetection = tqdm(total=len(human_files_short), desc='Human Faces Detected in Dog Detector', position=0)
dogDetection = tqdm(total=len(dog_files_short), desc='Dog Faces Detected in Dog Detector', position=1)

for file in human_files_short:
    if dog_detector(file):
        humanDetection.update(1)

for file in dog_files_short:
    if dog_detector(file):
        dogDetection.update(1)


if __name__ == 'main':
    pass