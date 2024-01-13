'''
The code is inspired from the work of Dr.Alwin Poulose on Facial Emotion Recognition
'''

from PIL import Image
import numpy as np
import sys
import os
import csv

base_directory = r"D:\Users\HARSHU\Downloads\A database of eight common tomato pest images\A database of eight common tomato pest images\Tomato pest image enhancement\Tomato pest image enhancement//"

folder_name = "images"
file_name = "images_data.csv"
# cifar_labels = ['airplane', 'automobile', 'bird' ,'cat' ,'deer' ,'dog' ,'frog' ,'horse' ,'ship' ,'truck']
label_names = [] #change this into our 10 class names
# label_names = cifar_labels
width, height = 299, 299 #pixel value


# default format can be changed as needed
def createFileList(myDir, formats=('.jpg')):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(formats):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList


# load the original image
myFileList = createFileList(base_directory + folder_name)

if (folder_name == "Training" or folder_name == "PrivateTest"):
    with open(base_directory + file_name, "a") as f:
        f.write(f'emotion,pixels,Usage')
        f.write("\n")
count = 0
for file in myFileList:
    print(file)
    img_file = Image.open(file)
    count += 1
    category = os.path.basename(os.path.dirname(file))
    # img_file.show()

    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')
    # img_grey.save('result.png')
    # img_grey.show()

    # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=np.int32).reshape((img_grey.size[1], img_grey.size[0]))
    value = value.flatten()
    print(value)
    # value = str(value).lstrip('[').rstrip(']')
    with open(base_directory + file_name, "a") as f:
        f.write(
            f'{category},{" ".join([str(pixel) for pixel in value.reshape(width * height)])},{folder_name}')
        f.write("\n")
print(f'number of pixalized count: {count}')
