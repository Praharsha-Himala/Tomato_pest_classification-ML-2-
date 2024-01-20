'''
The code is inspired from the work of Dr.Alwin Poulose on Facial Emotion Recognition
'''

from PIL import Image
import numpy as np
import sys
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
base_directory = r"D:\Users\HARSHU\Downloads\A database of eight common tomato pest images\A database of eight common tomato pest images\Tomato pest image enhancement\Tomato pest image enhancement//"

folder_name = "images"
file_name = "images_data1.csv"
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

if (folder_name == 'images'):
    with open(base_directory + file_name, "a") as f:
        f.write(f'pest,pixels')
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
            f'{category},{" ".join([str(pixel) for pixel in value.reshape(width * height)])}')
        f.write("\n")
print(f'number of pixalized count: {count}')
#########################################################################################
# labels are changed to numeric type
csv_file_path = r"D:\Users\HARSHU\Downloads\A database of eight common tomato pest images\A database of eight common tomato pest images\Tomato pest image enhancement\Tomato pest image enhancement\images_data1.csv"

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path)
label_encoder = LabelEncoder()
df['pest'] = label_encoder.fit_transform(df['pest'])

# print(df)
# Display the first few rows of the DataFrame after replacing labels
print(df.head())
#
# Assuming 'df' is your DataFrame with updated labels
csv_file_path_updated = r"D:\Users\HARSHU\Downloads\A database of eight common tomato pest images\A database of eight common tomato pest images\Tomato pest image enhancement\Tomato pest image enhancement\images_data_updated.csv"

# Save the DataFrame to a new CSV file
df.to_csv(csv_file_path_updated, index=False)

# Extract label and pixel values for a specific image (change the index as needed)
index_to_load = 0
label = df.loc[index_to_load, 'pest']
pixels_str = df.loc[index_to_load, 'pixels']

# Convert the string representation of pixels to a list of integers
pixels = np.array(list(map(int, pixels_str.split())))

# Reshape the 1D array back to a 2D array (assuming a square image)
image_size = int(np.sqrt(len(pixels)))
image_array = pixels.reshape((image_size, image_size))

# Display the updated label and plot the image
plt.imshow(image_array, cmap='gray', aspect='auto')
plt.title(f"Updated Label: {label}")
plt.show()
