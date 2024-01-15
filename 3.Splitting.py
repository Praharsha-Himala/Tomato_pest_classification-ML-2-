import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Specify the path to your CSV file
csv_file_path = r"D:\Users\HARSHU\Downloads\A database of eight common tomato pest images\A database of eight common tomato pest images\Tomato pest image enhancement\Tomato pest image enhancement\images_data.csv"

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path)
# #########################################################################
# Splitting the data into test and train (80:20)

X = df['pixels'].values
y = df['pest'].values
X = np.array([np.fromstring(pixel, dtype=int, sep=' ') for pixel in X]) # making sure every value is numeric
# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
# Save X_train, X_test, y_train, y_test as .npy files
np.save("D:\Projects\Tomato_pest_classification/X_train.npy", X_train)
np.save("D:\Projects\Tomato_pest_classification/X_test.npy", X_test)
np.save("D:\Projects\Tomato_pest_classification/y_train.npy", y_train)
np.save("D:\Projects\Tomato_pest_classification/y_test.npy", y_test)

# loading the numpy files
xtrain = np.load("D:\Projects\Tomato_pest_classification/X_train.npy")
xtest = np.load("D:\Projects\Tomato_pest_classification/X_test.npy")
ytrain = np.load("D:\Projects\Tomato_pest_classification/y_train.npy")
ytest = np.load("D:\Projects\Tomato_pest_classification/y_test.npy")

# Choose an index for the image you want to plot
index = 0

# Extract the image and label
image = xtrain[index]
label = ytrain[index]

# Assuming your images are square
image_size = int(np.sqrt(image.shape[0]))

# Reshape the flattened image to its original shape
image = image.reshape((image_size, image_size))

# Plot the image
plt.imshow(image, cmap='gray')
plt.title(f'Label: {label}')
plt.show()