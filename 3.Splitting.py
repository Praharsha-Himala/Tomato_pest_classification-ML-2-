import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Specify the path to your CSV file
csv_file_path = r"csv_file_path\images_data_updated.csv"

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path)
# #########################################################################
# Splitting the data into test and train (80:20)
print(df.head())
# Convert the 'pixels' column to a NumPy array of integers
# Assuming 'df' is your DataFrame
df['pixels'] = df['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))

# Find the maximum length of 'pixels' and pad shorter arrays with zeros
max_length = df['pixels'].apply(len).max()
df['pixels'] = df['pixels'].apply(lambda x: np.pad(x, (0, max_length - len(x))))

# Create a 2D NumPy array from the 'pixels' column
X = np.vstack(df['pixels'].to_numpy())
y = df['pest'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Save X_train, X_test, y_train, y_test as .npy files
np.save("D:\Projects\Tomato_pest_classification/X_train.npy", X_train)
np.save("D:\Projects\Tomato_pest_classification/X_test.npy", X_test)
np.save("D:\Projects\Tomato_pest_classification/y_train.npy", y_train)
np.save("D:\Projects\Tomato_pest_classification/y_test.npy", y_test)

# # loading the numpy files
xtrain = np.load("D:\Projects\Tomato_pest_classification/X_train.npy")
xtest = np.load("D:\Projects\Tomato_pest_classification/X_test.npy")
ytrain = np.load("D:\Projects\Tomato_pest_classification/y_train.npy", allow_pickle=True)
ytest = np.load("D:\Projects\Tomato_pest_classification/y_test.npy", allow_pickle=True)

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
print(xtrain.shape)
