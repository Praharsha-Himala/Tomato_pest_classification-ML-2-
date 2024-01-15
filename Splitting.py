import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Specify the path to your CSV file
csv_file_path = r"D:\Users\HARSHU\Downloads\A database of eight common tomato pest images\A database of eight common tomato pest images\Tomato pest image enhancement\Tomato pest image enhancement\images_data.csv"

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path)
# print(df.columns)
# Replace labels in the DataFrame
label_mapping = {'BA': 0, 'HA': 1, 'SE': 2, 'MP': 3, 'TP': 4, 'SL': 5, 'ZC': 6, 'TU': 7}
df['pest'] = df['pest'].replace(label_mapping)

# print(df)
# Display the first few rows of the DataFrame after replacing labels
print(df.head())
#
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
print(f"Updated Label: {label}")
plt.imshow(image_array, cmap='gray')
plt.show()
