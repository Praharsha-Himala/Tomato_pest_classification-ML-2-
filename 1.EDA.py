'''
The tomato pest classification dataset is explored using
1. Distribution of images in each class - determined if the dataset is imbalanced
2. Distribution of average RBG intensities in each class
3. Distribution of sizes of images in each class
'''
#importing necessary libraries and packages to perform EDA
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as snsrff
import pandas as pd
#################################################################################################################
# 1. counting number of images in each class
def count_jpg_images_in_subfolders(root_folder):
    jpg_folder_counts = defaultdict(int)

    for foldername, subfolders, filenames in os.walk(root_folder):
        jpg_image_count = sum(1 for filename in filenames if filename.lower().endswith('.jpg'))
        jpg_folder_counts[foldername] += jpg_image_count

    return jpg_folder_counts
# Plotting the count of images
def plot_bar_chart(jpg_image_counts):
    subfolders = [subfolder for foldernames, subfolder, filenames in os.walk(root_folder)][0]
    counts = list(jpg_image_counts.values())[1:]  # Exclude count for root folder

    plt.bar(subfolders, counts, color='dodgerblue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    # plt.title('Original Dataset Distribution')

    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
#################################################################################################################
# 2. Calculating average RGB intensity of each class
def calculate_average_rgb_intensity(image_path):
    image = Image.open(image_path)
    rgb_array = np.array(image)
    average_intensity = np.mean(rgb_array, axis=(0, 1))
    return average_intensity

def calculate_average_intensity_per_class(root_folder):
    class_intensities = {}

    for class_folder in os.listdir(root_folder):
        class_path = os.path.join(root_folder, class_folder)
        if os.path.isdir(class_path):
            red_sum, green_sum, blue_sum = 0, 0, 0
            image_count = 0

            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_path = os.path.join(class_path, image_file)
                    intensity = calculate_average_rgb_intensity(image_path)
                    red_sum += intensity[0]
                    green_sum += intensity[1]
                    blue_sum += intensity[2]
                    image_count += 1

            if image_count > 0:
                average_red = red_sum / image_count
                average_green = green_sum / image_count
                average_blue = blue_sum / image_count

                class_intensities[class_folder] = {
                    'average_red': average_red,
                    'average_green': average_green,
                    'average_blue': average_blue
                }

    return class_intensities

def plot_intensity_bar_chart(class_intensities):
    classes = list(class_intensities.keys())
    average_reds = [class_intensities[class_]['average_red'] for class_ in classes]
    average_greens = [class_intensities[class_]['average_green'] for class_ in classes]
    average_blues = [class_intensities[class_]['average_blue'] for class_ in classes]

    bar_width = 0.2
    index = np.arange(len(classes))

    plt.bar(index, average_reds, width=bar_width, label='Average Red', align='center', color='red')
    plt.bar(index + bar_width, average_greens, width=bar_width, label='Average Green', align='center', color='green')
    plt.bar(index + 2 * bar_width, average_blues, width=bar_width, label='Average Blue', align='center', color='blue')

    plt.xlabel('Classes')
    plt.ylabel('Intensity')
    # plt.title('Average RGB Intensity Distribution')
    plt.xticks(index + bar_width, classes, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()
#################################################################################################################
# 3. Size distribution of the dataset
def get_image_sizes(folder_path):
    image_sizes = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(folder_path, filename)
            size = os.path.getsize(image_path)
            image_sizes.append(size)

    return image_sizes

def create_box_plot(root_folder):
    image_sizes_per_class = []

    for class_folder in os.listdir(root_folder):
        class_path = os.path.join(root_folder, class_folder)
        if os.path.isdir(class_path):
            sizes = get_image_sizes(class_path)
            class_label = class_folder  # You may adjust this depending on your class labeling strategy
            image_sizes_per_class.extend([(size, class_label) for size in sizes])

    plt.figure(figsize=(12, 7))

    ax = sns.boxplot(x="Size", y="Class", data=pd.DataFrame(image_sizes_per_class, columns=["Size", "Class"]), orient="h")

    # plt.title("Distribution of Image Sizes for Each Class")
    plt.xlabel("Size(bytes)")
    plt.ylabel("Class")

    plt.show()

#################################################################################################################
root_folder = r"D:\Users\HARSHU\Downloads\A database of eight common tomato pest images\A database of eight common tomato pest images\Tomato pest image enhancement\Tomato pest image enhancement"

# Plotting dataset distribution
jpg_image_counts = count_jpg_images_in_subfolders(root_folder)
for folder, count in jpg_image_counts.items():
    print(f"Folder: {folder}, JPG Image Count: {count}")
plot_bar_chart(jpg_image_counts)

# Average intensities of RGB channels for each class in original dataset
class_intensities = calculate_average_intensity_per_class(root_folder)
for class_name, intensities in class_intensities.items():
    print(f"Class: {class_name}, Average Red: {intensities['average_red']:.2f}, Average Green: {intensities['average_green']:.2f}, Average Blue: {intensities['average_blue']:.2f}")
plot_intensity_bar_chart(class_intensities)

# Box plots of images sizes of each class
create_box_plot(root_folder)