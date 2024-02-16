import os
import random
from pathlib import Path

def delete_random_images(folder_path, target_count):
    # Get a list of subfolders in the main folder
    subfolders = [subfolder for subfolder in Path(folder_path).iterdir() if subfolder.is_dir()]

    for subfolder in subfolders:
        # Get the list of images in the subfolder
        images = list(subfolder.glob('*.jpg'))  # You may need to adjust the file extension

        # Check if the subfolder already has the target count of images
        if len(images) <= target_count:
            print(f"No deletion needed in {subfolder}")
        else:
            # Calculate the number of images to delete
            images_to_delete = len(images) - target_count

            # Randomly select images to delete
            images_to_delete_list = random.sample(images, images_to_delete)

            # Delete the selected images
            for image in images_to_delete_list:
                image.unlink()
                print(f"Deleted {image} from {subfolder}")

# Specify the path to your main folder and the target count of images
main_folder_path = r"D:\Users\HARSHU\Downloads\A database of eight common tomato pest images\A database of eight common tomato pest images\Tomato pest image enhancement\Tomato pest image enhancement\images"
target_image_count = 168

# Call the function
delete_random_images(main_folder_path, target_image_count)
