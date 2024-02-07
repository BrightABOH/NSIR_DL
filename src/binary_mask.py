import os
import cv2
import numpy as np

# Define input and output directories
input_dir = 'crop_type_images_3'  # Directory containing TCI images in subfolders
output_dir = 'try'  # Directory to save filtered images
mask_output_dir = 'binary_mask_2_thick'  # Directory to save binary masks
os.makedirs(output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)  # Create the mask output directory

# Define the cloud cover threshold (adjust as needed)
cloud_cover_threshold = 0.5  # Experiment with different threshold values

# Function to calculate cloud cover
def calculate_cloud_cover(binary_mask):
    # Calculate the cloud cover as the ratio of white pixels to total pixels
    total_pixels = binary_mask.size
    white_pixels = np.count_nonzero(binary_mask)
    cloud_cover = white_pixels / total_pixels
    return cloud_cover

# Function to remove images with high cloud cover, create binary masks, and save them
def process_image(image_path, output_dir, mask_output_dir, cloud_cover_threshold):
    # Load the TCI image
    tci_image = cv2.imread(image_path)

    if tci_image is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    # Convert the TCI image to grayscale
    gray_tci = cv2.cvtColor(tci_image, cv2.COLOR_BGR2GRAY)

    # Define a cloud threshold (adjust as needed)
    cloud_threshold = 100  # Experiment with different threshold values

    # Create a binary mask where clouds are white and the rest is black
    binary_mask = cv2.adaptiveThreshold(gray_tci, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,  # Block size (size of the neighborhood)
            20)

    # Calculate cloud cover
    cloud_cover = calculate_cloud_cover(binary_mask)

    # Determine the relative path of the image within the input directory
    relative_path = os.path.relpath(image_path, input_dir)

    # If cloud cover is below the threshold, save the binary mask and filtered image
    if cloud_cover <= cloud_cover_threshold:
        # Create the corresponding output directory structure
        output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
        os.makedirs(output_subdir, exist_ok=True)

        mask_output_subdir = os.path.join(mask_output_dir, os.path.dirname(relative_path))
        os.makedirs(mask_output_subdir, exist_ok=True)

        # Define the output path for the binary mask
        binary_mask_path = os.path.join(mask_output_subdir, os.path.basename(image_path))

        # Save the binary mask
        cv2.imwrite(binary_mask_path, binary_mask)

        # Define the output path for the filtered image
        filtered_image_path = os.path.join(output_subdir, os.path.basename(image_path))

        # Save the filtered image
        cv2.imwrite(filtered_image_path, tci_image)

        print(f"Processed: {image_path} (Cloud Cover: {cloud_cover})")
    else:
        print(f"Removed (High Cloud Cover): {image_path} (Cloud Cover: {cloud_cover})")

# Process TCI images in subfolders recursively
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.tif'):
            image_path = os.path.join(root, file)
            process_image(image_path, output_dir, mask_output_dir, cloud_cover_threshold)

