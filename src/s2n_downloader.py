import os
import tarfile
from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from sentinelhub import SHConfig, BBox, CRS, DataCollection, SentinelHubRequest, MimeType
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# Define your bands of interest
bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "SCL", "B11", "B12"]

# Define the desired image size
start_date = '2023-04-01'
end_date = '2023-06-30'
image_size = (512, 512)
cloud_cover_max = 0.10
output_folder = 'croptype'

# Define your Sentinel Hub credentials
config = SHConfig()
config.sh_client_id = 'your client_id'
config.sh_client_secret = 'your client_secret'

# Set the configuration to allow downloading of data
# Define your evalscript
evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
        output: [
            { id: "B01", bands: 1, sampleType: SampleType.AUTO },
            { id: "B02", bands: 1, sampleType: SampleType.AUTO },
            { id: "B03", bands: 1, sampleType: SampleType.AUTO },
            { id: "B04", bands: 1, sampleType: SampleType.AUTO },
            { id: "B05", bands: 1, sampleType: SampleType.AUTO },
            { id: "B06", bands: 1, sampleType: SampleType.AUTO },
            { id: "B07", bands: 1, sampleType: SampleType.AUTO },
            { id: "B08", bands: 1, sampleType: SampleType.AUTO },
            { id: "B8A", bands: 1, sampleType: SampleType.AUTO },
            { id: "B09", bands: 1, sampleType: SampleType.AUTO },
            { id: "B11", bands: 1, sampleType: SampleType.AUTO },
            { id: "B12", bands: 1, sampleType: SampleType.AUTO },
            { id: "RGB", bands: 3, sampleType: SampleType.AUTO },
            { id: "RGBN", bands: 4, sampleType: SampleType.AUTO },
            { id: "TCI", bands: 3, sampleType: SampleType.AUTO },
            { id: "NDVI", bands: 1, sampleType: SampleType.FLOAT32 },  // NDVI band
            { id: "SAVI", bands: 3, sampleType: SampleType.FLOAT32 } 
           
        ]
    };
}

function evaluatePixel(samples, scenes, inputMetadata, customData, outputMetadata) {
    ndvi = (samples.B08 - samples.B04) / (samples.B08 + samples.B04);
    
    // Calculate SAVI
    L = 0.5;  // Soil brightness correction factor (adjust as needed)
    savi = ((samples.B08 - samples.B04) / (samples.B08 + samples.B04 + L)) * (1 + L);

    return {
        B01: [samples.B01],
        B02: [samples.B02],
        B03: [samples.B03],
        B04: [samples.B04],
        B05: [samples.B05],
        B06: [samples.B06],
        B07: [samples.B07],
        B08: [samples.B08],
        B8A: [samples.B8A],
        B09: [samples.B09],
        B11: [samples.B11],
        B12: [samples.B12],
        RGB: [2.5*samples.B04, 2.5*samples.B03, 2.5*samples.B02],
        RGBN: [samples.B04, samples.B03, samples.B02, samples.B08],
        TCI: [3*samples.B04, 3*samples.B03, 3*samples.B02],
        NDVI: [ndvi],  
        SAVI: [savi] 
    };
}
"""

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths, crop_types):
        self.image_paths = image_paths
        self.crop_types = crop_types

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        crop_type = self.crop_types[idx]
        return image, crop_type


# Define the dataset root and masks root directories
dataset_root = 'crop_type_dataset'  # Replace with the path to your dataset
masks_root = 'crop_masks-'  # Directory to save the masks
cloud_cover_threshold = 80  # Define your cloud cover threshold (adjust as needed)

# Create the masks directory if it doesn't exist
os.makedirs(masks_root, exist_ok=True)

# Define a function to create a mask from a TCI image using a percentage threshold and add boundary thickness
def create_mask_from_tci_with_boundary_thickness(tci_image, mask_dir, threshold_percentage, boundary_thickness):
    # Convert the TCI image to grayscale
    grayscale_image = transforms.functional.to_grayscale(tci_image, num_output_channels=1)

    # Calculate the threshold value as a percentage of the maximum pixel intensity
    max_pixel_intensity = 255  # Maximum pixel intensity for 8-bit grayscale
    threshold_value = (threshold_percentage / 100) * max_pixel_intensity

    # Threshold the grayscale image to create a binary mask
    mask = transforms.functional.to_tensor(grayscale_image)
    mask = (mask > (threshold_value / 255.0)).float() * 255.0

    # Apply boundary thickness using dilation
    mask = Image.fromarray(mask.squeeze().numpy().astype(np.uint8))
    mask = mask.filter(ImageFilter.MaxFilter(boundary_thickness))

    # Convert the grayscale mask to an RGB mask with the same number of output channels as input channels
    mask = transforms.ToTensor()(mask)
    mask = mask.repeat(3, 1, 1)  # Convert grayscale to RGB format

    # Save the mask with a counter to avoid overwriting
    mask_counter = 0
    while True:
        mask_filename = f"{mask_counter:02d}_mask.tif"
        mask_path = os.path.join(mask_dir, mask_filename)
        if not os.path.exists(mask_path):
            transforms.functional.to_pil_image(mask).save(mask_path)
            break
        mask_counter += 1

# Define the threshold percentage (e.g., 50%) and boundary thickness
threshold_percentage = 50  # Adjust this threshold percentage as needed
boundary_thickness = 5  # Adjust the thickness as needed

# Load the dataset
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to a fixed size (adjust as needed)
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
])

# Load the dataset
dataset = datasets.ImageFolder(root=dataset_root, transform=data_transform)

# Dictionary to store the number of images before and after filtering
num_images_before_filtering = {}
num_images_after_filtering = {}

# Function to download and process Sentinel-2 images
def download_and_process_images(lat, lon, start_date, end_date, output_folder, max_images=4):
    bbox = BBox([lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02], CRS.WGS84)

    request = SentinelHubRequest(
        data_folder=output_folder,
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=(start_date, end_date),
                maxcc=cloud_cover_max
            )
        ],
        responses=[
            SentinelHubRequest.output_response('RGB', MimeType.TIFF),
            SentinelHubRequest.output_response('TCI', MimeType.TIFF),
            SentinelHubRequest.output_response('RGB', MimeType.TIFF),
            SentinelHubRequest.output_response('RGBN', MimeType.TIFF),
            SentinelHubRequest.output_response('TCI', MimeType.TIFF),
            SentinelHubRequest.output_response('B02', MimeType.TIFF),
            SentinelHubRequest.output_response('B03', MimeType.TIFF),
            SentinelHubRequest.output_response('B04', MimeType.TIFF),
            SentinelHubRequest.output_response('NDVI', MimeType.TIFF),
            SentinelHubRequest.output_response('SAVI', MimeType.TIFF),
        ],
        bbox=bbox,
        size=image_size,
        config=config,
    )

    # Save data for up to max_images images
    image_count = 0
    for response in request.get_data():
        if image_count >= max_images:
            break
        image_count += 1

    request.save_data()

    # Extract tar files and organize images into crop type subfolders
    for root, dirs, files in os.walk(output_folder):
        for file in files:
            if file.endswith(".tar"):
                tar_path = os.path.join(root, file)
                crop_type = os.path.basename(os.path.dirname(tar_path))

                # Create the crop type subfolder if it doesn't exist
                crop_type_folder = os.path.join(output_folder, crop_type)
                os.makedirs(crop_type_folder, exist_ok=True)

                # Extract the tar file into the crop type subfolder
                with tarfile.open(tar_path, "r") as tar:
                    tar.extractall(crop_type_folder)

                # Remove the original tar file
                os.remove(tar_path)

                # Filter TCI images based on pixel intensity
                tci_image_path = os.path.join(crop_type_folder, "TCI.tif")
                if os.path.exists(tci_image_path):
                    tci_image = Image.open(tci_image_path)
                    blue_channel = np.array(tci_image)[:, :, 2]  # Blue channel is at index 2
                    mean_blue_intensity = np.mean(blue_channel)
                    if mean_blue_intensity <= cloud_cover_threshold:
                        num_images_after_filtering[crop_type] = num_images_after_filtering.get(crop_type, 0) + 1
                    num_images_before_filtering[crop_type] = num_images_before_filtering.get(crop_type, 0) + 1

                    if mean_blue_intensity <= cloud_cover_threshold:
                        # Create an RGB mask with boundary thickness using the specified threshold percentage
                        mask_dir = os.path.join(masks_root, crop_type)
                        os.makedirs(mask_dir, exist_ok=True)
                        create_mask_from_tci_with_boundary_thickness(
                            tci_image, mask_dir, threshold_percentage, boundary_thickness
                        )

                        # Save the filtered TCI image to the crop type folder
                        filtered_tci_image_path = os.path.join(crop_type_folder, "Filtered_TCI.tif")
                        tci_image.save(filtered_tci_image_path)

# Read the input CSV with latitudes, longitudes, and crop names
input_csv = 'imagery_source/sample/sample.csv'  # Replace with your CSV file
df = pd.read_csv(input_csv)

# Load and process Sentinel-2 images for each location in the CSV
for _, row in df.iterrows():
    lat, lon = row['y_coord'], row['x_coords']
    crop_type = row['Crop name']

    # Define output folder for each location
    crop_output_folder = os.path.join(dataset_root, str(crop_type))  # Convert to string
    os.makedirs(crop_output_folder, exist_ok=True)

    # Download and process Sentinel-2 images and assign crop type names
    download_and_process_images(lat, lon, start_date, end_date, crop_output_folder, max_images=4)

print("Number of Images Before Filtering:")
for crop_type, count in num_images_before_filtering.items():
    print(f"{crop_type}: {count}")

print("\nNumber of Images After Filtering:")
for crop_type, count in num_images_after_filtering.items():
    print(f"{crop_type}: {count}")

print("Dataset creation completed.")
