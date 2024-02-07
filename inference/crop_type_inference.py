import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os
import torch.nn.functional as F

# Function to get raw model output based on an input image
def get_raw_output(model, input_image_path):
    # Load and pre-process the input image
    input_image = Image.open(input_image_path).convert('RGB')
    input_image = transforms.Resize((config["input_shape"][0], config["input_shape"][1]))(input_image)
    input_tensor = transforms.ToTensor()(input_image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)

    # Get raw output from the model
    with torch.no_grad():
        model.eval()
        raw_output = model(input_tensor)

    return raw_output, input_tensor

# Load the configuration from the JSON file
config_file = "configs/crop_type_config.json"
with open(config_file, "r") as f:
    config = json.load(f)

# Define input and output directories with the absolute path
image_root_dir = config["image_root_dir"]

# Load the class_to_index mapping
class_to_index = {class_name: i for i, class_name in enumerate(sorted(os.listdir(image_root_dir)))}
print("Class to Index Mapping:", class_to_index)

# Load the trained model
model_path = 'models/classification_model_resnet50.pth'
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
num_classes = len(class_to_index)
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Update the pixel-to-hectares conversion factor based on Sentinel-2 10m resolution
pixel_resolution = 10  # meters
pixel_to_hectares_conversion_factor = (pixel_resolution / 1000) ** 2  # Convert square meters to hectares

def predict_crop_types(image_path):
    # Get raw model output and input tensor
    raw_output, input_tensor = get_raw_output(model, image_path)

    # Apply softmax
    probabilities = F.softmax(raw_output, dim=1)

    # Get the class indices with the highest probabilities
    _, predicted_class_indices = torch.topk(probabilities, k=3)  # Top-3 predicted classes

    # Print the raw output, probabilities, and predicted class indices
    print("Raw Model Output:", raw_output)
    print("Class Probabilities:", probabilities)
    print("Predicted Class Indices:", predicted_class_indices)

    # Map predicted class indices to class names
    predicted_class_names = [class_name for index in predicted_class_indices[0] for class_name, class_index in class_to_index.items() if index == class_index]

    # Print the mapped class names
    print("Predicted Class Names:", predicted_class_names)

    return {predicted_class_names[0]: probabilities[0, predicted_class_indices[0, 0]].item(),
            predicted_class_names[1]: probabilities[0, predicted_class_indices[0, 1]].item(),
            predicted_class_names[2]: probabilities[0, predicted_class_indices[0, 2]].item()}

# Example usage
image_path = "/Users/brightabohsilasedem/Desktop/NSIR_Project/crop_type_dataset2/Bush beens/42819f6c901c1934f430c488fefeb451/RGB.tif"
predicted_areas = predict_crop_types(image_path)
print("Predicted Crop Types and Areas:")
for crop_type, area in predicted_areas.items():
    print(f"{crop_type}: {area} hectares")
