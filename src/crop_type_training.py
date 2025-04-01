import os
import json
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import argparse
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from PIL import Image
from tqdm import tqdm
import PIL
import mlflow

# Define the command-line argument parser
parser = argparse.ArgumentParser(description='Crop Type Classification')
parser.add_argument('--input_type', type=str, choices=['NDVI','SAVI', 'TCI', 'RGB', 'TCI+NDVI'], default='TCI+NDVI',
                    help='Type of input data for the model (NDVI, TCI, RGB)')
parser.add_argument('--model_type', type=str, choices=['resnet34', 'resnet50', 'vgg19','inception_v3','alexnet'], default='resnet50',
                    help='Type of pre-trained model architecture to use (default: vgg19)')
args = parser.parse_args()

# Extract the model type
model_type = args.model_type

# Load the configuration from the JSON file
config_file = "configs/crop_type_config.json"
with open(config_file, "r") as f:
    config = json.load(f)

# Parameter tuning
LEARNING_RATE = config["LEARNING_RATE"]
NUM_EPOCHS = config["NUM_EPOCHS"]
input_shape = config["input_shape"]
batch_size = config["batch_size"]

# Define input and output directories with the absolute path
image_root_dir = config["image_root_dir"]
mask_root_dir = config["mask_root_dir"]
output_dir = config['output_dir']

# Get the input type selected via command line argument
selected_input_type = args.input_type
class_names = sorted(entry for entry in os.listdir(image_root_dir) if not entry.startswith('.'))
print(len(class_names))
###Add augmentation to training data

# Augmentation for training data
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_shape, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.GaussianBlur(kernel_size=5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# No augmentation for validation data
val_transform = transforms.Compose([
    transforms.Resize(input_shape),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# Custom Dataset for Semantic Segmentation
class CustomDataset(Dataset):
    def __init__(self, image_root_dir, mask_root_dir, transform=None, mask_transform=None):
        self.image_root_dir = image_root_dir
        self.mask_root_dir = mask_root_dir

        self.transform = transform
        self.mask_transform = mask_transform
        class_names = sorted(entry for entry in os.listdir(image_root_dir) if not entry.startswith('.'))

        self.class_to_index = {class_name: i for i, class_name in enumerate(class_names)}

        print("Classes in class_to_index:", self.class_to_index)
        print("Classes in class_to_name:",class_names)
        print(len(class_names))

        self.image_paths = []
        for root, _, files in os.walk(image_root_dir):
            for file in files:
                image_types = set()
                if "NDVI" in file:
                    image_types.add("NDVI")
                if 'SAVI' in file:
                    image_types.add("SAVI")
                if "TCI" in file:
                    image_types.add("TCI")
                if "RGB" in file:
                    image_types.add("RGB")
                if "TCI" in file:
                    image_types.add("TCI+NDVI")

                if "NDVI" in image_types or "TCI" in image_types or "RGB" in image_types:
                    image_path = os.path.join(root, file)
                    corresponding_mask_path = self.get_corresponding_mask_path(image_path)
                    if corresponding_mask_path:
                        print(f"Image Path: {image_path}, Mask Path: {corresponding_mask_path}, Class Index: {self.get_class_index_from_path(image_path)}")
                        self.image_paths.append((image_path, corresponding_mask_path))

    def get_corresponding_mask_path(self, image_path):
        relative_path = os.path.relpath(image_path, self.image_root_dir)
        mask_path = os.path.join(self.mask_root_dir, relative_path)
        if os.path.isfile(mask_path):
            return mask_path
        return None

    def get_class_index_from_path(self, image_path):

        # Extract class name from the path and convert it to class index
        relative_path = os.path.relpath(image_path, self.image_root_dir)
        for class_name, index in self.class_to_index.items():
            if class_name in relative_path:
                return class_name

        # Print class names and corresponding indices
        print("Class Names:", list(self.class_to_index.keys()))
        print("Class Indices:", list(self.class_to_index.values()))

        print("Class Name:", class_name)


        return None

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        img_path, mask_path = self.image_paths[idx]

        try:
            # Skip RGBN images
            if "RGBN" in img_path:
                print(f"Skipped RGBN image: {img_path}")
                # Return None for skipped images
                return (
                    torch.zeros((3, input_shape[0], input_shape[1]), dtype=torch.float),
                    torch.zeros((1, input_shape[0], input_shape[1]), dtype=torch.float),
                    torch.tensor([]),  # Return an empty tensor for skipped images
                )

           
            # Choose the appropriate input type based on the selected_input_type
            if "NDVI" in selected_input_type:
                image = Image.open(img_path.replace("RGB", "NDVI")).convert('L')
                image = image.convert('RGB')
            elif "SAVI" in selected_input_type:
                image = Image.open(img_path.replace("RGB", "SAVI")).convert('L')
                image = image.convert('RGB')

            elif "TCI" in selected_input_type:
                image = Image.open(img_path.replace("RGB", "TCI")).convert('RGB')
            elif "RGB" in selected_input_type:
                image = Image.open(img_path).convert('RGB')
                print(image.size)
            elif "TCI+NDVI" in selected_input_type:
                # Combine TCI and NDVI images (example: assuming NDVI is a separate channel)
                tci_image = Image.open(img_path.replace("RGB", "TCI")).convert('RGB')
                ndvi_image = Image.open(img_path.replace("RGB", "NDVI")).convert('L')
                ndvi_image = ndvi_image.convert('RGB')  # Convert to RGB
                image = Image.merge('RGB', (tci_image, ndvi_image))
            else:
                raise ValueError("Invalid input_type")

            mask = Image.open(mask_path).convert('L')

            image = transforms.functional.resize(image, (input_shape[0], input_shape[1]))

            mask = mask.resize((input_shape[1], input_shape[0]), Image.NEAREST)

            if self.transform:
                image = self.transform(image)
            if self.mask_transform:
                mask = self.mask_transform(mask)

            if not isinstance(image, torch.Tensor):
                image = transforms.ToTensor()(image)
            if not isinstance(mask, torch.Tensor):
                mask = transforms.ToTensor()(mask)

            class_name = self.get_class_index_from_path(img_path)

            label = torch.tensor(self.class_to_index[class_name]).unsqueeze(0)
            print(label)
            #print(mask.shape)
            #print(image.shape)
            #print(label.shape)

            return image, mask, label

        except (PIL.UnidentifiedImageError, OSError) as e:
            print(f"Error loading image: {e}")
            # Return None for skipped images
            return (
                torch.zeros((3, input_shape[0], input_shape[1]), dtype=torch.float),
                torch.zeros((1, input_shape[0], input_shape[1]), dtype=torch.float),
                torch.tensor([-1], dtype=torch.long),  # Return a placeholder index for skipped images
            )






def custom_collate(batch):
    # Filter out samples with labels equal to -1
    batch = [sample for sample in batch if sample[2].numel() > 0 and sample[2].item() != -1]

    if len(batch) == 0:
        return torch.zeros((0, 3, input_shape[0], input_shape[1])), torch.zeros((0, 1, input_shape[0], input_shape[1])), torch.tensor([], dtype=torch.long)

    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    try:
        # Attempt to stack tensors
        images = torch.stack(images)
        masks = torch.stack(masks)
        labels = torch.stack(labels).long()  # Cast labels to Long type
    except RuntimeError as e:
        print(f"Error during stacking tensors: {e}")
        images = torch.zeros((0, 3, input_shape[0], input_shape[1]))
        masks = torch.zeros((0, 1, input_shape[0], input_shape[1]))
        labels = torch.tensor([], dtype=torch.long)

    # Normalize the images based on the input type
    for i in range(images.size(1)):
        if images[:, i, :, :].max() > 1:
            images[:, i, :, :] /= 255.0  # Normalize 8-bit images
    images = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(images)

    return images, masks, labels





# Modify your DataLoader to use the custom collate function
#crop_loader = DataLoader(crop_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

# Create dataset and dataloaders
crop_dataset = CustomDataset(image_root_dir, mask_root_dir, transform= train_transform)
crop_loader = DataLoader(crop_dataset, batch_size=batch_size, shuffle=True,collate_fn=custom_collate)
total_size = len(crop_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(crop_dataset, [train_size, val_size])

# Create data loaders for training and validation sets with custom collate function
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
# Use ResNet-50 pre-trained model
#### MODELS #####################################
#https://pytorch.org/vision/stable/models.html

#model = models.inception_v3()
#models.inception_v3
#num_ftrs = model.fc.in_features
num_classes = len(class_names)  # Extract number of classes from the data


# Use the specified pre-trained model architecture
model_type = args.model_type
if model_type == 'resnet34':
    model = models.resnet34(pretrained=True)
elif model_type == 'resnet50':
    model = models.resnet50(pretrained=True)
elif model_type == 'alexnet':
    model = models.alexnet(pretrained=True)
elif model_type == 'vgg19':
    model = models.vgg19(pretrained=True)
elif model_type == 'inception_v3':
    model = models.inception_v3(pretrained=True)
else:
    raise ValueError(f"Invalid model_type: {model_type}")

# For VGG models, modify the classifier part
if 'vgg' in model_type:
    # # Get the number of input features for the last fully connected layer in the classifier
    num_features = model.classifier[-1].in_features

    # Modify the last fully connected layer
    model.classifier[-1] = nn.Linear(num_features, num_classes)
else:
    # For ResNet models
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)


def compute_weight(dataset):
    # Count occurrences of each class in the dataset
    class_counts = torch.zeros(len(dataset.class_to_index))

    for _, _, labels in dataset:
        if labels.numel() > 0 and labels.item() != -1:
            class_counts[labels.item()] += 1

    # Identify non-empty classes
    non_empty_classes = [i for i, count in enumerate(class_counts) if count > 0]

    # Compute weights based on the inverse of class frequencies for non-empty classes
    class_weights = torch.zeros(len(dataset.class_to_index))
    class_weights[non_empty_classes] = 1.0 / class_counts[non_empty_classes].float()

    # Normalize the weights
    class_weights /= class_weights.sum()

    return class_weights


# Specify the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compute class weights
class_weights = compute_weight(crop_dataset)
class_weights = class_weights.to(device)

# Define the loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Move the model to the specified device
model = model.to(device)
# Training the model
model.train()

# Set the experiment name
experiment_name = "RGB_resnet50"
# Set the tracking URI to a local directory
experiment_dir = "Crop_classification_experiment"
mlflow.set_tracking_uri("file:" + experiment_dir)

# Check if the directory exists, create it if not
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

# Set the experiment
mlflow.set_experiment(experiment_name)
# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("input_type", args.input_type)
    mlflow.log_param("model_type", args.model_type)
    mlflow.log_param("LEARNING_RATE", LEARNING_RATE)
    mlflow.log_param("NUM_EPOCHS", NUM_EPOCHS)
    mlflow.log_param("input_shape", input_shape)
    mlflow.log_param("batch_size", batch_size)

    # ... (log other parameters or metrics)

    # Log the model
    mlflow.pytorch.log_model(model, "models")



    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        y_true_train = []
        y_pred_train = []

        # Use tqdm to display a progress bar
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', unit='batch') as tqdm_loader:
            for i, (inputs, masks, labels) in enumerate(tqdm_loader, 0):
                optimizer.zero_grad()

                # Ensure the model is set to training mode
                model.train()

                # If any element in labels is -1, it means the image is skipped, and you can handle it accordingly
                if labels.nelement() == 0:
                    continue

                # Squeeze the second dimension of the labels tensor
                labels = torch.squeeze(labels, dim=1)

                # Move inputs and labels to the appropriate device (e.g., GPU)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)

                # Calculate loss
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Append true and predicted labels for later metrics calculation
                y_true_train.extend(labels.cpu().numpy())
                y_pred_train.extend(torch.argmax(outputs, dim=1).cpu().numpy())

                if i % 100 == 99:
                    print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100:.5f}')
                    running_loss = 0.0

                    # Update tqdm progress bar with loss
                    tqdm_loader.set_postfix(loss=f'{running_loss / (i + 1):.5f}')

        # Calculate training metrics at the end of each epoch
        if y_true_train and y_pred_train:  # Check if there are non-empty lists
            accuracy_train = metrics.accuracy_score(y_true_train, y_pred_train)
            precision_train = metrics.precision_score(y_true_train, y_pred_train,average='macro')
            recall_train = metrics.recall_score(y_true_train, y_pred_train, average='macro')

            print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS} - Training Metrics:')
            print(f'Accuracy: {accuracy_train:.4f}')
            print(f'Precision: {precision_train:.4f}')
            print(f'Recall: {recall_train:.4f}')
        else:
            print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS} - No correct predictions made in this epoch.')



    print("Finished Training")
    torch.save(model.state_dict(), f'models/classification_model_{model_type}.pth')


###########################################################################################
#### Update config file with num_of_classes
# Add class_names and num_classes to the config dictionary
    config["class_names"] = class_names
    config["num_classes"] = num_classes

    # Save the modified configuration to the JSON file
    with open(config_file, "w") as f:
        json.dump(config, f)


    from sklearn.metrics import ConfusionMatrixDisplay

    y_true = []
    y_pred = []
    # Inside the validation loop
    # Inside the validation loop
    with torch.no_grad():
        y_true = []
        y_pred = []
        # Store unique class labels
        unique_labels = set()

        for inputs, masks, labels in val_loader:

            if labels.numel() == 0:  # Skip batch if labels tensor is empty
                print("Skipping batch. Labels tensor is empty.")
                continue

            # Pass the inputs directly to the crop classification model (exclude segmentation)
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
            unique_labels.update(labels.unique().tolist())  # Update the set of unique labels

    # Extract class names from the dataset using the class_to_index mapping
    class_names = [class_name for class_name, index in crop_dataset.class_to_index.items() if index in unique_labels]
    print(class_names)

        # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Calculate percentage counts for each class
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_percent = conf_matrix / row_sums


        # Compute validation metrics
    accuracy_val = metrics.accuracy_score(y_true, y_pred)
    precision_val = metrics.precision_score(y_true, y_pred, average='macro')
    recall_val = metrics.recall_score(y_true, y_pred, average='macro')

    print(f'Validation Metrics:')
    print(f'Accuracy: {accuracy_val:.4f}')
    print(f'Precision: {precision_val:.4f}')
    print(f'Recall: {recall_val:.4f}')

    print('Confusion Matrix:')
    # Use MLflow to log metrics
    #mlflow.log_metric('train_loss', running_loss / (i + 1), step=epoch)
    mlflow.log_metric('train_accuracy', accuracy_train, step=epoch)
    mlflow.log_metric('train_precision', precision_train, step=epoch)
    mlflow.log_metric('train_recall', recall_train, step=epoch)

    # Log validation metrics
    mlflow.log_metric("accuracy_val", accuracy_val)
    mlflow.log_metric("precision_val", precision_val)
    mlflow.log_metric("recall_val", recall_val)


    # Save the confusion matrix as an image
       # Display confusion matrix with class names
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_percent, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap='viridis', values_format='.1%', ax=ax, xticks_rotation='vertical')
    # Add xlabel and ylabel
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    # Adjust xticks and yticks
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    #ax.set_xticklabels(class_names, rotation='vertical')
    #ax.set_yticklabels(class_names)




    #fig, ax = plt.subplots(figsize=(5, 5))
    #disp.plot(cmap='viridis', values_format='.1%', ax=ax, xticks_rotation='vertical')
    #plt.savefig("confusion_matrix.png")
    # Add xlabel and ylabel
    #ax.set_xlabel("Predicted")
    #ax.set_ylabel("True")
    # Log the confusion matrix



    mlflow.log_artifact("confusion_matrix.png")
    plt.show()
    print(conf_matrix)

