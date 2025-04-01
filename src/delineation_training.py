import os
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import jaccard_score
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import albumentations as A


#Parameter tunning
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1
input_shape = (512, 512)
batch_size = 4



 #Define input and output directories with the absolute path
image_root_dir = '/Users/brightabohsilasedem/Desktop/NSIR_Project/try'  # Absolute path to directory containing input TCI images
mask_root_dir = '/Users/brightabohsilasedem/Desktop/NSIR_Project/binary_mask_2_thick'  # Absolute path to directory containing binary masks
output_dir = '/Users/brightabohsilasedem/Desktop/NSIR_Project/segmentation_results'  # Absolute path to directory to save segmentation results
# Create a "plots" folder if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Lists to store training and validation metrics
train_loss_history = []
val_loss_history = []
val_iou_history = []
val_jaccard_history = []





# Custom Dataset for Semantic Segmentation
class CustomDataset(Dataset):
    def __init__(self, image_root_dir, mask_root_dir, transform=None, mask_transform=None):
        self.image_root_dir = image_root_dir
        self.mask_root_dir = mask_root_dir

        self.transform = transform
        self.mask_transform = mask_transform

        # List all available TCI images recursively
        self.image_paths = []
        for root, _, files in os.walk(image_root_dir):
            for file in files:
                if "TCI" in file:  # Adjust this condition based on TCI image file naming convention
                    image_path = os.path.join(root, file)
                    corresponding_mask_path = self.get_corresponding_mask_path(image_path)
                    if corresponding_mask_path:
                        self.image_paths.append((image_path, corresponding_mask_path))

    def get_corresponding_mask_path(self, image_path):
        # Generate the corresponding mask path based on the subdirectory structure
        relative_path = os.path.relpath(image_path, self.image_root_dir)
        mask_path = os.path.join(self.mask_root_dir, relative_path)
        if os.path.isfile(mask_path):
            return mask_path
        return None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_paths[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Resize both image and mask to 512x512
        image = image.resize((input_shape[1], input_shape[0]))

        mask = mask.resize((input_shape[1], input_shape[0]), Image.NEAREST)


        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)  #
        #print("Image Shape:", image.shape)
        #print("Mask Shape:", mask.shape)
        return image, mask
# Define the mean and standard deviation values for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Define data transformations with normalization and standardization
transform = transforms.Compose([
    transforms.Resize(input_shape),  # Resize images to match the model's input size
    transforms.ToTensor(),
    # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(mean, std)  # Normalize the input with mean and standard deviation
])
# Define data transformations with normalization and standardization for masks
mask_transform = transforms.Compose([
    transforms.Resize(input_shape),  # Resize images to match the model's input size
    transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
    #transforms.Normalize([0.5], [0.5])  # Normalize the input with mean 0.5 and std 0.5
])

# Create dataset and dataloaders
train_dataset = CustomDataset(image_root_dir, mask_root_dir, transform=transform, mask_transform=mask_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



# Create dataset and data loaders for training and validation
full_dataset = CustomDataset(image_root_dir, mask_root_dir, transform=transform,mask_transform=mask_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Helper function to display images and masks

def display_images_masks(loader, num_samples=3):
    for i, (images, masks) in enumerate(loader):
        if i >= num_samples:
            break

        images = images.numpy()  # Convert torch.Tensor to numpy array
        masks = masks.numpy()

        # Display images and masks
        for j in range(len(images)):
            plt.figure(figsize=(12, 4))

            # Ensure images are in the correct format and range for display
            image_to_show = np.transpose(images[j], (1, 2, 0))
            image_to_show = (image_to_show - image_to_show.min()) / (image_to_show.max() - image_to_show.min())  # Normalize the image

            # Display the image
            plt.subplot(1, 2, 1)
            plt.imshow(image_to_show)
            plt.title('Image')
            plt.axis('off')

            # Display the mask
            plt.subplot(1, 2, 2)
            plt.imshow(masks[j][0], cmap='gray')
            plt.title('Mask')
            plt.axis('off')

            plt.show()


# Display sample images and masks from the train loader
print("Sample images and masks from the train loader:")
display_images_masks(train_loader, num_samples=3)

# Display sample images and masks from the val loader
print("Sample images and masks from the val loader:")
display_images_masks(val_loader, num_samples=3)




# Check for GPU availability and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DeepLab Model
# Load pre-trained DeepLab model with ResNet-18 backbone


def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_jaccard(y_true, y_pred):
    return jaccard_score(y_true.flatten(), y_pred.flatten())

def check_accuracy(loader, model, device=device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        iou_values = []
        jaccard_values = []

        for img, mask in tqdm(loader):
            img = img.to(device)
            mask = mask.float().to(device)  # Removed unnecessary .unsqueeze(1)

            preds = torch.sigmoid(model(img))
            preds = (preds > 0.6).float()

            num_correct += (preds == mask).sum().item()  # Added .item() to get a Python scalar
            num_pixels += mask.numel()  # Use mask.numel() to get the total number of elements
            dice_score += (2 * (preds * mask).sum()) / (preds.sum() + mask.sum() + 1e-7)  # Removed 3e-4

            # Calculate IoU and Jaccard Similarity for each batch
            iou = calculate_iou(mask.cpu().numpy(), preds.cpu().numpy())
            jaccard = calculate_jaccard(mask.cpu().numpy(), preds.cpu().numpy())

            iou_values.append(iou)
            jaccard_values.append(jaccard)

    pixel_accuracy = num_correct / num_pixels * 100
    dice_score = dice_score / len(loader) * 100

    iou_mean = np.mean(iou_values)
    jaccard_mean = np.mean(jaccard_values)

    print(f"Pixel Accuracy: {pixel_accuracy:.2f}%")
    print(f"Dice Score: {dice_score:.2f}%")
    print(f"Mean IoU: {iou_mean:.2f}")
    print(f"Mean Jaccard Similarity: {jaccard_mean:.2f}")

    model.train()

    return pixel_accuracy, dice_score, iou_mean, jaccard_mean

model = smp.UnetPlusPlus(encoder_name='resnet34', in_channels=3, classes=1, activation=None).to(device)
loss_fn   = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)

    for batch_idx, (image, mask) in enumerate(loop):
        image   = image.to(device=device)
        #mask    = mask.float().unsqueeze(1).to(device=device)
        mask = mask.to(device=device)

        # forward
        predictions = model(image)
        loss = loss_fn(predictions, mask)

        # backward
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


check_accuracy(val_loader, model, device=device)
# Save the trained model


# Initialize history lists
train_loss_history = []
val_loss_history = []
val_iou_history = []
val_jaccard_history = []

# Training loop
for epoch in range(NUM_EPOCHS):
    print('########################## epoch:', epoch)
    # Log parameters for the current run
    mlflow.log_param("epoch", epoch)
    train_loss = train_fn(train_loader, model, optimizer, loss_fn)
    #val_loss = check_accuracy(val_loader, model, device=device)
    # Evaluate the model on the validation set
    # With this line
    result = check_accuracy(val_loader, model, device=device)
    val_loss, val_iou, val_jaccard, *_ = result  #
    #val_loss, val_iou, val_jaccard = check_accuracy(val_loader, model, device=device)
    # Log metrics for the current run
    mlflow.log_metric("train_loss", train_loss)
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_iou", val_iou)
    mlflow.log_metric("val_jaccard", val_jaccard)

    # Log the model
    mlflow.pytorch.log_model(model, "models")

    # Append training and validation losses and metrics to the lists
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    #val_iou_history.append(val_iou)
    #val_jaccard_history.append(val_jaccard)

# Plot and save training and validation loss graphs after the training loop
# Save training and validation loss plots
plt.figure()
plt.plot(range(NUM_EPOCHS), train_loss_history, label="Training Loss")
plt.plot(range(NUM_EPOCHS), val_loss_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")


# Save the plot in the "plots" folder
plt.savefig(os.path.join("plots", "loss_final.png"))
plt.close()

# Save validation IoU and Jaccard Similarity plots
plt.figure()




def display_images_masks_predictions(loader, model, num_samples=5):
    # Get a batch of data
    for i, (images, masks) in enumerate(loader):
        if i >= num_samples:
            break

        images = images.numpy()
        masks = masks.numpy()

        # Predict with the model
        images = torch.from_numpy(images).to(device)
        with torch.no_grad():
            predictions = model(images)
            predictions = torch.sigmoid(predictions)

        images_rgb = np.transpose(images, (0, 2, 3, 1))  # Change from (batch, channel, height, width) to (batch, height, width, channel)
        masks = np.squeeze(masks, axis=1)  # Remove the channel dimension for masks
        predictions = predictions.cpu().numpy()

        # Display images, ground truth masks, and predicted masks
        for j in range(len(images)):
            plt.figure(figsize=(16, 5))

            # Normalize the input image to display
            image_to_show = images_rgb[j]
            image_to_show = (image_to_show - image_to_show.min()) / (image_to_show.max() - image_to_show.min())

            # Display the input image in RGB
            plt.subplot(1, 3, 1)
            plt.imshow(image_to_show)
            plt.title('Input Image')
            plt.axis('off')

            # Display the ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(masks[j], cmap='gray')
            plt.title('Ground Truth Mask')
            plt.axis('off')

            # Display the predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(predictions[j][0], cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')

            plt.show()


# Check accuracy
check_accuracy(val_loader, model, device=device)
torch.save(model.state_dict(), 'models/segmentation_model_resnet50.pth')
# Display sample images, masks, and predictions from the validation loader
print("Sample images, masks, and predictions from the validation loader:")
display_images_masks_predictions(val_loader, model, num_samples=5)

