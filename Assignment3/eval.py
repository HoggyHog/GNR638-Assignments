import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
import numpy
import numpy as np
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  
    transforms.RandomRotation(30),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2),
    #transforms.GaussianBlur(kernel_size=3),   
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform2 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=3),   
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_path = "./Ucmerced/Images"

# Load datasets

test_dataset1 = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform1)
test_dataset2 = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform2)

test_loader1 = DataLoader(test_dataset1, batch_size=32, shuffle=False)
test_loader2 = DataLoader(test_dataset2, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture (must match the saved model)
class SmallCNN1(nn.Module):
    def __init__(self, num_classes=21):
        super(SmallCNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3=nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)  # Assuming 128x128 input images
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5) 
        self.feature_maps = None

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x= torch.relu(self.conv3(x))
        self.feature_maps = x
        x=self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        #x= self.dropout(x) 
        x = self.fc2(x)
        return x
    
class SmallCNN2(nn.Module):
    def __init__(self, num_classes=21):
        super(SmallCNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3=nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)  # Assuming 128x128 input images
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5) 
        self.feature_maps = None

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x= torch.relu(self.bn3(self.conv3(x)))
        self.feature_maps = x
        x=self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        #x= self.dropout(x) 
        x = self.fc2(x)
        return x


# Initialize model
model1 = SmallCNN1()
model2 = SmallCNN2()
model3 = SmallCNN2()

# Load the saved state_dict
model1.load_state_dict(torch.load("./models/cnn_ucmerced_drop_epoch_100.pth", weights_only=True))
model2.load_state_dict(torch.load("./models/cnn_ucmerced_epoch_100.pth", weights_only=True))
model3.load_state_dict(torch.load("./models/cnn_ucmerced2_epoch_100.pth", weights_only=True))

# Set model to evaluation mode
model1.eval()
model2.eval()
model3.eval()

def generate_cam(model, image_tensor, class_idx):
    """
    Generates a CAM heatmap for a given image.
    """
    model.eval()  # Set model to evaluation mode

    # Forward pass to get feature maps and predictions
    output = model(image_tensor.unsqueeze(0))  # Add batch dimension
    feature_maps = model.feature_maps.detach().squeeze(0)  # Remove batch dimension
    weights = model.fc2.weight[class_idx].detach().cpu().numpy()  # Class-specific weights

    # Compute weighted sum of feature maps
    cam = torch.zeros(feature_maps.shape[1:], dtype=torch.float32)
    for i in range(feature_maps.shape[0]):
        cam += weights[i] * feature_maps[i].cpu().numpy()

    # Normalize CAM
    cam = np.maximum(cam, 0)  # ReLU activation
    cam = cam / cam.max()  # Normalize to [0,1]

    return cam


def overlay_cam(image_path, cam, filename):
    """
    Overlay the CAM heatmap on the original image.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))  # Resize to match CNN input

    cam = cam - cam.min()  
    cam = cam / (cam.max() + 1e-8)  # Normalize to [0,1]
    cam = np.power(cam, 0.5)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # Resize heatmap
    print(f"Image shape: {img.shape}, Heatmap shape: {heatmap.shape}")
    overlay = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    #plt.show()

# Example Usage:
image_path="./Ucmerced/Images/test/airplane/airplane25.jpg"
image = Image.open(image_path).convert("RGB")  # Open image
image_tensor = transform1(image).to(device)
cam = generate_cam(model1, image_tensor, class_idx=0)  # Change class index if needed
overlay_cam(image_path, cam,'image1.png')

image_tensor = transform1(image).to(device)
cam = generate_cam(model2, image_tensor, class_idx=0)  # Change class index if needed
overlay_cam(image_path, cam,'image2.png')

image_tensor = transform2(image).to(device)
cam = generate_cam(model3, image_tensor, class_idx=0)  # Change class index if needed
overlay_cam(image_path, cam,'image3.png')

#test

# Test the model
model1.eval()
model2.eval()
model3.eval()

test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader1:
        images, labels = images.to(device), labels.to(device)
        outputs = model1(images)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_acc = 100 * test_correct / test_total
print(f"🎯 Test Accuracy 1: {test_acc:.2f}%")

test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader1:
        images, labels = images.to(device), labels.to(device)
        outputs = model2(images)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_acc = 100 * test_correct / test_total
print(f"🎯 Test Accuracy 2: {test_acc:.2f}%")

test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader2:
        images, labels = images.to(device), labels.to(device)
        outputs = model3(images)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_acc = 100 * test_correct / test_total
print(f"🎯 Test Accuracy 3: {test_acc:.2f}%")

