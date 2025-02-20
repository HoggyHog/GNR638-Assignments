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



def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
def seed_worker(worker_id):
    np.random.seed(42)
    random.seed(42)

set_seed(42)  # Call this at the start of your script
# Define a Simple CNN Model
class SmallCNN(nn.Module):
    def __init__(self, num_classes=21):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # Assuming 128x128 input images
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5) 
        self.feature_maps = None

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        self.feature_maps = x
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
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x= self.pool(torch.relu(self.conv3(x)))
        self.feature_maps = x
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        #x= self.dropout(x) 
        x = self.fc2(x)
        return x

# Set dataset paths
dataset_path = "./Ucmerced/Images"

save_dir = "./models"  
os.makedirs(save_dir, exist_ok=True)  


# Define image transformations (Resize to 128x128 for CNN)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  
    transforms.RandomRotation(30),   
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load datasets
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
val_dataset = datasets.ImageFolder(root=f"{dataset_path}/val", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"✅ Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")


#train and val
# Initialize CNN model
num_classes = len(train_dataset.classes)  # Get the number of classes
model = SmallCNN2(num_classes=num_classes)

# Define Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100 # Adjust as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses=[]
val_losses=[]

for epoch in range(num_epochs):
    model.train()  # Training mode
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_losses.append(loss.item())
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # Validate the model
    model.eval()  # Evaluation mode
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
model_filename = f"cnn_ucmerced_drop_epoch_{epoch+1}.pth"  # Modify this as needed
model_path = os.path.join(save_dir, model_filename)
torch.save(model.state_dict(), model_path)
print(f"Model saved: {model_path}")
print("Training Complete!")

plt.figure(figsize=(8, 5))
plt.plot(np.arange(len(train_losses)), train_losses, linestyle='-', color='b', label='Data Points')
plt.xlabel('Index')
plt.ylabel('Train Loss')
plt.title('Training Curve')

plt.savefig("Training_Curve_drop.png", dpi=300, bbox_inches='tight')  # Save first graph
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(np.arange(len(val_losses)), val_losses, linestyle='-', color='b', label='Data Points')
plt.xlabel('Index')
plt.ylabel('Val Loss')
plt.title('Validation Curve')

plt.savefig("Validation_Curve_drop.png", dpi=300, bbox_inches='tight')  # Save first graph
plt.close()

#once model is trained, we apply cam on an image


