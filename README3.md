# GNR 638 Assignment-3

DATE - 20-02-2025

Task : To train a small CNN network to perform classification over the UCMERCED Data

Data : Already split into train, val and test folders, which the following size
- Train - 1050
- Val - 420
- Test - 630

So for this assignment, we experimented with three different model settings, but first would need to understand these terms

- Data transformation

Now in order to prevent the model from overfitting, we apply a few transformation to the data and also implement data augmentation so that the model generalizes better. The transformation sets we decided to use are

1) Transform1
```python
transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  
    transforms.RandomRotation(30),  
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

2) Transform2
```python
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
```

To explain each of these transformations
- 
- 
- 

The only difference between the 2 transformations is that in the 2nd one, we include the ColorJitter and the GaussianBlur

Now we also 2 model architectures to mainly study the effect of applying batch normalization

```python
class SmallCNN(nn.Module):
    def __init__(self, num_classes=21):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3=nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)  # Since input is 128x128
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5) 
        self.feature_maps = None


    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        self.feature_maps = x
        x = x.view(x.size(0), -1)  
        x = torch.relu(self.fc1(x))
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
```


So here are the experiment setups we have then

1) SmallCNN1 + Transform1
2) SmallCNN2 + Transform1
3) SmallCNN3 + Transform2

The results are as follows