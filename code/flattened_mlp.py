from __future__ import print_function
from random import shuffle
import os
from PIL import Image
import argparse
import pickle
from glob import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import numpy as np

def resize_and_flatten(image_path, size=(72, 72)):
    # Open image
     image = Image.open(image_path)
        
        # Resize image to 72x72
     image_resized = image.resize(size)

     image_array = np.array(image_resized)

        # Convert image to numpy array and flatten it
     img_flattened = image_array.flatten()
     return img_flattened

def process_images(image_paths):
    return np.array([resize_and_flatten(path) for path in image_paths])

parser = argparse.ArgumentParser()
parser.add_argument('--classifier', help='classifier', type=str, default='mlp')
args = parser.parse_args()

DATA_PATH = '../Merced_resized/'
CATEGORIES = os.listdir(DATA_PATH)
CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

TRAIN_PERCENT = 70
VAL_PERCENT = 10
TEST_PERCENT = 20
CLASSIFIER = args.classifier

def shuffle_lists(list1, list2):
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    shuffled_list1, shuffled_list2 = zip(*combined)
    return list(shuffled_list1), list(shuffled_list2)

def path_loader(data_path, categories, train_percent, val_percent, test_percent):
    train_image_paths, val_image_paths, test_image_paths = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for category in categories:
        image_paths = glob(os.path.join(data_path, category, '*.tif'))
        random.shuffle(image_paths)

        train_size = int(len(image_paths) * train_percent / 100)
        val_size = int(len(image_paths) * val_percent / 100)
        test_size = int(len(image_paths) * test_percent / 100)

        train_image_paths.extend(image_paths[:train_size])
        val_image_paths.extend(image_paths[train_size:train_size + val_size])
        test_image_paths.extend(image_paths[train_size + val_size:])

        train_labels.extend([category] * train_size)
        val_labels.extend([category] * val_size)
        test_labels.extend([category] * test_size)

    train_image_paths, train_labels = shuffle_lists(train_image_paths, train_labels)
    val_image_paths, val_labels = shuffle_lists(val_image_paths, val_labels)
    test_image_paths, test_labels = shuffle_lists(test_image_paths, test_labels)

    return train_image_paths, val_image_paths, test_image_paths, train_labels, val_labels, test_labels

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1=512, hidden_size2=256, num_classes=21):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.Linear()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.Linear()
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
def train_mlp(train_feats, train_labels, val_feats, val_labels, vocab_size, num_epochs=200, lr=0.001):
       le = LabelEncoder()
       train_labels = le.fit_transform(train_labels)
       val_labels = le.transform(val_labels)
       joblib.dump(le, "label_encoder.pkl")


       train_feats = torch.tensor(train_feats, dtype=torch.float32)
       train_labels = torch.tensor(train_labels, dtype=torch.long)
       val_feats = torch.tensor(val_feats, dtype=torch.float32)
       val_labels = torch.tensor(val_labels, dtype=torch.long)

       train_dataset = TensorDataset(train_feats, train_labels)
       train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

       model = MLP(input_size=train_feats.shape[1], num_classes=len(le.classes_))
       criterion = nn.CrossEntropyLoss()
       optimizer = optim.Adam(model.parameters(), lr=lr)

       print(f"Training MLP for vocab size {vocab_size}...")
       for epoch in range(num_epochs):
         for features, labels in train_loader:
             optimizer.zero_grad()
             outputs = model(features)
             loss = criterion(outputs, labels)
             loss.backward()
             optimizer.step()

         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

       torch.save(model.state_dict(), f'../store/mlp_model_linear_2048_1024_Flattened_{vocab_size}.pth')

from sklearn.model_selection import KFold

def cross_validate_mlp(val_feats, val_labels, vocab_size, num_folds=5, num_epochs=200, lr=0.001):
    le = joblib.load("label_encoder.pkl")
    val_labels = le.transform(val_labels)

    val_feats = torch.tensor(val_feats, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    HiddenLayers = [(512, 256), (1024, 512), (2048, 1024)]
    Activations = ['ReLU', 'Tanh', 'GELU', 'linear']

    accuracies = {}
    best_acc = 0
    layer1 = 0
    layer2 = 0
    act = ''

    #for i in range(len(HiddenLayers)):
        #for j in range(len(Activations)):
            # Recreate the model architecture before loading weights
    model = MLP(input_size=val_feats.shape[1], 
                        hidden_size1=HiddenLayers[0][0], 
                        hidden_size2=HiddenLayers[0][1], 
                        num_classes=len(le.classes_))

            # Load the state_dict
    state_dict = torch.load(f'../store/mlp_model_{Activations[3]}_{HiddenLayers[0][0]}_{HiddenLayers[0][1]}_Flattened_600.pth')
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
            predictions = model(val_feats).argmax(dim=1)
            accuracy = (predictions == val_labels).float().mean().item()
            print(f'Validation Accuracy (MLP) for pair_{Activations[3]}_{HiddenLayers[0][0]}_{HiddenLayers[0][1]} : {accuracy:.4f}')

    accuracies[((HiddenLayers[0][0],HiddenLayers[0][1]), Activations[3])] = accuracy

    if accuracy > best_acc:
            best_acc = accuracy
            layer1, layer2 = HiddenLayers[0][0], HiddenLayers[0][1]
            act = Activations[3]

    print(f'Best Validation Accuracy shows for pair ({layer1}, {layer2}, {act}) = {best_acc}')
    print('STARTING TEST ON BEST PARAMETERS')

    return accuracies, [layer1, layer2, act, best_acc]

def test_mlp2(model_paths, test_feats, test_labels, hidden_size1, hidden_size2, activation):
    le = joblib.load("label_encoder.pkl")
    test_labels = le.transform(test_labels)

    test_feats = torch.tensor(test_feats, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

            # Recreate the model architecture before loading weights
    model = MLP(input_size=test_feats.shape[1], 
                hidden_size1=hidden_size1, 
                hidden_size2=hidden_size2, 
                num_classes=len(le.classes_))

            # Load the state_dict
    state_dict = torch.load(f'../store/mlp_model_{activation}_{hidden_size1}_{hidden_size2}_Flattened_600.pth')
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
         predictions = model(test_feats).argmax(dim=1)
         accuracy = (predictions == test_labels).float().mean().item()
         print(f'Test Accuracy (MLP) for pair_{activation}_{hidden_size1}_{hidden_size2} : {accuracy:.4f}')

    return accuracy


def main():
    print("Setting training, validation, and testing splits")

    train_image_paths, val_image_paths, test_image_paths, train_labels, val_labels, test_labels = \
        path_loader(DATA_PATH, CATEGORIES, TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT)

    train_image_feats = process_images(train_image_paths)
    val_image_feats = process_images(val_image_paths)
    test_image_feats = process_images(test_image_paths)
    vocab_size=600

    if CLASSIFIER == 'mlp':
        model = MLP(input_size=train_image_feats.shape[1], num_classes=len(set(train_labels)))
        print("Model initialized with input size:", train_image_feats.shape[1])
        train_mlp(train_image_feats, train_labels, val_image_feats, val_labels, vocab_size, num_epochs=200, lr=0.001)
        accuracies_record, [layer1, layer2, act, best_acc]=cross_validate_mlp(val_image_feats, val_labels, vocab_size, num_folds=5, num_epochs=200, lr=0.001)
        test_labels = np.array(test_labels)
        model_path = f'../store/mlp_model_{act}_{layer1}_{layer2}_Flattened_600.pth'
        test_accuracy=test_mlp2(model_path, test_image_feats, test_labels,layer1,layer2,act)
        

if __name__ == '__main__':
    main()
