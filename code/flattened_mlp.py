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
import torch.optim as optim

import matplotlib.pyplot as plt

def resize_and_flatten(image_path, size=(72, 72)):
    # Open image
     image = Image.open(image_path)
     image = image.convert("L")
        # Resize image to 72x72
     image_resized = image.resize(size)

     image_array = np.array(image_resized)
     #print(image_array.shape)

        # Convert image to numpy array and flatten it
     img_flattened = image_array.flatten()
     return img_flattened

def process_images(image_paths):
    return np.array([resize_and_flatten(path) for path in image_paths])

parser = argparse.ArgumentParser()
parser.add_argument('--classifier', help='classifier', type=str, default='mlp')
args = parser.parse_args()

DATA_PATH = '../Merced/'
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
    def __init__(self, input_size, activation,hidden_size1=2048, hidden_size2=1024, num_classes=21):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.tanh1= nn.Tanh()
        self.tanh2= nn.Tanh()

        self.activation=activation


    def forward(self, x):

        if(self.activation=='ReLU'):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.fc3(x)
        
        elif(self.activation=='Linear'):
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
        
        elif(self.activation=='Tanh'):
            x = self.tanh1(self.fc1(x))
            x = self.tanh2(self.fc2(x))
            x = self.fc3(x)

        return x


def train_mlp(train_feats, train_labels, num_epochs=200, lr=0.01):
       
    train_feats = torch.tensor(train_feats, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    HiddenLayers = [(512, 256), (1024, 512), (2048, 1024)]
    Activations = ['ReLU', 'Tanh', 'Linear']

    for i in range(len(HiddenLayers)):
        for j in range(len(Activations)):


            print(f'TRAINING MODEL FOR PARAMETERS {HiddenLayers[i]} - {Activations[j]}')

            model = MLP(input_size=train_feats.shape[1], 
                        hidden_size1=HiddenLayers[i][0], 
                        hidden_size2=HiddenLayers[i][1],
                        activation=Activations[j],
                        num_classes=len(CATEGORIES))
            
            train_dataset = TensorDataset(train_feats, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15, verbose=True)

            for epoch in range(num_epochs):
                loss_count=0
                for features, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss_count+=loss.item()
                    loss.backward()
                    optimizer.step()
                scheduler.step(loss_count/len(train_loader))

                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
            
            torch.save(model,f'../store/mlp_model_{Activations[j]}_{HiddenLayers[i][0]}_{HiddenLayers[i][1]}_Flattened.pth')

def val_and_test_mlp(val_feats, val_labels,test_feats,test_labels):
    
    val_feats = torch.tensor(val_feats, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    test_feats = torch.tensor(test_feats, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    HiddenLayers = [(512, 256), (1024, 512), (2048, 1024)]
    Activations = ['ReLU', 'Tanh', 'Linear']

    accuracy_dict = {}
    best_acc = 0
    best_hidden=0
    best_act = ''

    print('Starting Validation of Models over Validation data')

    for i in range(len(HiddenLayers)):
        for j in range(len(Activations)):

            model =  torch.load(f'../store/mlp_model_{Activations[j]}_{HiddenLayers[i][0]}_{HiddenLayers[i][1]}_Flattened.pth')
            #print(f'../store/mlp_model_{Activations[j]}_{HiddenLayers[i][0]}_{HiddenLayers[i][1]}.pth')
            model.eval()
            with torch.no_grad():
                predictions = model(val_feats).argmax(dim=1)
                accuracy = (predictions == val_labels).float().mean().item()
                print(f'Validation Accuracy (MLP) for pair_{Activations[j]}_{HiddenLayers[i][0]}_{HiddenLayers[i][1]} : {accuracy:.4f}')

            accuracy_dict[f'{HiddenLayers[i][0]}-{HiddenLayers[i][1]}-{Activations[j]}'] = accuracy

            if accuracy > best_acc:
                best_acc = accuracy
                best_hidden=i
                best_act = j

    print(f'Best Validation Accuracy shows for pair ({HiddenLayers[best_hidden][0]}-{HiddenLayers[best_hidden][1]}, {Activations[best_act]}) = {best_acc}')
    
    graph_file='../results/val-accuracies-MLP-flattened.png'

    print(f'Saving graph of validation accuracies to {graph_file}')
    models = list(accuracy_dict.keys())
    accuracies = list(accuracy_dict.values())

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(models, accuracies, marker='o', linestyle='-', linewidth=2, markersize=8, color='b')

    # Labels and title
    plt.xlabel("Models", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Model Accuracy Comparison", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save the plot as a PNG file
    plt.savefig(graph_file, dpi=300, bbox_inches='tight')



    print('STARTING TEST ON BEST PARAMETERS')

    best_model =  torch.load(f'../store/mlp_model_{Activations[best_act]}_{HiddenLayers[best_hidden][0]}_{HiddenLayers[best_hidden][1]}_Flattened.pth')
    best_model.eval()

    with torch.no_grad():
        predictions = best_model(val_feats).argmax(dim=1)
        accuracy = (predictions == val_labels).float().mean().item()
        print(f'ACCURACY FOR BEST MODEL for pair_({HiddenLayers[best_hidden][0]}-{HiddenLayers[best_hidden][1]}, {Activations[best_act]}) : {accuracy:.4f}')


def main():
    print("Setting training, validation, and testing splits")

    train_image_paths, val_image_paths, test_image_paths, train_labels, val_labels, test_labels = \
        path_loader(DATA_PATH, CATEGORIES, TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT)

    train_image_feats = process_images(train_image_paths)
    val_image_feats = process_images(val_image_paths)
    test_image_feats = process_images(test_image_paths)

    le = LabelEncoder()
    
    train_labels = le.fit_transform(train_labels)
    val_labels = le.transform(val_labels)
    test_labels = le.transform(test_labels)
    joblib.dump(le, "label_encoder.pkl")


    # print(len(train_image_feats),len(train_labels))

    if CLASSIFIER == 'mlp':
        #train_mlp(train_image_feats,train_labels)
        val_and_test_mlp(val_image_feats,val_labels,test_image_feats,test_labels)

        

if __name__ == '__main__':
    main()
