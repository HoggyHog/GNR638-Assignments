import pickle
import argparse
import matplotlib.pyplot as plt
from nearest_neighbor_classify import nearest_neighbor_classify
import os
from svm_classify import svm_classify
from visualize import visualize,build_confusion_mtx,plot_confusion_matrix
import numpy as np
from PIL import Image
from cyvlfeat.sift.dsift import dsift
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument('--classifier', help='classifier', type=str, default='nearest_neighbor')
args = parser.parse_args()

DATA_PATH = '../Merced/'

CLASSIFIER = args.classifier

CATEGORIES=os.listdir(DATA_PATH)

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

ABBR_CATEGORIES = [i[:3] for i in CATEGORIES]


vocab_sizes=[200,300,400,500,600]
accuracies=[]

max_accuracy=0
best_vocab=0

with open(f'../store/history-200.pkl', 'rb') as handle:
    history=pickle.load(handle)
    train_image_paths=history['train_image_paths']
    val_image_paths=history['val_image_paths']
    test_image_paths=history['test_image_paths']
    train_labels=history['train_labels']
    val_labels=history['val_labels']
    test_labels=history['test_labels']



for vocab_size in vocab_sizes:

    with open(f'../store/train_image_feats-{vocab_size}.pkl', 'rb') as handle:
        train_image_feats = pickle.load(handle)
    with open(f'../store/val_image_feats-{vocab_size}.pkl', 'rb') as handle:
        val_image_feats = pickle.load(handle)
    
    if CLASSIFIER == 'nearest_neighbor':
            # YOU CODE nearest_neighbor_classify.py
            predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, val_image_feats,CATEGORIES)

    elif CLASSIFIER == 'support_vector_machine':
        # YOU CODE svm_classify.py
        predicted_categories = svm_classify(train_image_feats, train_labels, val_image_feats)

    accuracy = float(len([x for x in zip(val_labels,predicted_categories) if x[0]== x[1]]))/float(len(val_labels))
    print("Accuracy of vocab size ",vocab_size ," = ", accuracy)


    accuracies.append(accuracy)
    if(accuracy>max_accuracy):
        max_accuracy=accuracy
        best_vocab=vocab_size
    
print('Best vocab size ',best_vocab)

filename=f'../results/val-accuracies-{CLASSIFIER}.png'

# PLOT A GRAPH FOR THE VALIDATION ACCURACIES

plt.figure(figsize=(10, 6))
plt.plot(vocab_sizes, accuracies)
plt.xlabel('VOCAB SIZES')
plt.ylabel('ACCURACY')
plt.title('Graph of Accuracy for different number of codewords over the validation set')
plt.grid(True)

plt.savefig(filename)


# DISPLAY THE CLASS-WISE ACCURACIES

with open(f'../store/train_image_feats-{vocab_size}.pkl', 'rb') as handle:
    best_train_feats = pickle.load(handle)

with open(f'../store/train_image_feats-{best_vocab}.pkl', 'rb') as handle:
    best_train_feats = pickle.load(handle)
with open(f'../store/test_image_feats-{best_vocab}.pkl', 'rb') as handle:
    best_test_feats = pickle.load(handle)

if CLASSIFIER == 'nearest_neighbor':
        # YOU CODE nearest_neighbor_classify.py
    predicted_categories = nearest_neighbor_classify(best_train_feats, train_labels, best_test_feats,CATEGORIES)

elif CLASSIFIER == 'support_vector_machine':
    # YOU CODE svm_classify.py
    predicted_categories = svm_classify(best_train_feats, train_labels, best_test_feats)

accuracy = float(len([x for x in zip(test_labels,predicted_categories) if x[0]== x[1]]))/float(len(test_labels))
print("Accuracy = ", accuracy)

print()

print("PRINTING CLASS-WISE ACCURACIES")

print()

for category in CATEGORIES:
    accuracy_each = float(len([x for x in zip(test_labels,predicted_categories) if x[0]==x[1] and x[0]==category]))/float(test_labels.count(category))
    print(str(category) + ': ' + str(accuracy_each))

test_labels_ids = [CATE2ID[x] for x in test_labels]
predicted_categories_ids = [CATE2ID[x] for x in predicted_categories]
train_labels_ids = [CATE2ID[x] for x in train_labels]

build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES,f'../results/confusion-{CLASSIFIER}')
#visualize(CATEGORIES, test_image_paths, test_labels_ids, predicted_categories_ids, train_image_paths, train_labels_ids)

images=['../Merced/airplane/airplane00.tif', '../Merced/beach/beach00.tif','../Merced/forest/forest00.tif','../Merced/harbor/harbor00.tif','../Merced/river/river00.tif']
classes=['airplane','beach','forest','harbor','river']
print()
print('CREATING t-SNE files')
print()
for i in range(5):
    path=images[i]
    obj=classes[i]
    print('obj ->',obj)
    img=Image.open(path)
    img = img.convert("L")
    img = np.asarray(img,dtype='float32')
    frames, descriptors = dsift(img, step=[3,3], fast=True)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    descriptors_2d = tsne.fit_transform(descriptors)

    # Plot the 2D t-SNE result
    plt.figure(figsize=(8, 8))
    plt.scatter(descriptors_2d[:, 0], descriptors_2d[:, 1], s=5, cmap='viridis')
    plt.title(f"t-SNE Visualization of SIFT Descriptors for {path.split('/')[-1]}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(f'../results/t-SNE-{obj}.png')
    plt.close()