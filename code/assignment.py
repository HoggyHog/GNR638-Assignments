from __future__ import print_function
from random import shuffle
import os
import argparse
import pickle
from glob import glob
import random

from get_image_paths import get_image_paths
from get_tiny_images import get_tiny_images
from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from visualize import visualize

from nearest_neighbor_classify import nearest_neighbor_classify
from svm_classify import svm_classify




# Step 0: Set up parameters, category list, and image paths.

#For this project, you will need to report performance for three
#combinations of features / classifiers. It is suggested you code them in
#this order, as well:
# 1) Tiny image features and nearest neighbor classifier
# 2) Bag of sift features and nearest neighbor classifier
# 3) Bag of sift features and linear SVM classifier
#The starter code is initialized to 'placeholder' just so that the starter
#code does not crash when run unmodified and you can get a preview of how
#results are presented.

parser = argparse.ArgumentParser()
parser.add_argument('--classifier', help='classifier', type=str, default='dumy_classifier')
args = parser.parse_args()

DATA_PATH = '../Merced/'

#This is the list of categories / directories to use. The categories are
#somewhat sorted by similarity so that the confusion matrix looks more
#structured (indoor and then urban and then rural).

CATEGORIES=os.listdir(DATA_PATH)

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

ABBR_CATEGORIES = [i[:3] for i in CATEGORIES]



CLASSIFIER = args.classifier
# CLASSIFIER = 'support vector machine'

#number of training examples per category to use. Max is 100. For
#simplicity, we assume this is the number of test cases per category, as
#well.

TRAIN_PERCENT=70
VAL_PERCENT=10
TEST_PERCENT=20

def shuffle_lists(list1,list2):
    combined = list(zip(list1, list2))
    random.shuffle(combined)

    # Unzip the combined list back into two lists
    shuffled_list1, shuffled_list2 = zip(*combined)

    # Convert back to lists (optional, since zip returns tuples)
    shuffled_list1 = list(shuffled_list1)
    shuffled_list2 = list(shuffled_list2)

    return shuffled_list1,shuffled_list2

def path_loader(data_path,categories,train_percent,val_percent,test_percent):
    train_image_paths=[]
    val_image_paths=[]
    test_image_paths=[]
    train_labels=[]
    val_labels=[]
    test_labels=[]

    for category in categories:

        image_paths = glob(os.path.join(data_path, category, '*.tif'))
        random.shuffle(image_paths)

        train_size=int(len(image_paths)*train_percent/100)
        val_size=int(len(image_paths)*val_percent/100)
        test_size=int(len(image_paths)*test_percent/100)

        cat_train=image_paths[:train_size]
        cat_val=image_paths[train_size:train_size+val_size]
        cat_test=image_paths[train_size+val_size:]

        train_image_paths.extend(cat_train)
        val_image_paths.extend(cat_val)
        test_image_paths.extend(cat_test)

        train_labels.extend([category]* train_size)
        val_labels.extend([category]* val_size)
        test_labels.extend([category]* test_size)


    train_image_paths,train_labels=shuffle_lists(train_image_paths,train_labels)
    val_image_paths,val_labels=shuffle_lists(val_image_paths,val_labels)
    test_image_paths,test_labels=shuffle_lists(test_image_paths,test_labels)
   

    return train_image_paths, val_image_paths, test_image_paths, train_labels, val_labels, test_labels


def main():

    print("Setting training, val and testing splits")

    if os.path.isfile(f'../store/history-200.pkl'):
    
        with open(f'../store/history-200.pkl', 'rb') as handle:
            history=pickle.load(handle)
            train_image_paths=history['train_image_paths']
            val_image_paths=history['val_image_paths']
            test_image_paths=history['test_image_paths']
            train_labels=history['train_labels']
            val_labels=history['val_labels']
            test_labels=history['test_labels']
        
    else:
    
        train_image_paths, val_image_paths, test_image_paths, train_labels,val_labels, test_labels = \
            path_loader(DATA_PATH, CATEGORIES,TRAIN_PERCENT,VAL_PERCENT,TEST_PERCENT)
    
    # Sizes have been checked to be exact, training gets 1420, val gets 210 and test gets 420



    store={}
    vocab_sizes=[200,300,400,500,600]

    for vocab_size in vocab_sizes:

        print('CREATING VOCAB FOR SIZE -> ',vocab_size)


        if os.path.isfile(f'../store/vocab-{vocab_size}.pkl') is False:
            print('No existing visual word vocabulary found. Computing one from training images\n')
            vocab = build_vocabulary(train_image_paths, vocab_size)
            with open(f'../store/vocab-{vocab_size}.pkl', 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        if os.path.isfile(f'../store/train_image_feats-{vocab_size}.pkl') is False:
            # YOU CODE get_bags_of_sifts.py
            train_image_feats = get_bags_of_sifts(train_image_paths, vocab_size)
            with open(f'../store/train_image_feats-{vocab_size}.pkl', 'wb') as handle:
                pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(f'../store/train_image_feats-{vocab_size}.pkl', 'rb') as handle:
                train_image_feats = pickle.load(handle)
        
        if os.path.isfile(f'../store/val_image_feats-{vocab_size}.pkl') is False:
            # YOU CODE get_bags_of_sifts.py
            val_image_feats = get_bags_of_sifts(val_image_paths, vocab_size)
            
        else:
            with open(f'../store/val_image_feats-{vocab_size}.pkl', 'rb') as handle:
                val_image_feats = pickle.load(handle)

        
        if os.path.isfile(f'../store/test_image_feats-{vocab_size}.pkl') is False:
            test_image_feats  = get_bags_of_sifts(test_image_paths,vocab_size)
            with open(f'../store/test_image_feats-{vocab_size}.pkl', 'wb') as handle:
                pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(f'../store/test_image_feats-{vocab_size}.pkl', 'rb') as handle:
                test_image_feats = pickle.load(handle)

        if CLASSIFIER == 'nearest_neighbor':
            # YOU CODE nearest_neighbor_classify.py
            predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, val_image_feats,CATEGORIES)

        elif CLASSIFIER == 'support_vector_machine':
            # YOU CODE svm_classify.py
            predicted_categories = svm_classify(train_image_feats, train_labels, val_image_feats)

        
        accuracy = float(len([x for x in zip(val_labels,predicted_categories) if x[0]== x[1]]))/float(len(train_labels))
        print("Accuracy of vocab size ",vocab_size ," = ", accuracy)

        history={}
        history['train_image_paths']=train_image_paths
        history['train_image_feats']=train_image_feats
        history['train_labels']=train_labels
        history['val_image_paths']=val_image_paths
        history['val_image_feats']=val_image_feats
        history['val_labels']=val_labels
        history['test_image_paths']=test_image_paths
        history['test_image_feats']=test_image_feats
        history['test_labels']=test_labels
        history['accuracy']=accuracy

        with open(f'../store/history-{vocab_size}.pkl', 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        store[vocab_size]=history

    max_accuracy=0
    best_vocab=0
    for vocab_size in vocab_sizes:

        with open(f'../store/history-{vocab_size}.pkl', 'rb') as handle:
            history = pickle.load(handle)
        if(history['accuracy']>max_accuracy):
            max_accuracy=history['accuracy']
            best_vocab=vocab_size
    
    print('Best vocab size ',best_vocab)

    




   


if __name__ == '__main__':
    main()
