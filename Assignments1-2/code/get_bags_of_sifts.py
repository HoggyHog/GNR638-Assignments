from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time
import pdb
from tqdm import tqdm

def get_bags_of_sifts(image_paths,vocab_size):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
                                                                    
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    
    # You will want to construct SIFT features here in the same way you        #
    # did in build_vocabulary.m (except for possibly changing the sampling     #
    # rate) and then assign each local feature to its nearest cluster center   #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''
    
    with open(f'vocab-{vocab_size}.pkl', 'rb') as handle:
        vocab = pickle.load(handle)
    
    image_feats = []
    
    start_time = time()
    print("Construct bags of sifts...")
    
    for path in tqdm(image_paths):
        img=Image.open(path)
        img = img.convert("L")
        img = np.asarray(img,dtype='float32')
        frames, descriptors = dsift(img, step=[1,1], fast=True)
        dist = distance.cdist(vocab, descriptors, metric='euclidean')
        idx = np.argmin(dist, axis=0)
        hist, bin_edges = np.histogram(idx, bins=len(vocab))
        hist_norm = [float(i)/sum(hist) for i in hist]
        
        image_feats.append(hist_norm)  #going to be a 400 long vector
        
    image_feats = np.asarray(image_feats)  # going to be (N,400)
    
    end_time = time()
    print("It takes ",(end_time - start_time), " to construct bags of sifts.")
    
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    
    return image_feats