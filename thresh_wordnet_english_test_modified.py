# Full test on SVC_2004 dataset

import os
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
from network_architecture import computation_graph
from scipy.misc import imread, imresize
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
os.environ['GLOG_minloglevel'] = '2' 
# import caffe


def get_features(source_path, uid):
	''' 
		Function that runs the trained Autoencoder model and outputs the features learned (latent representation).
		@param
		source_path => The path where the files are present.
		@return => The features encoded by tensorflow.
	'''

	# load the graph to execute:
	myGraph = computation_graph.myGraph

	# Get list of files
	files = os.listdir(source_path)

	height, width, channels = 96, 192, 3

	# create empty numpy array for the data to be collected	
	batch = np.ndarray(shape=(len(files), height, width, channels), dtype=np.float32)

	# fill the empty array with image data
	count = 0
	for f in files:
		img_name = source_path + f
		img = imread(img_name, mode="RGB") # load in RGB format. truncate the alpha channel if it exists
		img = imresize(img, (height, width, channels)) # resize the images to our dimensions.
		batch[count] = img # insert this image in the batch
		count += 1 # increment the counter

	# start tensorflow session and get the features (latent representation)
	with tf.Session(graph=myGraph) as sess:
		# define the path where the model is saved:
		saver = tf.train.Saver() # create instance of the saver module
		model_path = "Models/Model3/"

		# load the trained weights in the graph
		saver.restore(sess, tf.train.latest_checkpoint(model_path))

		# get a handle of the input and the output op.
		inputs = sess.graph.get_tensor_by_name("inputs:0")
		output = sess.graph.get_tensor_by_name("relu1_4:0")				

		features = sess.run(output, feed_dict={inputs: batch})
		
	print('Features extracted.\n')
	features = features.flatten().reshape(len(files), -1)
	return features


def train_svm(train_feats):
    print('Training one-class svm..')
    clf=OneClassSVM(nu=0.01,kernel='linear',gamma=.1)  #'linear'/'poly'/'sigmoid'/'rbf'
    clf.fit(train_feats)
    with open('svm_weights.pickle', 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Model saved')
    
    train_pred = clf.predict(train_feats)
    n_error_train = train_pred[train_pred == -1].size
    print('\nTraining error rate:')
    train_error = n_error_train*1.0/len(train_feats)
    print(train_error)
    return train_error

        
def train(train_path,uid):
    train_feats = get_features(train_path, uid)
    train_error = train_svm(train_feats)
    return train_error

    
def test_svm(test_feats):
    # Load weights
    with open('svm_weights.pickle', 'rb') as handle:
        clf = pickle.load(handle)
    # Test
#    test_pred = clf.predict(test_feats)
#    n_error_test = test_pred[test_pred == -1].size

#    test_error = n_error_test*1.0/len(test_feats)
#    print(test_error)
#    print(test_pred)
#    print(clf.decision_function(test_feats))
    scores = clf.decision_function(test_feats)
#    print "test_scores: " + str(scores)
    test_error = []
    th_range = np.array(range(-10,1,1))*0.1
    for th in th_range:        
        test_pred = np.array([-1 if x < th else 1 for x in scores])
        n_error_test = test_pred[test_pred == -1].size
        test_error.append(n_error_test*1.0/len(test_feats))
    print('Test error rate:')
    print(test_error)    
    return test_error
    
    
def test_positives(test_path,uid):
    print('\nTesting true positives:')
    test_feats = get_features(test_path, uid)
    test_error = test_svm(test_feats)
    return test_error


def test_outliers_svm(outliers_feats):
    # Load weights
    with open('svm_weights.pickle', 'rb') as handle:
        clf = pickle.load(handle)
    # Test
    #outliers_pred = clf.predict(outliers_feats)
    #n_error_outliers = outliers_pred[outliers_pred == 1].size
    scores = clf.decision_function(outliers_feats)
    outliers_error = []
    th_range = np.array(range(-10,1,1))*0.1
    for th in th_range:        
        outliers_pred = np.array([1 if x < th else -1 for x in scores])
        n_error_outliers = outliers_pred[outliers_pred == -1].size
        outliers_error.append(n_error_outliers*1.0/len(outliers_feats))
    print('Outliers error rate:')
    print(outliers_error)
    
    return outliers_error
    
    
def test_outliers(outliers_path,uid):
    print('\nTesting outliers:')
    outliers_feats = get_features(outliers_path, uid)
    outliers_error = test_outliers_svm(outliers_feats)
    return outliers_error


def full_test():
    
    train_error_sum = 0.0
    test_error_sum = 0.0
    outliers_error_sum = 0.0
    data_folder = 'svc2004/user'
    english_user_ids = [2, 4, 6, 8, 10, 11, 12, 13, 15, 18, 20, 22, 24, 25, 26, 28, 30, 32, 34, 35, 37, 38, 39] 
    num_users = len(english_user_ids)
    
    for i in english_user_ids:
        print('\nuser'+`i`+':\n')
        train_path = data_folder+ `i` + '/train/'
        test_path = data_folder+ `i` + '/test/'
        outliers_path = data_folder+ `i` + '/outliers/'
        
        train_error = train(train_path,i)
        test_error = test_positives(test_path,i)
        outliers_error = test_outliers(outliers_path,i)        
        
        train_error_sum = train_error_sum + np.array(train_error)
        test_error_sum = test_error_sum + np.array(test_error)
        outliers_error_sum = outliers_error_sum + np.array(outliers_error)
        #break

    print('Final train error:')    
    print(train_error_sum*1.0/num_users)
    print('Final test error:')    
    print(test_error_sum*1.0/num_users)
    print('Final outliers error:')
    print(outliers_error_sum*1.0/num_users)
    


full_test()




