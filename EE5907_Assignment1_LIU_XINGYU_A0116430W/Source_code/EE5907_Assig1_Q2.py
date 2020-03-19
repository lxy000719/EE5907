"""
Gaussian Naive Bayes
"""

import numpy as np
import scipy.io
from sklearn.metrics import confusion_matrix

# Get the train and test data
dataset = scipy.io.loadmat('spamData.mat')
x_train = dataset['Xtrain']
y_train = dataset['ytrain']
x_test = dataset['Xtest']
y_test = dataset['ytest']

# To avoid log0
x_train = np.log(x_train+0.1)
x_test = np.log(x_test+0.1)

y_train_flatten = y_train.flatten()

# Gaussian Naive Bayes
def gaussian_naive_bayes(test_data,test_result):
    pred_y1 = list()
    pred_y0 = list()
# Total number of results y=1 in y_train
    N1 = np.sum(y_train)
# Total number of results of y_train 
    N = len(y_train)
# Maximum Likelihood
    ml = N1/N
# Get the predication arry for result y=1
# Log of Maximum Likelihood
    log_ml_y1 = np.log(ml)
# Get all the x train data with result y=1
    x_train_class_y1 = x_train[y_train_flatten == 1]
# ML estimation of mean and variance
    mean_y1 = np.sum(x_train_class_y1,axis=0)/len(x_train_class_y1)
    var_y1 = np.sum((x_train_class_y1-mean_y1)**2,axis=0)/len(x_train_class_y1)
    px_y1 = (np.e**(-0.5*((test_data-mean_y1)**2)/var_y1))/((2*np.pi*var_y1)**0.5)
    px_y1_log = np.log(px_y1)
    pred_y1.append(log_ml_y1+np.sum(px_y1_log,axis=1))
    pred_y1_array = np.array(pred_y1).transpose()

# Get the predication arry for result y=1
# Log of Maximum Likelihood
    log_ml_y0 = np.log(1-ml)
# Get all the x train data with result y=1
    x_train_class_y0 = x_train[y_train_flatten == 0]
# ML estimation of mean and variance
    mean_y0 = np.sum(x_train_class_y0,axis=0)/len(x_train_class_y0)
    var_y0 = np.sum((x_train_class_y0-mean_y0)**2,axis=0)/len(x_train_class_y0)
    px_y0 = (np.e**(-0.5*((test_data-mean_y0)**2)/var_y0))/((2*np.pi*var_y0)**0.5)
    px_y0_log = np.log(px_y0)
    pred_y0.append(log_ml_y0+np.sum(px_y0_log,axis=1))
    pred_y0_array = np.array(pred_y0).transpose()
    
    y_pred_list = list()

    k=0
    for i in range(len(test_data)):
        if pred_y1_array[i,k]>pred_y0_array[i,k]:
            y_pred_list.append(1)        
        else:
            y_pred_list.append(0)
    k=k+1
# Transform the prediction list to array
    y_pred = np.array(y_pred_list)
# Calculate the error rate
    cm = confusion_matrix(test_result, y_pred)
    error_rate = (cm[0,1]+cm[1,0])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
    return error_rate

result_training = gaussian_naive_bayes(x_train,y_train)
print('Error rate for training data is: ',result_training)
result_testing = gaussian_naive_bayes(x_test,y_test)
print('Error rate for testing data is: ',result_testing)