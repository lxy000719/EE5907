"""
Beta-binomial Naive Bayes
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.metrics import confusion_matrix

# Get the train and test data
dataset = scipy.io.loadmat('spamData.mat')
x_train = dataset['Xtrain']
y_train = dataset['ytrain']
x_test = dataset['Xtest']
y_test = dataset['ytest']

# Data processing, feature binarize
x_train = np.where(x_train>0, 1, 0)
x_test = np.where(x_test>0, 1, 0)

y_train_flatten = y_train.flatten()

# Naive Bayes for input alpha and training data and comparison with testing data
def naive_bayes(alpha,test_data,test_result):
    pred_y1 = list()
    pred_y0 = list()
    beta = alpha
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
    n1_y1 = np.sum(x_train_class_y1, axis=0)
    n_y1 = len(x_train_class_y1)
# Posterior predictive distribution (N1+a)/(N+a+b) that feature x is 1
    posterior_y1 = (n1_y1+alpha) / (n_y1+alpha+beta)
# Posterior prediction array for all 57 features of x=0 and x=1
    p_y1 = (test_data==1)*posterior_y1 + (test_data==0)*(1-posterior_y1)
    p_y1_log = np.log(p_y1)
    pred_y1.append(log_ml_y1+np.sum(p_y1_log,axis=1))
    pred_y1_array = np.array(pred_y1).transpose()

# Get the predication arry for class result y=0
# Log of Maximum Likelihood
    log_ml_y0 = np.log(1-ml)
# Get all the x train data with result y=0
    x_train_class_y0 = x_train[y_train_flatten == 0]
    n1_y0 = np.sum(x_train_class_y0, axis=0)
    n_y0 = len(x_train_class_y0)
# Posterior predictive distribution (N1+a)/(N+a+b) that feature x is 1
    posterior_y0 = (n1_y0+alpha) / (n_y0+alpha+beta)
# Posterior prediction array for all 57 features of x=0 and x=1
    p_y0 = (test_data==1)*posterior_y0 + (test_data==0)*(1-posterior_y0)
    p_y0_log = np.log(p_y0)
    pred_y0.append(log_ml_y0+np.sum(p_y0_log,axis=1))
    pred_y0_array = np.array(pred_y0).transpose()

    y_pred_list = list()

# Get the y predication list by comparing the values in 2 result array of pred_y1_array and pred_y0_array
# for all testing data
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

error_list_xtrain = list()
error_list_xtest = list()
alpha_list_xtrain = list()
alpha_list_xtest = list()


# Get the error rate for each alpha from 0 to 100 for training data
for alpha_train in np.linspace(0,100,num=201):
    error_list_xtrain.append(naive_bayes(alpha_train,x_train,y_train))
    alpha_list_xtrain.append(alpha_train)
    
# Get the error rate for each alpha from 0 to 100 for testing data
for alpha_test in np.linspace(0,100,num=201):
    error_list_xtest.append(naive_bayes(alpha_test,x_test,y_test))
    alpha_list_xtest.append(alpha_test)

# Transform the error rate list into array
error_array_xtrain = np.array(error_list_xtrain)
error_array_xtest = np.array(error_list_xtest)
alpha_array_xtrain = np.array(alpha_list_xtrain)
alpha_array_xtest = np.array(alpha_list_xtest)

# Plot the error rate versus different alpha from 0 to 100 for training and testing data
plt.figure()
plt.plot(alpha_array_xtrain,error_array_xtrain,label='Error rate for training data')
plt.plot(alpha_array_xtest,error_array_xtest,label='Error rate for testing data')
plt.xlabel('Alpha')
plt.ylabel('Error Rate')
plt.title('Beta-binomial Naive Bayes ')
plt.legend()
plt.show()

# Print the error rate of alpha 1, 10 and 100 for training and testing data
print('Training error rates for alpha 1 is ',error_array_xtrain[2*1])
print('Training error rates for alpha 10 is ',error_array_xtrain[2*10])
print('Training error rates for alpha 100 is ',error_array_xtrain[2*100])
print('Testing error rates for alpha 1 is ',error_array_xtest[2*1])
print('Testing error rates for alpha 10 is ',error_array_xtest[2*10])
print('Testing error rates for alpha 100 is ',error_array_xtest[2*100])

