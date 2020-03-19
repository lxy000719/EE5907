"""
K-Nearest Neighbors
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

# To avoid log0
x_train = np.log(x_train+0.1)
x_test = np.log(x_test+0.1)

# K value
K1 = np.arange(1, 11, 1)
K2 = np.arange(15, 105, 5)
K_array = np.concatenate((K1, K2), axis=0)

euclidean_dist_train = np.zeros((len(x_train), len(x_train)))
euclidean_dist_test = np.zeros((len(x_test), len(x_train)))

# Euclidean distance for each data in training data and testing data
for i in range(len(x_train)):
  for j in range(len(x_train)):
    euclidean_dist_train[i][j] = np.sqrt(np.sum((x_train[i,:] - x_train[j,:]) ** 2))

for i in range(len(x_test)):
  for j in range(len(x_train)):
    euclidean_dist_test[i][j] = np.sqrt(np.sum((x_test[i,:] - x_train[j,:]) ** 2))

# Sortting the Euclidean distance in ascending sequence and list the index of each distance    
eudist_sortind_train = np.argsort(euclidean_dist_train)
eudist_sortind_test = np.argsort(euclidean_dist_test)

# Create empty error list for later use
errRate_train = []
errRate_test = []

for l in range(len(K_array)):
    K = K_array[l]

    sort_index_train = eudist_sortind_train[:, : K]
    label_train = y_train[sort_index_train]
    label_train_sque = np.squeeze(label_train, axis = -1)
    sort_index_test = eudist_sortind_test[:, : K]
    label_test = y_train[sort_index_test]
    label_test_sque = np.squeeze(label_test, axis = -1)

    prob_train = []
    prob_test = []
   
    prob_train.append(np.sum(label_train_sque == 1 , axis = 1) / K)
    prob_train.append(np.sum(label_train_sque == 0 , axis = 1) / K)
    prob_train = np.array(prob_train).transpose()
    result_train = []
    for a in range(3065):
        if prob_train[a,0] > prob_train[a,1]:
            result_train.append(1)
        else:
            result_train.append(0)
    result_train = np.array(result_train)
    cm_train = confusion_matrix(result_train, y_train)
    error_rate_train = (cm_train[0,1]+cm_train[1,0])/(cm_train[0,0]+cm_train[0,1]+cm_train[1,0]+cm_train[1,1])
    errRate_train.append(error_rate_train)
    
    prob_test.append(np.sum(label_test_sque == 1 , axis = 1) / K)
    prob_test.append(np.sum(label_test_sque == 0 , axis = 1) / K)
    prob_test = np.array(prob_test).transpose()
    result_test = []
    for b in range(1536):
        if prob_test[b,0] > prob_test[b,1]:
            result_test.append(1)
        else:
            result_test.append(0)
    result_test = np.array(result_test)
    cm_test = confusion_matrix(result_test, y_test)
    error_rate_test = (cm_test[0,1]+cm_test[1,0])/(cm_test[0,0]+cm_test[0,1]+cm_test[1,0]+cm_test[1,1])
    errRate_test.append(error_rate_test)

errRate_train = np.array(errRate_train)
errRate_test = np.array(errRate_test)

plt.plot(K_array, errRate_train, label = 'Training Data')
plt.plot(K_array, errRate_test, label = 'Testing Data')
plt.xlabel('K Values')
plt.ylabel('Error Rate')
plt.title('K-Nearest Neighbors')
plt.legend(loc = 0)
plt.show()

for c in [0, 9, 27]:
    print('K =', K_array[c])
    print('training error:', errRate_train[c])
    print('testing error:', errRate_test[c])
