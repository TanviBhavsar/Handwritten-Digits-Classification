import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle
counter = 0
train_label_real = 0
validation_label_real = 0
import time
start_time = time.time()
"""
#Returns the unique columns in a Matrix A
#Uses lexsort to sort the columns then gets the unique ones
#returns the same and the order which is used to get columns for the test data
#Inspired by the post:
#http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array
"""
def unique_cols(a):
    a = a.T
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    return (a[ui]).T, ui, order
    
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    retVal=0
    retVal = 1/(1+np.exp(-z))
    return retVal
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    shp = mat.get("train0").shape
    
    num_input= 60000

    #Number of output classes.
    k = 10
    #Feature Vector size
    d = shp[1]
    #Extract Data
    total_data = []
    total_label = []
    total_number_label = []
    #Load the test data 
    #Real Labels are stored as 0,1,2,3..9
    #labels are stored as 1x10 vectors
    #eg: [0,0,0,0,1,0,0,0,0,0] indicates a 4
    for i in range(0,10):
        row = mat.get("train" + str(i))
        total_data.append(row)
        for j in range(0,row.shape[0]):
            label_arr = np.zeros(10)
            label_arr[i] = 1
            total_label.append(label_arr)
        real_labels = np.ones((row.shape[0],1))
        real_labels*=i
        total_number_label.append(real_labels)
    total_data = np.vstack(total_data)
    total_label = np.vstack(total_label)
    total_number_label = np.vstack(total_number_label)
    

    total_data=total_data.astype(float)
    #Feature Reduction (Getting unique cols)
    total_data, ui, order = unique_cols(total_data)      
    #Normalization (divide by the Max. Value in the train set)
    maxValue = np.amax(total_data)
    total_data = np.divide(total_data,maxValue)
    
    #Separate the data into train and validate sets
    num_train = 50000
    num_validate = 10000
    perms = range(total_data.shape[0])
    aperm = np.random.permutation(perms)
    train_data = total_data[aperm[0:num_train],:]
    validation_data  = total_data[aperm[num_train:],:]
    train_label = total_label[aperm[0:num_train],:]
    validation_label  = total_label[aperm[num_train:],:]
    global validation_label_real
    global train_label_real
    train_label_real = total_number_label[aperm[0:num_train],:]
    validation_label_real = total_number_label[aperm[num_train:],:]
    
    test_data = []
    test_label = []
    #Load the test data
    for i in range(0,10):
        row = mat.get("test" + str(i))
        test_data.append(row)
        real_labels = np.ones((row.shape[0],1))
        real_labels*=i
        test_label.append(real_labels)
    test_data = np.vstack(test_data)
    test_label = np.vstack(test_label)
    #Getting Unique Cols
    test_data = test_data.T
    test_data = test_data[order]
    test_data = (test_data[ui]).T
    test_data = test_data.astype(float)   
    #Normalize test data
    test_data = np.divide(test_data,maxValue)

    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
   
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
   
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    
    train_shape = training_data.shape
    X = training_data
   
    #Add bias D+1
    X = np.ones((train_shape[0],n_input+1))
    X[:,:-1] = training_data
    #A=W1T.X
    A = np.dot(w1,X.T)
    A = A.T
    Z = sigmoid(A)
    #Add Bias M+1
    Z_withBias = np.ones((Z.shape[0],n_hidden+1))
    Z_withBias[:,:-1] = Z
    #B = W2T.Z
    B = np.dot(w2,Z_withBias.T)
    O = sigmoid(B)
    O = O.T
    Y = training_label
    #Delta: Error between true label and predicted label
    delta = O - Y
    #Jacobian W1W2
    JW1W2 = sum(sum(((np.multiply(Y,np.log(O)))+ np.multiply((1-Y),(np.log(1-O))))))
    JW1W2 = (-1)*(JW1W2/train_shape[0])
    
    #Adding regularization 
    wSum = sum(sum(w1**2)) + sum(sum(w2**2))   
    wSum = (lambdaval*wSum)
    wSum = wSum/(2*train_shape[0])
    #adding regularization
    JTildaW1W2 = JW1W2 + wSum
    obj_val = JTildaW1W2

    #GradW2 andGradW1
    gradW2 = np.dot(delta.T,Z_withBias)
    gradW1 = (1-Z_withBias)*Z_withBias
    gradW1 = gradW1 * np.dot(delta,w2)
    
    gradW1 = np.dot(gradW1.T,X)
    gradW1 = gradW1[:-1,:]
    gradW2 = (gradW2 + (lambdaval*w2))/train_shape[0]
    gradW1 = (gradW1 + (lambdaval*w1))/train_shape[0]
    
    grad_w1 = gradW1
    grad_w2 = gradW2
    
    
    #Your code here
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    #Same as the feedForward part of nnObjFunction
    labels = []
    O = []
    train_shape = data.shape
    X = data
    X = np.ones((train_shape[0],n_input+1))
    X[:,:-1] = data
    A = np.dot(w1,X.T)
    A = A.T
    Z = sigmoid(A)
    Z_withBias = np.ones((Z.shape[0],n_hidden+1))
    Z_withBias[:,:-1] = Z
    B = np.dot(w2,Z_withBias.T)
    O = sigmoid(B)
    O = O.T
    Ol=[]
    #Convert the labels from 1x10 to 1 numeric value
    for i in range(0,data.shape[0]):
        row = O[i]
        Ol.append(np.argmax(row))  
    labels = np.vstack(Ol)
    return labels
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.5


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)
train_label = train_label_real

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset
validation_label = validation_label_real
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
print("\n --- %s seconds ---" % (time.time() - start_time))
# Code to measure time taken for execution is taken from this 
#http://stackoverflow.com/questions/8889083/how-to-time-execution-time-of-a-batch-of-code-in-python

pickleFileP = open('params.pickle','wb')
pickle.dump([n_hidden,w1,w2,lambdaval],pickleFileP)
pickleFileP.close()