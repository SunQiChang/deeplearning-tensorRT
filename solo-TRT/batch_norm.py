import numpy as np
def pure_batch_norm(X,gamma,beta,eps = 1e-5):
    assert len(X.shape) in (2,4)
    # fully connected neural network or Convolutional Neural Network
    if len(X.shape) == 2:
        # fully connected neural network for every feature
        # batch_size * feature
        mean = X.mean(axis = 0)
        variance = ((X-mean)**2).mean(axis = 0)
    else:
        # calculate mean and variance for every channel
        # batch_size * channel * height * width
        mean = X.mean(axis=(0,2,3),keepdims = True)
        variance = ((X-mean)**2).mean(axis=(0,2,3),keepdims = True)

    print('mean:{} variance:{}'.format(mean.shape, variance.shape))
    # batchnorm
    X_hat = (X-mean)/np.sqrt(variance+eps)
    # scale and shift 
    # print('mean.shape:{} mean:{}'.format(mean.shape, mean))
    return gamma.reshape(mean.shape)*X_hat+beta.reshape(mean.shape)

A = np.arange(12).reshape((3,4))
print(A)
gamma_ = np.ones(shape=(4))
beta_ = np.ones(shape=(4))
res =pure_batch_norm(A,gamma = gamma_,beta = beta_)
print (res)
print('====================================================')
#2345
A = np.arange(120).reshape((2,3,4,5))
gamma_ = np.ones(shape=(3))
beta_ = np.ones(shape=(3))
res = pure_batch_norm(A, gamma = gamma_,beta = beta_)
print (res)
