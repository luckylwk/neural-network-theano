# THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python run_mlp.py

import sys


import numpy as np
import theano


sys.path.append('../') # theano folder path.

from core.datasets import mnist
from core.utils.activation import ReLU
from core.layers import InputLayer, HiddenLayer, LogisticLayer, Convolutional2DLayer, PoolingLayer



if __name__ == '__main__':

	# Set a random seed.
	random_seed = 1234

	# # LOAD the DATA.
	# datasets = mnist.fn_T_load_data_MNIST( path_to_file='../../../_DATA/mnist.pkl.gz' )
	# # print type(datasets[0][0]) # <class 'theano.tensor.sharedvar.TensorSharedVariable'> - datasets[0][0].shape.eval()

	# ...
	rng = np.random.RandomState(random_seed)
	# X = theano.tensor.matrix('x') # <class 'theano.tensor.var.TensorVariable'>
	# y = theano.tensor.ivector('y')


	print '\n'
	NN_0 = InputLayer( layerInput=( 100, 1*28*28 ), verbose=True )
	NN_1 = HiddenLayer( layerInput=NN_0, activation=ReLU, layerSize=1024, verbose=True )
	NN_2 = HiddenLayer( layerInput=NN_1, activation=ReLU, layerSize=512, verbose=True )
	NN_X = LogisticLayer( layerInput=NN_2, layerSize=10, verbose=True )



	print '\n\n'
	CNN_0 = InputLayer( layerInput=( 100, 1, 28, 28 ), verbose=True )
	CNN_1 = Convolutional2DLayer( layerInput=CNN_0, kernels=8, kernelSize=(7,7), kernelStride=1, verbose=True )
	CNN_2 = PoolingLayer( layerInput=CNN_1, downSample=(2,2), verbose=True )
	# normalisation layer

	# Create the MODEL.

