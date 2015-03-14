# THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python run_mlp.py

import sys


import numpy as np
import theano


sys.path.append('../') # theano folder path.

from core.datasets import mnist
from core.layers import InputLayer, HiddenLayer, Convolutional2DLayer, PoolingLayer






if __name__ == '__main__':

	# Set a random seed.
	random_seed = 1234

	# LOAD the DATA.
	datasets = mnist.fn_T_load_data_MNIST( path_to_file='../../../_DATA/mnist.pkl.gz' )
	# print type(datasets[0][0]) # <class 'theano.tensor.sharedvar.TensorSharedVariable'> - datasets[0][0].shape.eval()

	# ...
	rng = np.random.RandomState(random_seed)
	X = theano.tensor.matrix('x') # <class 'theano.tensor.var.TensorVariable'>
	y = theano.tensor.ivector('y')

	# Create the LAYERS.
	# Input layer.
	LAYER_0 = InputLayer( layerInput=( 100, 1*28*28 ), verbose=True )
	# Hidden layer, dense
	LAYER_1 = HiddenLayer( layerInput=LAYER_0, layerSize=1024, verbose=True )
	print LAYER_1.input_layer
	# Hidden layer, dense
	LAYER_2 = HiddenLayer( layerInput=LAYER_1, layerSize=1024, verbose=True )
	print LAYER_2.input_layer
	# Output layer.

	# Create the MODEL.


	CNN_0 = InputLayer( layerInput=( 100, 1, 28, 28 ), verbose=True )
	CNN_1 = Convolutional2DLayer( layerInput=CNN_0, kernels=8, kernelSize=(7,7), kernelStride=(1,1), verbose=True )
	print CNN_1.input_layer
	CNN_2 = PoolingLayer( layerInput=CNN_1, downSample=(2,2), verbose=True )
	print CNN_2.input_layer
