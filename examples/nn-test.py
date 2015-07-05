# THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python run_mlp.py

import sys


import numpy as np
import theano


sys.path.append('../') # theano folder path.

# from core.datasets import mnist
from core.utils.activation import ReLU, SoftMax
from core.layers import DenseLayer, Convolutional2DLayer, OutputLayer
from core.models import Model



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


	# Create a MODEL.
	model = Model( name="Testing model", verbose=True )
	# Use the model to ADD layers.
	batchSize = 32
	nClasses = 10
	model.addLayer( DenseLayer( inputDim=(batchSize,1,28,28), numUnits=50, activation=ReLU ) )
	model.addLayer( DenseLayer( numUnits=50, activation=ReLU ) )
	model.addLayer( OutputLayer( numUnits=nClasses ) )
	y = theano.tensor.matrix(name="Y", dtype=theano.config.floatX) # theano.tensor.ivector('y')
	model.compile()

	


	# Create a MODEL.
	# model2 = Model( name="Testing model 2", verbose=True )
	# # Use the model to ADD layers.
	# model2.addLayer( Convolutional2DLayer( inputDim=(10,1,28,28), kernels=8, kernelSize=(3,3), kernelStride=1 ) )
	# model2.addLayer( DenseLayer( numUnits=50, activation=ReLU ) )
	# model2.addLayer( OutputLayer( numUnits=nClasses ) )
	# model.compile( y=y )
	# Model internally calls a checkDimensions function on each add to make sure the layer is ok with the previous one.


