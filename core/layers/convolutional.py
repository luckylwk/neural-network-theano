import numpy as np

import theano

from ..utils.activation import *

from .base import Layer



__all__ = [
	"Convolutional2DLayer"
]



class Convolutional2DLayer(Layer):
	'''
	Standard Convolutional 2D Layer
	'''
	def __init__( self, layerInput, kernels, kernelSize, kernelStride=(1,1), **kwargs ):
		super(Convolutional2DLayer, self).__init__( layerInput=layerInput, name="Convolutional 2D Layer", **kwargs )

		self.kernels = kernels # Int
		self.kernelSize = kernelSize # Tuple
		self.kernelStride = kernelStride # Tuple

		if self.verbose: self.printVerbose()
		# -------------------- #

	def fn_get_outputSizeFor( self, inputSize ):
		'''
		Only allows for standard border mode.
		'''
		cols = ( (inputSize[2] - self.kernelSize[0]) / self.kernelStride[0] + 1 )
		rows = ( (inputSize[3] - self.kernelSize[1]) / self.kernelStride[1] + 1 )
		return ( inputSize[0], self.kernels, cols, rows )
		# -------------------- #




	def printVerbose( self ):
		print '\t    --- Initialising {}'.format( self.name )
		print '\t\t# Kernels:           {}'.format( self.kernels )
		print '\t\tKernel Size:         {}'.format( self.kernelSize )
		print '\t\tKernel Stride:       {}'.format( self.kernelStride )
		print '\t\tOutput Size:         {}'.format( self.fn_get_outputSizeFor(self.input_shape) )
		# -------------------- #

		# # convolve input feature maps with filters
		# # Docs: http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d
		# # <class 'theano.tensor.var.TensorVariable'>
		# conv_out = theano.tensor.nnet.conv.conv2d( 
		# 	input=layer_input, 
		# 	filters=self.W, 
		# 	filter_shape=self.filter, 
		# 	image_shape=input_dim, 
		# 	border_mode='valid' # what does this do exactly?
		# )
	# -------------------- #

