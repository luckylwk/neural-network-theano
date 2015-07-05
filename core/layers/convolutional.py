import numpy as np

import theano

from ..utils.activation import *
from ..utils import weights
from .base import Layer



__all__ = [
	"Convolutional2DLayer"
]



class Convolutional2DLayer(Layer):
	'''
	Standard Convolutional 2D Layer
	'''
	
	def __init__( self, kernels=8, kernelSize=(3,3), kernelStride=1, **kwargs ):
		super(Convolutional2DLayer, self).__init__( name="Convolutional 2D Layer", **kwargs )

		self.kernels = kernels # Int
		self.kernelSize = kernelSize # Tuple
		self.kernelStride = kernelStride # Int
		self.filter = ( kernels, kernelStride, kernelSize[0], kernelSize[1] )

		if self.verbose: self.printVerbose()
		# -------------------- #


	def _calcOutputDimensions( self ):
		cols = ( (self.inputDim[2] - self.kernelSize[0]) / self.kernelStride + 1 )
		rows = ( (self.inputDim[3] - self.kernelSize[1]) / self.kernelStride + 1 )
		self.outputDim = ( self.inputDim[0], self.kernels, cols, rows )
		# -------------------- #


	def _initParams( self ):
		self.W = theano.shared( weights.Constant(val=0.0)(self.filter), borrow=True )
		self.b = theano.shared( weights.Constant(val=0.0)((self.filter[0],)), borrow=True )
		print 'Debug: ', type(self.W), self.W.shape.eval() #, self.W.get_value()[:2]
		self.params = [ self.W, self.b ]
		# -------------------- #


	def _calcOutput( self ):
		# Documentation: http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d
		self.output = theano.tensor.nnet.conv.conv2d( 
			input=self.input,
			filters=self.W,
			filter_shape=(self.kernels, self.kernelStride) + self.kernelSize,
			image_shape=self.inputDim,
			border_mode='valid'
		)
		# -------------------- #


	def printVerbose( self ):
		print '\t    --- Initialising {}'.format( self.name )
		print '\t\t# Kernels:           {}'.format( self.kernels )
		print '\t\tKernel Size:         {}'.format( self.kernelSize )
		print '\t\tKernel Stride:       {}'.format( self.kernelStride )
		print '\t\tInput Size:          {}'.format( self.inputDim )
		print '\t\tOutput Size:         {}'.format( self.outputDim )
		# -------------------- #

		
	# -------------------- #

