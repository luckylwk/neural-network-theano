import numpy as np

import theano

# from ..utils.activation import *

from .base import Layer



__all__ = [
	"PoolingLayer"
]



class PoolingLayer(Layer):
	'''
	Standard Pooling 2D Layer.
	'''
	def __init__( self, layerInput, downSample=(2,2), **kwargs ):
		super(PoolingLayer, self).__init__( layerInput=layerInput, name="Pooling Layer", **kwargs )
		
		# Make sure the dimensions don't cause a problem.
		assert self.inputSize[2] % downSample[0] == 0
		assert self.inputSize[3] % downSample[1] == 0
		self.downSample = downSample
		
		if self.verbose: self.printVerbose()
		# -------------------- #

	
	def fn_get_outputSizeFor( self, inputSize ):
		return ( inputSize[0], inputSize[1], inputSize[2]/self.downSample[0], inputSize[3]/self.downSample[1] )
		# -------------------- #


	def fn_get_outputFor( self, input=None, *args, **kwargs ):
		# Documentation: http://deeplearning.net/software/theano/library/tensor/signal/downsample.html
		# theano.tensor.signal.downsample.max_pool_2d( 
		# 	input=layerInput.fn_get_outputFor( input=input, *args, **kwargs ), 
		# 	ds=self.downSample, 
		# 	ignore_border=True 
		# )
		raise NotImplementedError
		# -------------------- #



	def printVerbose( self ):
		print '\t    --- Initialising {}'.format( self.name )
		print '\t\tDownSample:          {}'.format( self.downSample )
		print '\t\tSampling:            {}'.format( '2D Max Pooling (Theano)' )
		print '\t\tOutput Size:         {}'.format( self.fn_get_outputSizeFor(self.inputSize) )
		# -------------------- #

	# -------------------- #

