import numpy as np

import theano

from ..utils.activation import *

from .base import Layer



__all__ = [
	"HiddenLayer"
]



class HiddenLayer(Layer):
	'''
	Standard Hidden Layer
	'''
	def __init__( self, layerInput, layerSize, activation=Sigmoid, **kwargs ):
		super(HiddenLayer, self).__init__( layerInput=layerInput, name="Hidden Layer", **kwargs )
		
		self.activation = activation
		self.layerSize = layerSize

		if self.verbose: self.printVerbose()
		# -------------------- #

	def fn_get_outputSizeFor( self, inputSize ):
		return ( inputSize[0], self.layerSize )
		# -------------------- #

	def printVerbose( self ):
		print '\t    --- Initialising {}'.format( self.name )
		print '\t\tActivation fn.:      {}'.format( self.activation.name )
		print '\t\tInput size:          {}'.format( self.input_shape )



	# -------------------- #

