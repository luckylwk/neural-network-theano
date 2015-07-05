import numpy as np

import theano

from ..utils.activation import *
from ..utils import weights
from .base import Layer



__all__ = [
	"LogisticLayer"
]



class LogisticLayer(Layer):
	'''
	Standard Logistic Layer
	'''
	def __init__( self, layerInput, layerSize, activation=SoftMax, **kwargs ):
		super(LogisticLayer, self).__init__( layerInput=layerInput, name="Logistic Output Layer", **kwargs )
		
		self.activation = activation
		self.layerSize = layerSize

		self.W = self._set_params( init=weights.Uniform(), shape=(self.inputSize[1], self.layerSize), name="W" )
		self.b = np.zeros( (1,self.layerSize) )
		
		if self.verbose: self.printVerbose()
		# -------------------- #

	
	def fn_get_outputSizeFor( self, inputSize ):
		return ( inputSize[0], self.layerSize )
		# -------------------- #


	def fn_get_outputFor( self, input=None, *args, **kwargs ):
		return activation.fn( theano.tensor.dot( input, self.W ) + self.b )
		# -------------------- #


	def printVerbose( self ):
		print '\t    --- Initialising {}'.format( self.name )
		print '\t\tActivation fn.:      {}'.format( self.activation.name )
		print '\t\tInput size:          {}'.format( self.inputSize )
		print '\t\tOutput size:         {}'.format( self.fn_get_outputSizeFor(self.inputSize) )
		# -------------------- #

	# -------------------- #

