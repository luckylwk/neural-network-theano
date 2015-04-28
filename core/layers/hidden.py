import numpy as np

import theano

from ..utils.activation import *
from ..utils import weights
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

		self.W = self._set_params( init=weights.Uniform(), shape=(self.inputSize[1], self.layerSize), name="W" )
		self.b = np.zeros( (1,self.layerSize) )
		# print self.W.shape.eval() # self.W.get_value()

		if self.verbose: self.printVerbose()
		# -------------------- #

	
	def fn_get_outputSizeFor( self, inputSize ):
		return ( inputSize[0], self.layerSize )
		# -------------------- #

	
	def fn_get_outputFor( self, input=None, *args, **kwargs ):
		if input.ndim > 2:
			raise RuntimeError("need to flatten input")
		else:
			zeta = theano.tensor.dot( input, self.W )
			if self.b is not None:
				zeta = zeta + self.b.dimshuffle('x', 0)
			# done, return activations based on zeta.
			return self.activation.fn(zeta)

		# raise NotImplementedError
		# if input.ndim > 2:
		# 	# if the input has more than two dimensions, flatten it into a
		# 	# batch of feature vectors.
		# 	input = input.flatten(2)
		# -------------------- #


	def printVerbose( self ):
		print '\t    --- Initialising {}'.format( self.name )
		print '\t\tActivation fn.:      {}'.format( self.activation.name )
		print '\t\tInput size:          {}'.format( self.inputSize )
		print '\t\tOutput size:         {}'.format( self.fn_get_outputSizeFor(self.inputSize) )
		# -------------------- #


	# -------------------- #

