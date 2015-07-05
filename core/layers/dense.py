import numpy as np

import theano

from ..utils.activation import *
from ..utils import weights
from .base import Layer



__all__ = [
	"DenseLayer"
]



class DenseLayer(Layer):
	'''
	Standard Dense Hidden Layer
	'''
	def __init__( self, numUnits=None, activation=Sigmoid, **kwargs ):
		super(DenseLayer, self).__init__( name="Dense Layer", **kwargs )
		
		self.activation = activation
		self.numUnits = numUnits

		# # Input flattening?
		# if len(self.inputDim) > 2:
		# 	self.inputDim = ( self.inputDim[0], self.inputDim[1]*self.inputDim[2]*self.inputDim[3] )
		# -------------------- #

	
	def _calcOutputDimensions( self ):
		self.outputDim = ( self.inputDim[0], self.numUnits )
		# -------------------- #


	def _initParams( self ):
		self.W = theano.shared( weights.Constant(val=0.0)((self.inputDim[1],self.outputDim[1])), borrow=True )
		self.b = theano.shared( weights.Constant(val=0.0)((self.outputDim[1],)), borrow=True )
		print 'Debug: ', type(self.W), self.W.shape.eval() #, self.W.get_value()[:2]
		self.params = [ self.W, self.b ]
		# -------------------- #


	def _calcOutput( self ):
		# Check the dimensionality of the input. Flatten if needed.
		if self.input.ndim > 2:
			self.input = self.input.flatten(2)
		# Proceed to calculate the layer-output.
		zeta = theano.tensor.dot( self.input, self.W )
		if self.b is not None:
			zeta = zeta + self.b.dimshuffle('x', 0)
		# done, return activations based on zeta.
		self.output = self.activation.fn(zeta)	
		# -------------------- #


	def printVerbose( self ):
		print '\t    --- Initialising {}'.format( self.name )
		print '\t\tActivation function: {}'.format( self.activation.name )
		print '\t\tInput size:          {}'.format( self.inputDim )
		print '\t\tOutput size:         {}'.format( self.outputDim )
		# -------------------- #


	# -------------------- #

