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
	def __init__( self, layerInput, n_in, n_out, activation=Sigmoid, W_in=None, b_in=None, use_bias=True, **kwargs ):
		super(HiddenLayer, self).__init__( layerInput=layerInput, name="Hidden Layer", **kwargs )
		
		self.activation = activation
		
		print self.activation.name
		print self.name
		# -------------------- #

	# def get_output_shape( self ):
	# 	raise NotImplementedError
	# 	# -------------------- #


	# -------------------- #
