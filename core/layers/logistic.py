import numpy as np

import theano

from ..utils.activation import *

from .base import Layer



__all__ = [
	"LogisticLayer"
]



class LogisticLayer(Layer):
	'''
	Standard Logistic Layer
	'''
	def __init__( self, layerInput, activation=SoftMax, **kwargs ):
		super(HiddenLayer, self).__init__( layerInput=layerInput, name="Logistic Layer", **kwargs )
		
		self.activation = activation
		
		print self.activation.name
		print self.name
		# -------------------- #

	# def get_output_shape( self ):
	# 	raise NotImplementedError
	# 	# -------------------- #


	# -------------------- #

