import numpy as np

import theano

# from ..utils.activation import *

from .base import Layer



__all__ = [
	"SamplingLayer"
]



class SamplingLayer(Layer):
	'''
	Standard Sampling Layer
	'''
	def __init__( self, layerInput, **kwargs ):
		super(HiddenLayer, self).__init__( layerInput=layerInput, name="Sampling Layer", **kwargs )
		
		print self.name
		# -------------------- #

	# def get_output_shape( self ):
	# 	raise NotImplementedError
	# 	# -------------------- #


	# -------------------- #

