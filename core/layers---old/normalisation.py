import numpy as np

import theano

# from ..utils.activation import *

from .base import Layer



__all__ = [
	"NormalisationLayer"
]



class NormalisationLayer(Layer):
	'''
	Standard Normalisation Layer
	'''
	def __init__( self, layerInput, **kwargs ):
		super(HiddenLayer, self).__init__( layerInput=layerInput, name="Normalisation Layer", **kwargs )

		print self.name
		# -------------------- #

	# def get_output_shape( self ):
	# 	raise NotImplementedError
	# 	# -------------------- #


	# -------------------- #

