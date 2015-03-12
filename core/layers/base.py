import numpy as np

import theano



__all__ = [
	"Layer"
]


# Layer base class
class Layer(object):
	'''
	Single layer of a neural network.
	Basic functions that can be overwritten by specific subclasses.
	'''
	def __init__( self, layerInput, name=None, verbose=False ):
		# If the input is a tuple this layer is the first in the network.
		if isinstance(layerInput, tuple):
			self.input_shape = layerInput
			self.input_layer = None
		# If the input is a layer, get its output-size.
		else:
			self.input_shape = layerInput.get_output_shape()
			self.input_layer = layerInput
		# Store the name.
		self.name = name
		# -------------------- #

	def get_output_shape( self ):
		raise NotImplementedError
		# -------------------- #


	# -------------------- #
