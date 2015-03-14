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
			self.input_shape = layerInput._get_outputSize()
			self.input_layer = layerInput
		# Store the name.
		self.name = name
		self.verbose = verbose
		# -------------------- #

	def _get_outputSize( self ):
		return self.fn_get_outputSizeFor(self.input_shape)
		# -------------------- #

	def fn_get_outputSizeFor( self, inputSize ):
		'''
		Default return is the its own input-size.
		This assumes the layer does not transform the input.
		'''
		return inputSize
		# -------------------- #

	def printVerbose( self ):
		'''
		Function to print layer information. Usually overwritten per subclass.
		'''
		print '\t    --- Initialising {}'.format( self.name )
		# -------------------- #

	# -------------------- #
