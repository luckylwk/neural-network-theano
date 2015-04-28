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
			self.inputSize = layerInput
			self.inputLayer = None
		# If the input is a layer, get its output-size.
		else:
			self.inputSize = layerInput._get_outputSize()
			self.inputLayer = layerInput
		# Store the name.
		self.name = name
		self.verbose = verbose
		# -------------------- #


	def _get_outputSize( self ):
		return self.fn_get_outputSizeFor(self.inputSize)
		# -------------------- #


	def fn_get_outputSizeFor( self, inputSize ):
		'''
		Default return is the its own input-size.
		This assumes the layer does not transform the input.
		'''
		return inputSize
		# -------------------- #


	def _get_output( self, input=None, *args, **kwargs ):
		if self.inputLayer == None:
			raise RuntimeError('You cannot get output for the input-layer.')
		else:
			thisInput = self.inputLayer.get_output( input=input, *args, **kwargs )
			return self.fn_get_outputFor( input=thisInput, *args, **kwargs )
		# -------------------- #


	def fn_get_outputFor( self, input=None, *args, **kwargs ):
		'''
		To be specified for each specific layer-subclass.
		'''
		raise NotImplementedError
		# -------------------- #


	def _set_params( self, init, shape, name ):
		'''
		Function to set the layer parameters.
		'''
		if hasattr(init, '__call__'):
			arr = init(shape)
			if not isinstance(arr, np.ndarray):
				raise RuntimeError("base._set_params: needs to be numpy array")

			return theano.shared( arr, name=name, borrow=True )
		# -------------------- #


	def printVerbose( self ):
		'''
		Function to print layer information. Usually overwritten per layer-subclass.
		'''
		print '\t    --- Initialising {}'.format( self.name )
		# -------------------- #

	# -------------------- #
