import numpy as np

import theano



__all__ = [
	"Layer"
]


# Layer base class
class Layer(object):
	'''
	Single base-layer of a 'Neural Network'.
	Basic functions that can be overwritten by specific subclasses.
	'''
	
	def __init__( self, inputDim=None, outputDim=None, name=None, verbose=False ):
		#
		self.inputLayer = None
		self.input = None
		self.output = None
		# Store the dimensions of this layer.
		# They are both always tuples with at least 2-dimensions (number-of-samples, number-of-features)
		self.inputDim = inputDim
		self.outputDim = outputDim
		
		# Initialise the parameters as an empty array.
		self.params = []
		
		# Store the name.
		self.name = name
		self.verbose = verbose
		# -------------------- #


	def _setInputDimensions( self ):
		'''
		To be specified for each specific layer-subclass.
		'''
		if self.inputLayer is not None:
			self.inputDim = self.inputLayer.outputDim
		else:
			raise Exception("\n\t*** ERROR: No inputLayer available!")
		# -------------------- #


	def _calcOutputDimensions( self ):
		'''
		To be specified for each specific layer-subclass.
		'''
		raise NotImplementedError
		# -------------------- #


	def _calcOutput( self ):
		'''
		To be specified for each specific layer-subclass.
		'''
		raise NotImplementedError
		# -------------------- #


	def _initParams( self ):
		'''
		To be specified for each specific layer-subclass.
		'''
		raise NotImplementedError
		# -------------------- #


	# -------------------- #
