import numpy as np

import theano

from .base import Layer



__all__ = [
	"InputLayer"
]



class InputLayer(Layer):
	'''
	Input Layer.
	This layer will hold the dimensions or data-tensor that the model takes as an input.
	'''
	def __init__( self, layerInput, **kwargs ):
		super(InputLayer, self).__init__( layerInput=layerInput, name="Input Layer", **kwargs )
		
		#
		if self.verbose: self.printVerbose()
		# -------------------- #


	def printVerbose( self ):
		'''
		Function to print layer information. Usually overwritten per subclass.
		'''
		print '\t    --- Initialising {}'.format( self.name )
		print '\t\tInput size:          {}'.format( self.inputSize )
		# -------------------- #

	# -------------------- #

