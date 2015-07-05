import numpy as np

import theano


__all__ = [
	"Model"
]


# Layer base class
class Model(object):
	'''
	Standard Model
	Basic functions that can be overwritten by specific subclasses.
	'''

	def __init__( self, name=None, verbose=False ):
		# Model initialisation function.
		self.name = name
		self.verbose = verbose
		if self.verbose: print '\t*** INITIALISING MODEL {}'.format( self.name )
		# Stack of layers is initialised empty.
		self.layers = []
		# The parameter vector is initialized empty.
		self.params = []
		# Allow for constraints.
		# self.regularizers = [] # same size as params?
		# self.constraints = [] # same size as params?
		# -------------------- #


	def addLayer( self, layer=None ):
		# Function to add a layer to the model.
		if layer is not None:
			self.layers.append(layer)
			# If this is not the first layer, connect it with the previous one.
			if len(self.layers) > 1:
				# Connect the last added layer to the previous one.
				self.layers[-1].inputLayer = self.layers[-2]
				# Obtain the input-dimensions for the last added layer from its previous one.
				self.layers[-1]._setInputDimensions()
				# Set the input of this layer.
				self.layers[-1].input = self.layers[-2].output
			else:
				# This is the first layer of the network. It therefore becomes the
				# input-layer. Initialize a tensor-variable.
				self.layers[0].input = theano.tensor.tensor4(name="X", dtype=theano.config.floatX)
			# Calculate the output dimensions for that layer now that we know the input dimensions.
			self.layers[-1]._calcOutputDimensions()
			# Initialize the weights.
			self.layers[-1]._initParams()
			# We have now connected the layers, set the dimensions accordingly and initialized the parameters.
			# Now connect the output-to-input of the layers.
			self.layers[-1]._calcOutput()
			#
			if self.verbose: self.layers[-1].printVerbose()
		else:
			raise Exception("\n\t*** ERROR: You need to supply a valid layer to add!")
		# -------------------- #


	def compile( self ):
		#theano.tensor.argmax( self.p_y_given_x, axis=1 )
		self.y_pred = self.layers[-1].output
		self.cost = self.layers[-1]._calcLoss
		self.errors = self.calcErrors
		# put all parameters in the params vector.
		self.params = [ param for layer in self.layers for param in layer.params ]
		print 'Number of param. matrices: ', len(self.params)
		# -------------------- #


	def calcErrors( self, y ):
		# Check for dimensions.
		print y.ndim, y_pred.ndim
		if y.ndim != self.y_pred.ndim:
			raise TypeError( 'y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type) )
		# check if y is of the correct datatype
		if y.dtype.startswith('int'):
			# the T.neq operator returns a vector of 0s and 1s, where 1 represents a mistake in prediction
			return theano.tensor.mean( theano.tensor.neq( self.y_pred, y ) )
		else:
			raise NotImplementedError()
		# -------------------- #


	def train( self, Xval, yval ):
		# Set the parameters.
		# Set training model
		# Set CV model
		# Set learning-rate decay.
		# Epochs of training.
		raise NotImplementedError
		# -------------------- #
		

	def fit( self ):
		raise NotImplementedError


	def predict( self ):
		raise NotImplementedError


	def predict_proba( self ):
		raise NotImplementedError


	def score( self ):
		raise NotImplementedError


