
import numpy as np

import theano




# def create_weights_uniform( rng, n_in, n_out, activation=None ):
# 	# Bound ?
# 	W_bound = np.sqrt(6. / (n_in + n_out))
# 	# Create W_init from a Uniform distribution.
# 	W_init = np.asarray( rng.uniform( low=-W_bound, high=W_bound, size=(n_in, n_out) ), dtype=theano.config.floatX )
# 	# ??
# 	if activation is not None and activation.name == 'Sigmoid':
# 		W_init *= 4.0
# 	# Return the weights.
# 	return W_init
# 	##################


# -------------------- #

class Init(object):
	
	def __call__(self, shape):
		return self.sample(shape)

	def sample(self, shape):
		raise NotImplementedError()

	# -------------------- #



# -------------------- #
class Uniform(Init):
	
	def __init__(self, range=None):
		self.range = range

	def sample(self, shape):
		if self.range is None:
			# no range given, use the Glorot et al. approach.
			# This code makes some assumptions about the meanings of
			# the different dimensions, which hold for
			# layers.DenseLayer and layers.Conv*DLayer, but not
			# necessarily for other layer types.
			n1, n2 = shape[:2]
			receptive_field_size = np.prod(shape[2:])
			m = np.sqrt(6.0 / ((n1 + n2) * receptive_field_size))
			range = (-m, m)

		elif isinstance(self.range, Number):
			range = (-self.range, self.range)

		else:
			range = self.range

		return np.asarray( np.random.uniform( low=range[0], high=range[1], size=shape), dtype=theano.config.floatX )
	# -------------------- #


# -------------------- #
class Normal(Init):
    """
    Sample initial weights from the Gaussian distribution.
    Initial weight parameters are sampled from N(mean, std).
    Parameters
    ----------
    std : float
        Std of initial parameters.
    mean : float
        Mean of initial parameters.
    """
    def __init__(self, std=0.01, mean=0.0):
        self.std = std
        self.mean = mean

    def sample(self, shape):
        return np.asarray( np.random.normal(self.mean, self.std, size=shape), dtype=theano.config.floatX )
    # -------------------- #


# -------------------- #
class Constant(Init):
    """
    Initialize weights with constant value.
    Parameters
    ----------
     val : float
        Constant value for weights.
    """
    def __init__(self, val=0.0):
        self.val = val

    def sample(self, shape):
        return np.asarray( np.ones(shape) * self.val, dtype=theano.config.floatX )
    # -------------------- #






