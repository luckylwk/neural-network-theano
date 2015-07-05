
import numpy as np

import theano




# Classes for each activation function
class CrossEntropy():
	"""
	Return the mean of the negative log-likelihood of the prediction
    of this model under a given target distribution. Mathematics::
        
        \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
        \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})
    
    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the correct label
    Note: we use the mean instead of the sum so that the learning rate is less dependent on the batch size
    """
	#
	name = 'Cross-Entropy'
	#
	# y.shape[0] is (symbolically) the number of rows in y, i.e.,
    # number of examples (call it n) in the minibatch
    # T.arange(y.shape[0]) is a symbolic vector which will contain
    # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
    # Log-Probabilities (call it LP) with one row per example and
    # one column per class LP[T.arange(y.shape[0]),y] is a vector
    # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
    # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
    # the mean (across minibatch examples) of the elements in v,
    # i.e., the mean log-likelihood across the minibatch.
	@staticmethod
	def fn( y_pred, y ):
		return -theano.tensor.mean( theano.tensor.log(y_pred)[ theano.tensor.arange(y.shape.eval()[0]), y ] )
		# -------------------- #
	# -------------------- #
