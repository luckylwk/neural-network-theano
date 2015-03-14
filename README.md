# Neural Network using Theano.

Notes.

### Data

~~~python
# Load in the DATA
>>> from core.datasets import mnist
>>> datasets = mnist.fn_T_load_data_MNIST( path_to_file='../../_DATA/mnist.pkl.gz' )
~~~

The `datasets` variable is a list of tuples. It holds the training data, cross-validation data, testing data and the image-dimensions.

### Layers

~~~python
# Setup the LAYERS
# Start with setting up an input layer.
>>> l0 = InputLayer(  )
# Define the next layer.
>>> l1 = HiddenLayer( layerInput=(1,1), verbose=True )
~~~

~~~python
# Setup the MODEL
# Pass the model all the layers.
~~~

~~~python
# Setup the TRAINER
~~~

Go!