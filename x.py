
import numpy as np

from core.layers import HiddenLayer

random_seed = 1234



if __name__ == '__main__':


	rng = np.random.RandomState(random_seed)
	
	l0 = InputLayer()
	l1 = HiddenLayer( layerInput=(1,1), verbose=True )

