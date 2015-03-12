
import activation


def test_sigmoid():
	assert activation.Sigmoid.fn(0.0).eval() == 0.5