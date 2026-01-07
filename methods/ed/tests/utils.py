import numpy as np

# since atol and rtol is in most cases the same, we define them here
def assert_allclose(actual, expected, atol: float = 1e-14, rtol: float = 0.0):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)