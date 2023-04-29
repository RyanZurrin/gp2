import scipy.stats
import numpy as np

def test_T_215():
    """ Test whether scipy.stats.mode works.

    Tests:
        1. test whether computed and expected results are the same when axis=0 (by default) configured
        2. test whether computed and expected results are the same when axis=None (mode of whole array) configured
    """

    example_arr = np.array([
        [6, 8, 3, 0],
        [3, 2, 1, 7],
        [8, 1, 8, 4],
        [5, 3, 0, 5],
        [4, 7, 5, 9]
    ])

    # axis=0 by default
    result = scipy.stats.mode(example_arr)
    assert(np.array_equal(result[0], np.array([[3, 1, 0, 0]])))
    assert(np.array_equal(result[1], np.array([[1, 1, 1, 1]])))

    # asix=None to find mode of whole array
    result = scipy.stats.mode(example_arr, axis=None)
    assert(np.array_equal(result[0], np.array([3])))
    assert(np.array_equal(result[1], np.array([3])))

