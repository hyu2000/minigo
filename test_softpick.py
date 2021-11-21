import numpy as np


def test_searchsorted():
    """ softpick gives me lots of edge moves, which is suspicious

    Looks like policy is very weak at this point.
    """
    N = 104
    child_n2 = np.array([8, 7, 7, 7, 7, 7, 6, 2, 2, 2, 2])
    # 47 ~ 45%
    ones = N - child_n2.sum()
    child_N = np.concatenate([child_n2, np.ones(ones)])

    # prune
    child_N[child_N == 1] = 0

    # children_as_pi
    squash = True
    if squash:
        child_N = child_N ** 0.98
    child_N /= child_N.sum()

    cdf = child_N.cumsum()
    if cdf[-2] > 1e-6:
        cdf /= cdf[-2]  # Prevents passing via softpick.
        selection = np.random.random()  # probably better than random.random()?
        fcoord = cdf.searchsorted(selection)
        print(fcoord)
