import numpy as np


def check_gradient(f, x, delta=1e-5, tol=1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''

    print("CHECK GRADIENT")
    print("predictions is \n", x)

    assert isinstance(x, np.ndarray)
    assert x.dtype == float

    orig_x = x.copy()
    fx, analytic_grad = f(x)
    print("analytic grad is \n", analytic_grad)
    assert np.all(np.isclose(orig_x, x, tol)
                  ), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()
    numeric_grad_arr = numericDerivative(f, x, delta)

    it = np.nditer(numeric_grad_arr, flags=[
                   'multi_index'], op_flags=['readwrite'])
    print("==========================================")
    while not it.finished:
        ix = it.multi_index
        # print("ix is ", ix)
        analytic_grad_at_ix = analytic_grad[ix]
        numeric_grad_at_ix = numeric_grad_arr[ix]

        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (
                ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True


def numericDerivative(f, x, delta=1e-5):
    # We will go through every dimension of x and compute numeric
    # derivative for it
    numeric_grad_arr = np.zeros(x.shape)
    it = np.nditer(numeric_grad_arr, flags=[
                   'multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        # print("ix is ", ix)
        deviation = np.zeros(x.shape)
        deviation[ix] = delta
        y2, _ = f(x + deviation)
        y1, _ = f(x - deviation)
        numeric_grad_arr[ix] = (y2 - y1) / (2*delta)
        it.iternext()

    print("numeric grad array is \n", numeric_grad_arr)
    return numeric_grad_arr
