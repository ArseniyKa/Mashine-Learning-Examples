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
    print("prediction is \n", x)

    assert isinstance(x, np.ndarray)
    assert x.dtype == float

    orig_x = x.copy()
    fx, analytic_grad = f(x)
    print("analytic grad is \n", analytic_grad)
    # analytic_grad_average_arr = np.average(analytic_grad, axis=0)
    # print("analytic average grad is \n", analytic_grad_average_arr)
    assert np.all(np.isclose(orig_x, x, tol)
                  ), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    numeric_grad_arr = np.zeros(x.shape)
    print("x shape[1] ", x.shape[1])
    print("==========================================")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            deviation = np.zeros(x.shape)
            deviation[i][j] = delta

        # deviation_column = np.zeros(x.shape)
        # # numeric_grad = np.zeros(analytic_grad.shape)
        # deviation_column[:, i] = delta
        # print("deviation array is \n", deviation_column)
            func2, _ = f(x + deviation)
            func1, _ = f(x - deviation)
        # print("func1 is ", func1)
        # print("func2 is ", func2)
            numeric_grad = (func2 - func1) / (2*delta)
            numeric_grad_arr[i][j] = numeric_grad
        # print("func2 is ", func2)
        # print("numeric grad is \n", numeric_grad)

    print("numeric grad array is \n", numeric_grad_arr)
    it = np.nditer(numeric_grad_arr, flags=[
                   'multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        # print("ix is ", ix)
        # print("it is ", it.value)
        analytic_grad_at_ix = analytic_grad[ix]
        numeric_grad_at_ix = numeric_grad_arr[ix]
        # print("analytic_grad_at_ix ", analytic_grad_at_ix)
        # print("numeric_grad_at_ix ", numeric_grad_at_ix)

        # TODO compute value of numeric gradient of f to idx
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (
                ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True
