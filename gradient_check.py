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

    # func2, _ = f(x+delta)
    # func1, _ = f(x-delta)
    # _, df_analyt = f(x)

    # df_num = (func2 - func1)/(2*delta)
    # checking = abs(df_analyt - df_num) < tol

    # print("df_analyt is ", df_analyt)
    # print("df_num is ", df_num)
    # print("gradient checking ", checking)
    # return checking

    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float

    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)
                  ), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()
    print("analytic_grad ", analytic_grad)

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    delta_array = np.ones(x.shape)*delta
    numeric_grad = np.zeros(analytic_grad.shape)
    while not it.finished:
        ix = it.multi_index
        # print("ix is ", ix)
        # print("it is ", it)

        analytic_grad_at_ix = analytic_grad[ix]
        print("analytic_grad_at_ix ", analytic_grad_at_ix)
        print("it value is ", it.value)
        print("ix ", ix)
        val = it.value
        zero_arr = np.zeros(x.shape)
        zero_arr[ix] = delta
        print("zero arr is ", zero_arr)
        func2, _ = f(orig_x + zero_arr)
        func1, _ = f(orig_x - zero_arr)
        print("func1 is ", func1)
        print("func2 is ", func2)
        numeric_grad[ix] = (func2 - func1) / (2*delta)
        numeric_grad_at_ix = numeric_grad[ix]
        print("numeric_grad_at_ix ", numeric_grad_at_ix)

        # TODO compute value of numeric gradient of f to idx
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (
                ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True
