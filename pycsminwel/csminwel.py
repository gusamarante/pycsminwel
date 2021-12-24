import numpy as np


def csminwel(fcn, x0, h0=None, grad=None, crit=None, nit=None, verbose=True):
    # TODO verbose False

    # TODO possible renameming of variables

    # TODO Documentation (minimization. Uses a quasi-Newton method with BFGS
    #  update of the estimated inverse hessian. It is robust against certain pathologies
    #  common on likelihood functions. It attempts to be robust against "cliffs", i.e.
    #  hyperplane discontinuities, though it is not really clear whether what it does in
    #  such cases succeeds reliably.)

    # TODO assert types. x0 has ndim=1. fcn and grad ar functions

    nx = x0.shape[0]
    iter_count = 0
    fcount = 0
    numGrad = True if grad is None else False

    f0 = fcn(x0)

    if numGrad:
        g, badg = numgrad(fcn, x0)
    else:
        g = grad(x0)
        badg = False

    x = x0
    f = f0
    h = h0
    done = False

    while not done:
        print('f at the beginning of new iteration is', f)  # TODO if verbose
        iter_count += 1
        f1, x1, fc, retcode1 = csminit(fcn, x, f, g, badg, h)

        # TODO parei aqui - linha 73 do MATLAB csminwel

    return g


def csminit(fcn, x0, f0, g0, badg, h0):

    # Do not ask where these come from
    angle = 0.005
    theta = 0.3
    fchange = 1000
    minlamb = 1e-9
    mindfac = 0.01

    fcount = 0
    lambda_ = 1
    xhat = x0
    f = f0
    fhat = f0
    g = g0
    gnorm = np.linalg.norm(g)

    if gnorm < 1e-12 and not badg:
        # gradient convergence
        retcode = 1
        dxnorm = 0
    else:
        # with badg true, we don't try to match rate of improvement to
        # directional derivative.  We're satisfied just to get *some*
        # improvement in f.
        dx = - h0 @ g
        dxnorm = np.linalg.norm(dx)

        if dxnorm > 1e12:  # near singular H problem
            dx = dx @ fchange / dxnorm

        dfhat = dx.T @ g0

        if not badg:
            # test for alignment of dx with gradient and fix if necessary
            a = - dfhat / (gnorm * dxnorm)
            if a < angle:
                dx = dx - (angle * dxnorm / gnorm + dfhat/(gnorm ** 2)) * g
                dx = dx * dxnorm / np.linalg.norm(dx)
                dfhat = dx.T @ g

        done = False
        factor = 3
        shrink = 1
        lambdaMax = np.inf
        lambdaPeak = 0
        fPeak = f0
        lambdaHat = 0

        while not done:
            if x0.shape[0] > 1:
                dxtest = x0 + dx.T * lambda_
            else:
                dxtest = x0 + dx * lambda_

            f = fcn(dxtest)

            if f < fhat:
                fhat = f
                xhat = dxtest
                lambdaHat = lambda_

            fcount += 1

            shrinkSignal = ((not badg) and (f0 - f < max(- theta * dfhat * lambda_, 0))) or (badg and (f0 - f < 0))
            growSignal = (not badg) and ((lambda_ > 0) and (f0 - f > - (1 - theta) * dfhat * lambda_))

            # TODO parei aqui - linha 123 do MATLAB csminit

    return fhat, xhat, fcount, retcode


def numgrad(fcn, x):
    """
    Computes the numerical gradient of a function at point x.
    :param fcn: python function that must depend on a single argument
    :param x: 1-D numpy.array
    :return: 1-D numpy.array
    """

    delta = 1e-6
    bad_gradient = False
    n = x.shape[0]
    tvec = delta * np.eye(n)
    g = np.zeros(n)

    f0 = fcn(x)

    for i in range(n):
        tvecv = tvec[i, :]

        g0 = (fcn(x + tvecv) - f0) / delta

        if np.abs(g0) < 1e15:  # good gradient
            g[i] = g0
        else:  # bad gradient
            g[i] = 0
            bad_gradient = True

    return g, bad_gradient
