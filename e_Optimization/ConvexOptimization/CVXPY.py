import cvxpy as cp

def f_cvxpy(Q, b, x):
    return 0.5 * cp.quad_form(x, Q) - b.T @ x

def OptimalSolution(Q, b, n):
    x = cp.Variable((n, 1))
    
    obj = cp.Minimize(f_cvxpy(cp.psd_wrap(Q), b, x))
    c = [x >= 0]

    prob = cp.Problem(obj, c)
    prob.solve()

    return x.value, prob.value