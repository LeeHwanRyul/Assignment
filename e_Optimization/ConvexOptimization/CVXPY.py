import cvxpy as cp

def f_cvxpy(Q, b, x):
    return 0.5 * cp.quad_form(x, Q) - b.T @ x

def OptimalSolution(Q, b, n):
    x = cp.Variable((n, 1))
    
    obj = cp.Minimize(f_cvxpy(Q, b, x))

    prob = cp.Problem(obj)
    prob.solve()

    return x.value, prob.value