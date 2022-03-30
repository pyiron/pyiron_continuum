import fenics as FEN
from fenics import near
from pyiron_continuum.fenics.factory import StringInputParser

T = 2.0            # final time
num_steps = 10     # number of time steps
dt = T / num_steps # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

# Create mesh and define function space
nx = ny = 8
mesh = FEN.UnitSquareMesh(nx, ny)
V = FEN.FunctionSpace(mesh, 'P', 1)
u_L = FEN.Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                     degree=2, alpha=alpha, beta=beta, t=0)


s = StringInputParser('x[2]*19 + x[1]**2 + 1.2 * beta', beta=2)
print(s._split_input())
#assert s._split_input() == ['s', '19', 'hy', '2']

def wrap_dirichlet(function_space, condition, expr, **kwargs):
    # it is needed definitly: exp_kwargs=None, condition_kwargs=None
    # parsing conditions should be also done and tested!
    if isinstance(expr, str):
        print(expr)
        expr = FEN.Expression(expr, **kwargs)

    def boundary_func(x, on_boundary):
        try:
            return on_boundary and eval(condition)
        except Exception as err_msg:
            print(err_msg)

    return FEN.DirichletBC(function_space, expr, boundary_func)

condition_string = StringInputParser("near(x[0], 0, 1e-14)")
expr_string = StringInputParser(
    '1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
    alpha=alpha,
    beta=beta,
    t=0
)
bc = wrap_dirichlet(function_space=V, condition=condition_string.input_string,
                    expr=expr_string.input_string,
                    degree=2, alpha=alpha, beta=beta, t=0)

from fenics import *
# Define initial value
u_n = FEN.interpolate(u_L, V)
#u_n = project(u_D, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    u_L.t = t

    # Compute solution
    solve(a == L, u, bc)


    # Update previous solution
    u_n.assign(u)

u.compute_vertex_values()
