#!/home/dilara/acados/.venv/bin/python
from acados_template import AcadosOcp, AcadosOcpSolver
from quadcopter_model import export_quadcopter_ode_model
import numpy as np
from utils import plot_drone

def main():
    # create ocp object to formulate the OCP
    ocp=AcadosOcp()

    # set model

    modelquad=export_quadcopter_ode_model()
    ocp.model=modelquad

    Tf=1.0
    nx = modelquad.x.size()[0]
    nu = modelquad.u.size()[0]
    N =35

    print(modelquad.x)

    # set dimensions
    ocp.dims.N = N

    ocp.solver_options.tf = Tf  # Total time horizon

    # set cost
    #To do: (fidn) Change it according to needs and also check how to tune
    Q_mat = 2*np.diag([10000, 1, 100, 1, 1, 1])
    R_mat = 2*np.diag([0.001,1,1])

    #set cost type and other
    ocp.cost.cost_type='LINEAR_LS'
    ocp.cost.cost_type_e='LINEAR_LS'
    ocp.model.cost_expr_ext_cost= modelquad.x.T @ Q_mat @ modelquad.x + modelquad.u.T @ R_mat @ modelquad.u
    ocp.model.cost_expr_ext_cost_e = modelquad.x.T @ Q_mat @ modelquad.x


    #set constrains
    taux_max=19.35
    tauy_max=19.35
    tauz_max=5


    ocp.constraints.idxbu = np.array([0, 1, 2])
    ocp.constraints.lbu = np.array([-taux_max,-tauy_max,-tauz_max])
    ocp.constraints.ubu = np.array([+taux_max,+tauy_max,+tauz_max])
   

    ocp.constraints.x0=np.array([0.1,-0.1,0.2,0,0,0])

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' 
    # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    # ocp.solver_options.print_level = 1
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP


    # set prediction horizon
    ocp.solver_options.tf = Tf

    ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')

    simX = np.ndarray((N+1, nx))
    simU = np.ndarray((N, nu))

    status = ocp_solver.solve()
    ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

    if status != 0:
        raise Exception(f'acados returned status {status}.')

    # get solution
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
        simX[N,:] = ocp_solver.get(N, "x")

    plot_drone(np.linspace(0, Tf, N+1), [taux_max,tauy_max,tauz_max], simU, simX, latexify=False)


if __name__ == '__main__':
    main()









