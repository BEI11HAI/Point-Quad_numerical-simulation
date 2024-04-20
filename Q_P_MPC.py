# developer: DC "the next champnion" C

import os
import sys
import shutil
import errno
import timeit

from Quad_Point_model import Q_P_model
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

import numpy as np
import scipy.linalg

import pandas as pd
import matplotlib.pyplot as plt



def safe_mkdir_recursive(directory, overwrite=False):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory):
                pass
            else:
                raise
    else:
        if overwrite:
            try:
                shutil.rmtree(directory)
            except:
                print('Error while removing directory {}'.format(directory))


def draw_state_result(csv_file_name ):
    # 读取csv文件
    data = pd.read_csv(csv_file_name)

    # 获取列名
    columns = data.columns.tolist()

    # 创建子图网格
    fig, axs = plt.subplots(4, 4, figsize=(12, 8))  

    # 绘制
    axs[0, 0].plot(data[columns[0]])
    axs[0, 0].set_ylabel('x')
    axs[0, 1].plot(data[columns[1]])
    axs[0, 1].set_ylabel('y')
    axs[0, 2].plot(data[columns[2]])
    axs[0, 2].set_ylabel('z')
    axs[1, 0].plot(data[columns[3]])
    axs[1, 0].set_ylabel('vx')
    axs[1, 1].plot(data[columns[4]])
    axs[1, 1].set_ylabel('vy')
    axs[1, 2].plot(data[columns[5]])
    axs[1, 2].set_ylabel('vz')
    axs[2, 0].plot(data[columns[6]])
    axs[2, 0].set_ylabel('w')
    axs[2, 1].plot(data[columns[7]])
    axs[2, 1].set_ylabel('i')
    axs[2, 2].plot(data[columns[8]])
    axs[2, 2].set_ylabel('j')
    axs[2, 3].plot(data[columns[9]])
    axs[2, 3].set_ylabel('k')
    axs[3, 0].plot(data[columns[10]])
    axs[3, 0].set_ylabel('p')
    axs[3, 1].plot(data[columns[11]])
    axs[3, 1].set_ylabel('q')
    axs[3, 2].plot(data[columns[12]])
    axs[3, 2].set_ylabel('r')

    # 删除多余
    fig.delaxes(axs[0, 3])
    fig.delaxes(axs[1, 3])
    fig.delaxes(axs[3, 3])

    fig2, axs2 = plt.subplots(2, 3, figsize=(12, 8))
    axs2[0, 0].plot(data[columns[13]])
    axs2[0, 0].set_ylabel('cable_x')
    axs2[0, 1].plot(data[columns[14]])
    axs2[0, 1].set_ylabel('cable_y')
    axs2[0, 2].plot(data[columns[15]])
    axs2[0, 2].set_ylabel('cable_z')
    axs2[1, 0].plot(data[columns[16]])
    axs2[1, 0].set_ylabel('cable_vx')
    axs2[1, 1].plot(data[columns[17]])
    axs2[1, 1].set_ylabel('cable_vy')
    axs2[1, 2].plot(data[columns[18]])
    axs2[1, 2].set_ylabel('cable_vz')

    # 显示图形
    plt.tight_layout()  # 自动调整子图参数以给定指定的填充
    plt.show()


def draw_payload_result(csv_file_name):
    # 读取csv文件
    data = pd.read_csv(csv_file_name)

    # 获取列名
    columns = data.columns.tolist()

    # 创建子图网格
    fig, axs = plt.subplots(4, 4, figsize=(12, 8))  

    # 绘制
    axs[0, 0].plot(data[columns[0]])
    axs[0, 0].set_ylabel('x')
    axs[0, 1].plot(data[columns[1]])
    axs[0, 1].set_ylabel('y')
    axs[0, 2].plot(data[columns[2]])
    axs[0, 2].set_ylabel('z')
    axs[1, 0].plot(data[columns[3]])
    axs[1, 0].set_ylabel('vx')
    axs[1, 1].plot(data[columns[4]])
    axs[1, 1].set_ylabel('vy')
    axs[1, 2].plot(data[columns[5]])
    axs[1, 2].set_ylabel('vz')
    axs[2, 0].plot(data[columns[6]])
    axs[2, 0].set_ylabel('w')
    axs[2, 1].plot(data[columns[7]])
    axs[2, 1].set_ylabel('i')
    axs[2, 2].plot(data[columns[8]])
    axs[2, 2].set_ylabel('j')
    axs[2, 3].plot(data[columns[9]])
    axs[2, 3].set_ylabel('k')
    axs[3, 0].plot(data[columns[10]])
    axs[3, 0].set_ylabel('p')
    axs[3, 1].plot(data[columns[11]])
    axs[3, 1].set_ylabel('q')
    axs[3, 2].plot(data[columns[12]])
    axs[3, 2].set_ylabel('r')

    # 删除多余
    fig.delaxes(axs[0, 3])
    fig.delaxes(axs[1, 3])
    fig.delaxes(axs[3, 3])

    # 显示图形
    plt.tight_layout()  # 自动调整子图参数以给定指定的填充
    plt.show()

class Q_P_Opt():
    def __init__(self, d_model, d_constraint, t_horizon, n_nodes):
        model = d_model
        self.T = t_horizon
        self.N = n_nodes
        self.Nsim = 2*self.N

        # 保证当前工作目录为当前文件夹
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self.acados_models_dir = './acados_models'
        safe_mkdir_recursive(os.path.join(os.getcwd(), self.acados_models_dir))
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)

        nx = model.x.size()[0]
        self.nx = nx
        nu = model.u.size()[0]
        self.nu = nu
        ny = nx + nu
        n_params = len(model.p) # 可能有误

        # 建立OCP
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = model
        ocp.dims.N = self.N # OCP预测步数
        ocp.solver_options.tf = self.T

        # 初始化参数
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros(n_params)

        # 目标函数
        # Q = np.diag([200, 200, 500,  # pos_q
        #              0.1, 0.1, 0.1,  # vel_q
        #              0.1, 0.1, 0.1, 0.1, # att_q 
        #              0.1, 0.1, 0.1,  # rate_q
        #              0.1, 0.1, 0.1, # p_c
        #              0.1, 0.1, 0.1, # d_p_c
        #              0.1, 0.1, 0.1, # pos_l
        #              0.1, 0.1, 0.1]) # x的权重矩阵
        Q = np.diag([200, 200, 500,  # pos_q
                     0.1, 0.1, 0.1,  # vel_q
                     0.1, 0.1, 0.1, 0.1,  # att_q 
                     0.1, 0.1, 0.1,  # rate_q
                     0.1, 0.1, 0.1, # p_c
                     0.1, 0.1, 0.1 # d_p_c
                     ]) # x的权重矩阵
        ############## for debugging ###############
        # Q = np.diag([200, 200, 500,  # pos_q
        #              0.1, 0.1, 0.1,  # vel_q
        #              0.1, 0.1, 0.1, 0.1,  # att_q 
        #              0.1, 0.1, 0.1   # rate_q
        #              ]) # x的权重矩阵
        R = np.diag([0.1, 0.1, 0.1, 0.1]) # u的权重矩阵
        ocp.cost.cost_type = 'LINEAR_LS' # 阶段、终点代价均为线性最小二乘
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q

        # 
        # ocp.cost.Vx = np.zeros((ny-1, nx))
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:6, :6] = np.eye(6)
        ocp.cost.Vx[6:10, 6:10] = np.eye(4)
        ocp.cost.Vx[10:13, 10:13] = np.eye(3)
        ocp.cost.Vx[13:16, 13:16] = np.eye(3)
        ocp.cost.Vx[16:19, 16:19] = np.eye(3)
        # ocp.cost.Vx_e = ocp.cost.Vx[:(nx-1), :nx]
        ocp.cost.Vx_e = ocp.cost.Vx[:nx, :nx]

        # ocp.cost.Vu = np.zeros((ny-1, nu))
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)


        x_init = np.zeros(nx)
        x_init[6] = 1
        x_init[15] = -1 ######### for debugging
        u_ref = np.zeros(nu)
        # initial state
        ocp.constraints.x0 = x_init
        # x_ref = np.zeros(nx-1)
        x_ref = np.zeros(nx)
        ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref_e = x_ref

        # solver options
        # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        # ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # work
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM' # work when t_horizon = 2 & xs = (1,1,5)
        # all qp_solver type: 'PARTIAL_CONDENSING_HPIPM', 'FULL_CONDENSING_QPOASES', 
        # 'FULL_CONDENSING_HPIPM', 'PARTIAL_CONDENSING_QPDUNES', 'PARTIAL_CONDENSING_OSQP', 'FULL_CONDENSING_DAQP'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # explicit Runge-Kutta integrator
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        # compile acados ocp
        json_file = os.path.join('./'+model.name+'_acados_ocp.json')
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)

    def simulation(self, x0, xs):
        simX = np.zeros((self.N+1, self.nx))
        simU = np.zeros((self.N, self.nu))
        x_current = x0
        simX[0, :] = x0.reshape(1, -1) # -1表示自动计算列数量
        xs_between = np.concatenate((xs, np.zeros(self.nu)))
        time_record = np.zeros(self.N)

        # closed loop
        self.solver.set(self.N, 'yref', xs)
        for i in range(self.N):
            self.solver.set(i, 'yref', xs_between)

        for i in range(self.N):
            # solve ocp
            start = timeit.default_timer()
            ##  set inertial (stage 0)
            self.solver.set(0, 'lbx', x_current)
            self.solver.set(0, 'ubx', x_current)
            status = self.solver.solve()

            if status != 0 :
                raise Exception('acados acados_ocp_solver returned status {}. in closed loop iteration {}.'.format(status, i))

            simU[i, :] = self.solver.get(0, 'u')
            time_record[i] =  timeit.default_timer() - start
            # simulate system
            self.integrator.set('x', x_current)
            self.integrator.set('u', simU[i, :])

            status_s = self.integrator.solve()
            if status_s != 0:
                raise Exception('acados integrator returned status {}. in closed loop iteration {}.'.format(status, i))

            # update
            x_current = self.integrator.get('x')
            simX[i+1, :] = x_current


        print("average estimation time is {}".format(time_record.mean()))
        print("max estimation time is {}".format(time_record.max()))
        print("min estimation time is {}".format(time_record.min()))
        np.savetxt(fname="drone_state.csv", X=simX, fmt="%lf",delimiter=",")
        np.savetxt(fname="drone_control.csv", X=simU, fmt="%lf",delimiter=",")

        draw_state_result("drone_state.csv")
        # draw_payload_result("drone_control.csv")
        

if __name__ == '__main__':
    drone_model = Q_P_model()
    opt = Q_P_Opt(d_model=drone_model.model, d_constraint=drone_model.constraint, t_horizon=2, n_nodes=100)
    # opt.simulation(x0=np.array([0, 0, -5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), xs=np.array([1, 1, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
    opt.simulation(x0=np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0]), xs=np.array([1, 1, 5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0]))
    