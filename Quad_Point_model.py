# developer: DC "the next champnion" C

import numpy as np
from numpy import tan, sin, cos
import casadi as ca
from acados_template import AcadosModel



class Q_P_model():
    def __init__(self):
        model = AcadosModel()
        model.name = 'Quadrotor_Pointmass_Model'
        constraint = ca.types.SimpleNamespace()

        # 无人机、绳、球参数设置
        self.I_xx = 2.32e-3
        self.I_yy = 2.32e-3
        self.I_zz = 4.00e-3
        self.m = 0.5
        self.g = 9.8
        self.I = np.array([[self.I_xx, .0, .0], [.0, self.I_yy, .0], [.0, .0, self.I_zz]])
        self.l = 0.5 # 可设置为0.5
        self.ml = 0.1 # 可设置为0.1
        self.z_axis = np.array([0, 0, 1])


        # 控制输入
        F = ca.MX.sym('thrust', 1)
        M = ca.MX.sym('moment', 3)
        u = ca.vertcat(F, M)


        # 状态量 q:quadrotor c:cable l:load
        pos_q = ca.MX.sym('pos_q', 3)
        vel_q = ca.MX.sym('vel_q', 3)
        att_q = ca.MX.sym('att_q', 4) # quaternion form
        rate_q = ca.MX.sym('rate_q', 3)

        p_c = ca.MX.sym('p_c', 3) # 绳方向单位向量
        d_p_c = ca.MX.sym('d_p_c', 3) # 绳方向单位向量的导数

        # pos_l = ca.MX.sym('pos_l', 3)
        # vel_l = ca.MX.sym('vel_l', 3)
        # x = ca.vertcat(pos_q, vel_q, att_q, rate_q, p_c, d_p_c, pos_l, vel_l)
        x = ca.vertcat(pos_q, vel_q, att_q, rate_q, p_c, d_p_c)
        # x = ca.vertcat(pos_q, vel_q, att_q, rate_q) ################ for debugging

        d_pos_q = ca.MX.sym('d_pos_q', 3)
        d_vel_q = ca.MX.sym('d_vel_q', 3)
        d_att_q = ca.MX.sym('d_att_q', 4)
        d_rate_q = ca.MX.sym('d_rate_q', 3)

        dot_p_c = ca.MX.sym('d_p_c', 3)
        dotd_p_c = ca.MX.sym('dd_p_c', 3)

        # d_pos_l = ca.MX.sym('d_pos_l', 3)
        # d_vel_l = ca.MX.sym('d_vel_l', 3)
        # d_x = ca.vertcat(d_pos_q, d_vel_q, d_att_q, d_rate_q, d_p_c, dd_p_c, d_pos_l, d_vel_l)
        d_x = ca.vertcat(d_pos_q, d_vel_q, d_att_q, d_rate_q, dot_p_c, dotd_p_c)
        # d_x = ca.vertcat(d_pos_q, d_vel_q, d_att_q, d_rate_q) ########## for debugging


        # 动态函数
        F_world = F*self.q_rot(att_q, self.z_axis)
        # f_expl = ca.vertcat(vel_q,
        #                     np.array([.0,.0,self.g]) + ((np.dot(p_c, F_world)-self.m*self.l*np.dot(d_p_c, d_p_c))*p_c)/(self.m + self.ml) - self.l*(-np.dot(d_p_c, d_p_c)*p_c + np.cross(p_c, np.cross(p_c, F_world))/self.m),
        #                     1/2*self.q_muilty(att_q, ca.vertcat(0, att_q)),
        #                     np.linalg.inv(self.I)@(M - ca.cross(att_q, self.I@att_q)),
        #                     d_p_c,
        #                     -np.dot(d_p_c, d_p_c)*p_c + np.cross(p_c, np.cross(p_c, F_world))/self.m,
        #                     vel_l,
        #                     np.array([.0,.0,self.g]) + ((np.dot(p_c, F_world)-self.m*self.l*np.dot(d_p_c, d_p_c))*p_c)/(self.m + self.ml))
        f_expl = ca.vertcat(vel_q,
                            np.array([.0,.0,-self.g]) + ((ca.dot(p_c, F_world)-self.m*self.l*ca.dot(d_p_c, d_p_c))*p_c)/(self.m + self.ml) - self.l*(-ca.dot(d_p_c, d_p_c)*p_c + ca.cross(p_c, ca.cross(p_c, F_world))/self.m),
                            1/2*self.q_muilty(att_q, ca.vertcat(0, rate_q)),
                            np.linalg.inv(self.I)@(M - ca.cross(rate_q, self.I@rate_q)),
                            d_p_c,
                            -ca.dot(d_p_c, d_p_c)*p_c + ca.cross(p_c, ca.cross(p_c, F_world))/self.m
                            )
        
                            # np.array([.0,.0,self.g]) + ((np.dot(P, f)-self.m*self.l*np.dot(d_p, d_p))*P)/(self.m + self.ml) - self.l*(-np.dot(d_p, d_p)*P + np.cross(P, np.cross(P, f))/self.m) # numerical simulation
                            # ddp = -np.dot(d_p, d_p)*P + np.cross(P, np.cross(P, f))/self.m

        
        ########### for debugging ##########
        # f_expl = ca.vertcat(vel_q,
        #                     F*self.q_rot(att_q, self.z_axis)/self.m + np.array([.0, .0, self.g]),
        #                     1/2*self.q_muilty(att_q, ca.vertcat(0,rate_q)),
        #                     np.linalg.inv(self.I)@(M - ca.cross(rate_q, self.I@rate_q)))
        
        
        f_impl = f_expl - d_x


        # 设置模型属性
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = d_x
        model.u = u
        model.p = []


        # 约束
        constraint.rate_max = np.array([np.pi, np.pi, np.pi])
        constraint.rate_min = np.array([-np.pi, -np.pi, -np.pi])
        
        constraint.F_max = 10
        constraint.F_min = -10
        constraint.M_max = np.array([1, 1, 1])
        constraint.M_min = np.array([-1, -1, -1])


        # 元数据补充


        self.model = model
        self.constraint = constraint


    def q_muilty(self, q1, q2):
        result = ca.vertcat(q1[0]*q2[0] - ca.dot(q1[1:4],q2[1:4]), q1[0]*q2[1:4] + q2[0]*q1[1:4] + ca.cross(q1[1:4],q2[1:4]))
        return result
    
    def q_rot(self, q, axis):
        q_inv = ca.vertcat(q[0],-q[1:4])/ca.norm_2(q)**2
        cal_axis = ca.vertcat(0,axis)
        new_q = self.q_muilty(self.q_muilty(q,cal_axis), q_inv)
        new_axis = new_q[1:4]
        return new_axis
