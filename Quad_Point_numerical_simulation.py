import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan


class DroneControlSim:
    def __init__(self):
        self.sim_time = 10
        self.sim_step = 0.002
        self.drone_states = np.zeros((int(self.sim_time/self.sim_step), 24))    # 以p及dp为状态变量
        self.time= np.zeros((int(self.sim_time/self.sim_step),))
        self.rate_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.attitude_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.velocity_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.position_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.pointer = 0 

        self.I_xx = 2.32e-3
        self.I_yy = 2.32e-3
        self.I_zz = 4.00e-3
        self.m = 0.5
        self.g = 9.8
        self.I = np.array([[self.I_xx, .0,.0],[.0,self.I_yy,.0],[.0,.0,self.I_zz]])
        self.l = 1.0    # 增加绳长变量
        self.ml = 0.1    # 增加负载质量变量

        self.drone_states[0, 14] = 1    # 设置方向向量初值
        self.drone_states[0, 18:21] = self.drone_states[0, 0:3] + self.drone_states[0, 12:15] * self.l  # 设置负载位置初值

        self.vel_err = np.zeros((int(self.sim_time/self.sim_step), 3))
        self.pos_err = np.zeros((int(self.sim_time/self.sim_step), 3))


    def run(self):
        position_cmd = np.array([5.0, 5.0, 5.0])
        for self.pointer in range(self.drone_states.shape[0]-1):
            self.time[self.pointer] = self.pointer * self.sim_step
            thrust_cmd = 0
            M = np.zeros((3,))

            self.position_cmd[self.pointer, :] = position_cmd
            self.position_controller(self.position_cmd[self.pointer, :])

            # self.velocity_cmd[self.pointer, :] = np.array([1, 1, 1])    # 测试速度环
            thrust_cmd = self.velocity_controller(self.velocity_cmd[self.pointer, :])

            # thrust_cmd = -1*self.g*self.m   # 测试姿态环
            # self.attitude_cmd[self.pointer, :] = np.array([2, 2, 1])    # 测试姿态环
            self.attitude_controller(self.attitude_cmd[self.pointer, :])

            # self.rate_cmd[self.pointer, :] = np.array([1.0, 1.0, 1.0])  # 测试角速率环
            M = self.rate_controller(self.rate_cmd[self.pointer, :])

            dx = self.drone_dynamics(thrust_cmd, M)

            self.drone_states[self.pointer+1, ] = self.drone_states[self.pointer, ] + self.sim_step*dx
            
        self.time[-1] = self.sim_time
        

    def drone_dynamics(self,T,M):
        # Input:
        # T: float Thrust
        # M: np.array (3,)  Moments in three axes
        # Output: np.array (12,) the derivative (dx) of the drone 
        
        # 无人机状态量
        x = self.drone_states[self.pointer,0]
        y = self.drone_states[self.pointer,1]
        z = self.drone_states[self.pointer,2]
        vx = self.drone_states[self.pointer,3]
        vy = self.drone_states[self.pointer,4]
        vz = self.drone_states[self.pointer,5]
        phi = self.drone_states[self.pointer,6]
        theta = self.drone_states[self.pointer,7]
        psi = self.drone_states[self.pointer,8]
        p = self.drone_states[self.pointer,9]
        q = self.drone_states[self.pointer,10]
        r = self.drone_states[self.pointer,11]

        # 绳方向状态量
        px = self.drone_states[self.pointer, 12]
        py = self.drone_states[self.pointer, 13]
        pz = self.drone_states[self.pointer, 14]
        dpx = self.drone_states[self.pointer, 15]
        dpy = self.drone_states[self.pointer, 16]
        dpz = self.drone_states[self.pointer, 17]

        # 负载状态量
        xl = self.drone_states[self.pointer, 18]
        yl = self.drone_states[self.pointer, 19]
        zl = self.drone_states[self.pointer, 20]
        vxl = self.drone_states[self.pointer, 21]
        vyl = self.drone_states[self.pointer, 22]
        vzl = self.drone_states[self.pointer, 23]

        # 欧拉角对应的旋转矩阵
        R_d_angle = np.array([[1,tan(theta)*sin(phi),tan(theta)*cos(phi)],\
                             [0,cos(phi),-sin(phi)],\
                             [0,sin(phi)/cos(theta),cos(phi)/cos(theta)]])

        # body系转向earth系旋转矩阵
        R_E_B = np.array([[cos(theta)*cos(psi),cos(theta)*sin(psi),-sin(theta)],\
                          [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi),sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi),sin(phi)*cos(theta)],\
                          [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi),cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi),cos(phi)*cos(theta)]])

        # 计算dx时需要当前绳方向向量及其导数
        P = np.array([px, py, pz])
        d_p = np.array([dpx, dpy, dpz])

        # 计算各状态量导数1
        d_position = np.array([vx,vy,vz])
        f = R_E_B.transpose()@np.array([.0,.0,T])
        d_position_l = np.array([vxl, vyl, vzl])
        d_velocity_l = np.array([.0,.0,self.g]) + ((np.dot(P, f)-self.m*self.l*np.dot(d_p, d_p))*P)/(self.m + self.ml)    # 加速度式子有改动
        dd_p = -np.dot(d_p, d_p)*P + np.cross(P, np.cross(P, f))/self.m  # 方向向量微分方程

        # 风阻项（可选）
        # r = 0.2 # 风阻系数
        # print('dd_p1 = ', dd_p)
        # dd_p = dd_p - r*d_p # 加入风阻后修正加速度项
        # print('dd_p2 = ', dd_p)
        # dd_p = r*dd_p   # 另一种修正风阻的方法

        # 计算各状态量导数2
        d_velocity = d_velocity_l - self.l*dd_p   # 无人机加速度与负载加速度关系式
        d_angle = R_d_angle@np.array([p,q,r])
        d_q = np.linalg.inv(self.I)@(M-np.cross(np.array([p,q,r]),self.I@np.array([p,q,r])))

        # 计算绳中拉力，用于debug
        T1 = self.m*d_velocity - f - self.m*np.array([.0, .0, self.g])
        T2 = self.ml*np.array([.0, .0, self.g]) - self.ml*d_velocity_l

        # 获得当前步状态增量
        dx = np.concatenate((d_position, d_velocity, d_angle, d_q, d_p, dd_p, d_position_l, d_velocity_l))

        return dx 




    def rate_controller(self,cmd):
        # Input: cmd np.array (3,) rate commands
        # Output: M np.array (3,) moments  
        current_rate_err = cmd - self.drone_states[self.pointer, 9:12]
        kp = np.array([0.5, 0.5, 1.5])

        M = kp*current_rate_err
        return M 



    def attitude_controller(self,cmd):
        # Input: cmd np.array (3,) attitude commands
        # Output: M np.array (3,) rate commands

        current_att_err = cmd - self.drone_states[self.pointer, 6:9]
        kp = np.array([8, 8, 12])

        self.rate_cmd[self.pointer, :] = kp*current_att_err
        


    def velocity_controller(self,cmd):
        # Input: cmd np.array (3,) velocity commands
        # Output: M np.array (2,) phi and theta commands and thrust cmd
        
        self.vel_err[self.pointer, :] = cmd - self.drone_states[self.pointer, 3:6]

        kp = np.array([0.2, -0.2, 15])
        ki = np.array([0.0000, 0.0000, 0.01])
        kd = np.array([0, 0, 0.01])
        
        self.attitude_cmd[self.pointer, 0:2] = kp[0:2] * self.vel_err[self.pointer, -2::-1] \
                                             + ki[0:2] * self.vel_err.sum(axis = 0)[-2::-1] \
                                             + kd[0:2] * (self.vel_err[self.pointer, -2::-1] - self.vel_err[self.pointer - 1, -2::-1])

        thrust_cmd = kp[2] * self.vel_err[self.pointer, 2] - self.m * self.g \
                   + ki[2] * self.vel_err.sum(axis = 0)[2] \
                   + kd[2] * (self.vel_err[self.pointer, 2] - self.vel_err[self.pointer - 1, 2])

        return thrust_cmd



    def position_controller(self,cmd):
        # Input: cmd np.array (3,) position commands
        # Output: M np.array (3,) velocity commands
        
        current_pos_err = cmd - self.drone_states[self.pointer, 0:3]
        self.pos_err[self.pointer, :] = cmd - self.drone_states[self.pointer, 0:3]
        kp = np.array([0.6, 0.6, 1.5])  
        ki = np.array([0.000, 0.000, 0.0])
        kd = np.array([0., 0., 0.0])    

        self.velocity_cmd[self.pointer, :] = kp*current_pos_err
        self.velocity_cmd[self.pointer, :] = kp * self.pos_err[self.pointer, :] \
                                           + ki * self.pos_err.sum(axis = 0) \
                                           + kd * (self.pos_err[self.pointer, :] - self.pos_err[self.pointer-1, :])



    def plot_states(self):
        fig1, ax1 = plt.subplots(4,3)
        self.position_cmd[-1] = self.position_cmd[-2]
        ax1[0,0].plot(self.time,self.drone_states[:,0],label='real', linewidth=8)
        ax1[0,0].plot(self.time,self.position_cmd[:,0],label='cmd', linewidth=8)
        ax1[0,0].set_ylabel('x[m]',fontsize=25)
        ax1[0,0].tick_params(axis='both', which='major', labelsize=18)
        ax1[0,1].plot(self.time,self.drone_states[:,1], linewidth=8)
        ax1[0,1].plot(self.time,self.position_cmd[:,1], linewidth=8)
        ax1[0,1].set_ylabel('y[m]',fontsize=25)
        ax1[0,1].tick_params(axis='both', which='major', labelsize=18)
        ax1[0,2].plot(self.time,self.drone_states[:,2], linewidth=8)
        ax1[0,2].plot(self.time,self.position_cmd[:,2], linewidth=8)
        ax1[0,2].set_ylabel('z[m]',fontsize=25)
        ax1[0,2].tick_params(axis='both', which='major', labelsize=18)
        ax1[0,0].legend()

        self.velocity_cmd[-1] = self.velocity_cmd[-2]
        ax1[1,0].plot(self.time,self.drone_states[:,3], linewidth=8)
        ax1[1,0].plot(self.time,self.velocity_cmd[:,0], linewidth=8)
        ax1[1,0].set_ylabel('vx[m/s]',fontsize=25)
        ax1[1,0].tick_params(axis='both', which='major', labelsize=18)
        ax1[1,1].plot(self.time,self.drone_states[:,4], linewidth=8)
        ax1[1,1].plot(self.time,self.velocity_cmd[:,1], linewidth=8)
        ax1[1,1].set_ylabel('vy[m/s]',fontsize=25)
        ax1[1,1].tick_params(axis='both', which='major', labelsize=18)
        ax1[1,2].plot(self.time,self.drone_states[:,5], linewidth=8)
        ax1[1,2].plot(self.time,self.velocity_cmd[:,2], linewidth=8)
        ax1[1,2].set_ylabel('vz[m/s]',fontsize=25)
        ax1[1,2].tick_params(axis='both', which='major', labelsize=18)

        self.attitude_cmd[-1] = self.attitude_cmd[-2]
        ax1[2,0].plot(self.time,self.drone_states[:,6], linewidth=8)
        ax1[2,0].plot(self.time,self.attitude_cmd[:,0], linewidth=8)
        ax1[2,0].set_ylabel('phi[rad]',fontsize=25)
        ax1[2,0].tick_params(axis='both', which='major', labelsize=18)
        ax1[2,1].plot(self.time,self.drone_states[:,7], linewidth=8)
        ax1[2,1].plot(self.time,self.attitude_cmd[:,1], linewidth=8)
        ax1[2,1].set_ylabel('theta[rad]',fontsize=25)
        ax1[2,1].tick_params(axis='both', which='major', labelsize=18)
        ax1[2,2].plot(self.time,self.drone_states[:,8], linewidth=8)
        ax1[2,2].plot(self.time,self.attitude_cmd[:,2], linewidth=8)
        ax1[2,2].set_ylabel('psi[rad]',fontsize=25)
        ax1[2,2].tick_params(axis='both', which='major', labelsize=18)

        self.rate_cmd[-1] = self.rate_cmd[-2]
        ax1[3,0].plot(self.time,self.drone_states[:,9], linewidth=8)
        ax1[3,0].plot(self.time,self.rate_cmd[:,0], linewidth=8)
        ax1[3,0].set_ylabel('p[rad/s]',fontsize=25)
        ax1[3,0].tick_params(axis='both', which='major', labelsize=18)
        ax1[3,1].plot(self.time,self.drone_states[:,10], linewidth=8)
        ax1[3,1].plot(self.time,self.rate_cmd[:,1], linewidth=8)
        ax1[3,1].tick_params(axis='both', which='major', labelsize=18)
        ax1[3,1].set_ylabel('q[rad/s]',fontsize=25)
        ax1[3,2].plot(self.time,self.drone_states[:,11], linewidth=8)
        ax1[3,2].plot(self.time,self.rate_cmd[:,2], linewidth=8)
        ax1[3,2].set_ylabel('r[rad/s]',fontsize=25)
        ax1[3,2].tick_params(axis='both', which='major', labelsize=18)

        fig2, ax2 = plt.subplots(4,3)
        ax2[0,0].plot(self.time,self.drone_states[:,12], linewidth=8)
        ax2[0,0].set_ylabel('px[m]',fontsize=25)
        ax2[0,0].tick_params(axis='both', which='major', labelsize=18)
        ax2[0,1].plot(self.time,self.drone_states[:,13], linewidth=8)
        ax2[0,1].set_ylabel('py[m]',fontsize=25)
        ax2[0,1].tick_params(axis='both', which='major', labelsize=18)
        ax2[0,2].plot(self.time,self.drone_states[:,14], linewidth=8)
        ax2[0,2].set_ylabel('pz[m]',fontsize=25)    
        ax2[0,2].tick_params(axis='both', which='major', labelsize=18)

        ax2[1,0].plot(self.time,self.drone_states[:,15], linewidth=8)
        ax2[1,0].set_ylabel('v_px[m]',fontsize=25)
        ax2[1,0].tick_params(axis='both', which='major', labelsize=18)
        ax2[1,1].plot(self.time,self.drone_states[:,16], linewidth=8)
        ax2[1,1].set_ylabel('v_py[m]',fontsize=25)
        ax2[1,1].tick_params(axis='both', which='major', labelsize=18)
        ax2[1,2].plot(self.time,self.drone_states[:,17], linewidth=8)
        ax2[1,2].set_ylabel('v_pz[m]',fontsize=25)
        ax2[1,2].tick_params(axis='both', which='major', labelsize=18)
        

        ax2[2,0].plot(self.time,self.drone_states[:,18], linewidth=8)
        ax2[2,0].set_ylabel('xl[m]',fontsize=25)
        ax2[2,0].tick_params(axis='both', which='major', labelsize=18)
        ax2[2,1].plot(self.time,self.drone_states[:,19], linewidth=8)
        ax2[2,1].set_ylabel('yl[m]',fontsize=25)
        ax2[2,1].tick_params(axis='both', which='major', labelsize=18)
        ax2[2,2].plot(self.time,self.drone_states[:,20], linewidth=8)
        ax2[2,2].set_ylabel('zl[m]',fontsize=25)
        ax2[2,2].tick_params(axis='both', which='major', labelsize=18)
        

        ax2[3,0].plot(self.time,self.drone_states[:,21], linewidth=8)
        ax2[3,0].set_ylabel('vl_x[m]',fontsize=25)
        ax2[3,0].tick_params(axis='both', which='major', labelsize=18)
        ax2[3,1].plot(self.time,self.drone_states[:,22], linewidth=8)
        ax2[3,1].set_ylabel('vl_y[m]',fontsize=25)
        ax2[3,1].tick_params(axis='both', which='major', labelsize=18)
        ax2[3,2].plot(self.time,self.drone_states[:,23], linewidth=8)
        ax2[3,2].set_ylabel('vl_z[m]',fontsize=25)
        ax2[3,2].tick_params(axis='both', which='major', labelsize=18)



if __name__ == "__main__":
    drone = DroneControlSim()
    drone.run()
    drone.plot_states()
    plt.show()
    