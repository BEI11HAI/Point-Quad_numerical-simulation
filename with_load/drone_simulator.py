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

            # self.velocity_cmd[self.pointer, :] = np.array([1, 1, 1])    # test velocity loop
            # if self.pointer % 10 == 0:
            #     print('velocity = ', self.drone_states[self.pointer, 3:6])    # test velocity loop
            thrust_cmd = self.velocity_controller(self.velocity_cmd[self.pointer, :])

            # thrust_cmd = -1*self.g*self.m   # test attitude loop
            # self.attitude_cmd[self.pointer, :] = np.array([2, 2, 1])    # test attitude loop
            # print('attitude = ', self.drone_states[self.pointer, 6:9])    # test attitude loop
            self.attitude_controller(self.attitude_cmd[self.pointer, :])


            # self.rate_cmd[self.pointer, :] = np.array([1.0, 1.0, 1.0])  # test rate loop
            # print('rate = ', self.drone_states[self.pointer, 9: 12])  # test rate loop
            M = self.rate_controller(self.rate_cmd[self.pointer, :])

            dx = self.drone_dynamics(thrust_cmd, M)

            # if np.linalg.norm(self.drone_states[self.pointer, 12:15]) > 1:    # 方向向量模长超过1
                # self.drone_states[self.pointer, 12:15] = self.drone_states[self.pointer, 12:15] / np.linalg.norm(self.drone_states[self.pointer, 12:15])
            #     print('P1 = ', self.drone_states[self.pointer, 12:15])

            self.drone_states[self.pointer+1, ] = self.drone_states[self.pointer, ] + self.sim_step*dx

            # if np.linalg.norm(self.drone_states[self.pointer+1, 12:15]) > 1:    # 方向向量模长超过1
            #     self.drone_states[self.pointer, 12:15] = self.drone_states[self.pointer, 12:15] / np.linalg.norm(self.drone_states[self.pointer+1, 12:15])
            #     dx[12:15] = dx[12:15] / np.linalg.norm(self.drone_states[self.pointer+1, 12:15])
            #     self.drone_states[self.pointer+1, 12:15] = self.drone_states[self.pointer, 12:15] + self.sim_step*dx[12:15]

                # print('P2 = ', self.drone_states[self.pointer + 1, 12:15])

            # print("thrust = ", thrust_cmd)


            
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


        R_d_angle = np.array([[1,tan(theta)*sin(phi),tan(theta)*cos(phi)],\
                             [0,cos(phi),-sin(phi)],\
                             [0,sin(phi)/cos(theta),cos(phi)/cos(theta)]])


        R_E_B = np.array([[cos(theta)*cos(psi),cos(theta)*sin(psi),-sin(theta)],\
                          [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi),sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi),sin(phi)*cos(theta)],\
                          [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi),cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi),cos(phi)*cos(theta)]])

        P = np.array([px, py, pz])
        # print('P1 = ', P)
        # if np.linalg.norm(P) > 1:    # 方向向量模长超过1
            # P = P / np.linalg.norm(P)
            # print('P2 = ', P)

        

        d_p = np.array([dpx, dpy, dpz])

        # print('P*d_p = ', P*d_p)
        d_position = np.array([vx,vy,vz])
        # d_velocity = np.array([.0,.0,self.g]) + R_E_B.transpose()@np.array([.0,.0,T])/self.m
        f = R_E_B.transpose()@np.array([.0,.0,T])
        d_position_l = np.array([vxl, vyl, vzl])
        d_velocity_l = np.array([.0,.0,self.g]) + ((np.dot(P, f)-self.m*self.l*np.dot(d_p, d_p))*P)/(self.m + self.ml)    # 加速度式子有改动
        # print('P = ', P)
        dd_p = -np.dot(d_p, d_p)*P + np.cross(P, np.cross(P, f))/self.m  # 方向向量微分方程

        # r = 0.2 # 风阻系数（貌似有无风阻影响不大）
        # print('dd_p1 = ', dd_p)
        # dd_p = dd_p - r*d_p # 加入风阻后修正加速度项
        # print('dd_p2 = ', dd_p)
        # dd_p = r*dd_p   # 另一种修正风阻的方法

        d_velocity = d_velocity_l - self.l*dd_p   # 无人机加速度与负载加速度关系式
        
        
        d_angle = R_d_angle@np.array([p,q,r])
        d_q = np.linalg.inv(self.I)@(M-np.cross(np.array([p,q,r]),self.I@np.array([p,q,r])))

        T1 = self.m*d_velocity - f - self.m*np.array([.0, .0, self.g])
        T2 = self.ml*np.array([.0, .0, self.g]) - self.ml*d_velocity_l
        print('T1 = ',T1, ', T2 = ', T2)
        print('T1 - T2', T1 - T2)
        # print('postion = ', self.drone_states[self.pointer, 0:3])
        # print('positon_l = ', self.drone_states[self.pointer, 18:21])
        print('d_position_l = ', d_position_l)
        print('d_velocity_l = ', d_velocity_l)
        # print('P = ', self.drone_states[self.pointer, 12:15])
        # print('d_position_l - d_position', self.drone_states[self.pointer, 18:21] - self.drone_states[self.pointer, 3:6])
        # print('d_p = ', d_p)
        print('position - position_l = ', self.drone_states[self.pointer, 0:3] - self.drone_states[self.pointer, 18:21])
        # print('error = ', self.drone_states[self.pointer, 0:3] - self.drone_states[self.pointer, 18:21] + self.drone_states[self.pointer, 12:15])
        print('')

        # if self.pointer == 0:
        #     print('error_0 = ', self.drone_states[0, 0:3] - self.drone_states[0, 18:21] - self.drone_states[0, 12:15])

        # d_p = np.array([dpx, dpy, dpz])
        
        

        

        # if self.pointer <= 150:
        #     print('d_velocity = ', d_velocity)
        #     print('2nd_part = ', ((np.dot(P, f)-self.m*self.l*np.dot(d_p, d_p))*P)/(self.m + self.ml))
        #     print('np.dot(P, f) = ', np.dot(P, f))
        #     print('self.m*self.l*np.dot(d_p, d_p)', (self.m*self.l*np.dot(d_p, d_p)))
        #     print('P = ', P)
        #     print('(np.dot(P, f)-self.m*self.l*np.dot(d_p, d_p))*P = ', (np.dot(P, f)-self.m*self.l*np.dot(d_p, d_p))*P)
        #     print('d_p = ', d_p)
        #     print('f = ', f)
        #     print('')



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
        # for i in range(3):  
        #     if cmd[i] > 1:
        #         cmd[i] = 1

        # for i in range(3):  
        #     if cmd[i] < -1:
        #         cmd[i] = -1

        current_att_err = cmd - self.drone_states[self.pointer, 6:9]
        kp = np.array([8, 8, 12])

        self.rate_cmd[self.pointer, :] = kp*current_att_err
        


    def velocity_controller(self,cmd):
        # Input: cmd np.array (3,) velocity commands
        # Output: M np.array (2,) phi and theta commands and thrust cmd
        self.vel_err[self.pointer, :] = cmd - self.drone_states[self.pointer, 3:6]

        kp = np.array([0.2, -0.2, 15])
        # kp = np.array([0.5, 0.5, 5])    # p无限幅时参数
        ki = np.array([0.0000, 0.0000, 0.01])
        kd = np.array([0, 0, 0.01])
        # ki = np.array([0.0002, -0.0002, 0.0002])
        # kd = np.array([0.3, 0.3, 0.5])
        # kp = np.array([1.3, -1.2, 7])
        # ki = np.array([0.0001, -0.0001, 0.0001])
        # kd = np.array([5, 5, 1])
        
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
        kp = np.array([0.6, 0.6, 1.5])  # 
        # kp = np.array([2.2, 2.35, 2.5])
        ki = np.array([0.000, 0.000, 0.0])
        kd = np.array([0., 0., 0.0])    # 

        self.velocity_cmd[self.pointer, :] = kp*current_pos_err
        self.velocity_cmd[self.pointer, :] = kp * self.pos_err[self.pointer, :] \
                                           + ki * self.pos_err.sum(axis = 0) \
                                           + kd * (self.pos_err[self.pointer, :] - self.pos_err[self.pointer-1, :])



    def plot_states(self):
        fig1, ax1 = plt.subplots(4,3)
        self.position_cmd[-1] = self.position_cmd[-2]
        ax1[0,0].plot(self.time,self.drone_states[:,0],label='real')
        ax1[0,0].plot(self.time,self.position_cmd[:,0],label='cmd')
        ax1[0,0].set_ylabel('x[m]')
        ax1[0,1].plot(self.time,self.drone_states[:,1])
        ax1[0,1].plot(self.time,self.position_cmd[:,1])
        ax1[0,1].set_ylabel('y[m]')
        ax1[0,2].plot(self.time,self.drone_states[:,2])
        ax1[0,2].plot(self.time,self.position_cmd[:,2])
        ax1[0,2].set_ylabel('z[m]')
        ax1[0,0].legend()

        self.velocity_cmd[-1] = self.velocity_cmd[-2]
        ax1[1,0].plot(self.time,self.drone_states[:,3])
        ax1[1,0].plot(self.time,self.velocity_cmd[:,0])
        ax1[1,0].set_ylabel('vx[m/s]')
        ax1[1,1].plot(self.time,self.drone_states[:,4])
        ax1[1,1].plot(self.time,self.velocity_cmd[:,1])
        ax1[1,1].set_ylabel('vy[m/s]')
        ax1[1,2].plot(self.time,self.drone_states[:,5])
        ax1[1,2].plot(self.time,self.velocity_cmd[:,2])
        ax1[1,2].set_ylabel('vz[m/s]')

        self.attitude_cmd[-1] = self.attitude_cmd[-2]
        ax1[2,0].plot(self.time,self.drone_states[:,6])
        ax1[2,0].plot(self.time,self.attitude_cmd[:,0])
        ax1[2,0].set_ylabel('phi[rad]')
        ax1[2,1].plot(self.time,self.drone_states[:,7])
        ax1[2,1].plot(self.time,self.attitude_cmd[:,1])
        ax1[2,1].set_ylabel('theta[rad]')
        ax1[2,2].plot(self.time,self.drone_states[:,8])
        ax1[2,2].plot(self.time,self.attitude_cmd[:,2])
        ax1[2,2].set_ylabel('psi[rad]')

        self.rate_cmd[-1] = self.rate_cmd[-2]
        ax1[3,0].plot(self.time,self.drone_states[:,9])
        ax1[3,0].plot(self.time,self.rate_cmd[:,0])
        ax1[3,0].set_ylabel('p[rad/s]')
        ax1[3,1].plot(self.time,self.drone_states[:,10])
        ax1[3,1].plot(self.time,self.rate_cmd[:,1])
        ax1[3,0].set_ylabel('q[rad/s]')
        ax1[3,2].plot(self.time,self.drone_states[:,11])
        ax1[3,2].plot(self.time,self.rate_cmd[:,2])
        ax1[3,0].set_ylabel('r[rad/s]')

        fig2, ax2 = plt.subplots(4,3)
        ax2[0,0].plot(self.time,self.drone_states[:,12])
        ax2[0,0].set_ylabel('px[m]')
        ax2[0,1].plot(self.time,self.drone_states[:,13])
        ax2[0,1].set_ylabel('py[m]')
        ax2[0,2].plot(self.time,self.drone_states[:,14])
        ax2[0,2].set_ylabel('pz[m]')    

        ax2[1,0].plot(self.time,self.drone_states[:,15])
        ax2[1,0].set_ylabel('v_px[m]')
        ax2[1,1].plot(self.time,self.drone_states[:,16])
        ax2[1,1].set_ylabel('v_py[m]')
        ax2[1,2].plot(self.time,self.drone_states[:,17])
        ax2[1,2].set_ylabel('v_pz[m]')
        

        ax2[2,0].plot(self.time,self.drone_states[:,18])
        ax2[2,0].set_ylabel('xl[m]')
        ax2[2,1].plot(self.time,self.drone_states[:,19])
        ax2[2,1].set_ylabel('yl[m]')
        ax2[2,2].plot(self.time,self.drone_states[:,20])
        ax2[2,2].set_ylabel('zl[m]')
        

        ax2[3,0].plot(self.time,self.drone_states[:,21])
        ax2[3,0].set_ylabel('vl_x[m]')
        ax2[3,1].plot(self.time,self.drone_states[:,22])
        ax2[3,1].set_ylabel('vl_y[m]')
        ax2[3,2].plot(self.time,self.drone_states[:,23])
        ax2[3,2].set_ylabel('vl_z[m]')



if __name__ == "__main__":
    drone = DroneControlSim()
    drone.run()
    drone.plot_states()
    plt.show()
    
