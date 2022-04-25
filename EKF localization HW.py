import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# Covariance for EKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance 
R = np.diag([1.0, 0.8]) ** 2  # Observation covariance : std_range, std_bearing

#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(20.0)]) ** 2
LASER_NOISE = np.diag([0.5, 0.5]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 20.0  # simulation time [s]

show_animation = True


def calc_input(): # u = [v, w]
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


def observation(xTrue, xd, u, landmark_pos):
    xTrue = motion_model(xTrue, u)  # x state 4*1

    # add noise to laser
    z = observation_model(xTrue, landmark_pos) +  LASER_NOISE @ np.random.randn(2, 1)     # 0~1之間 2*1

    # add noise to input motion cmd
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)     # 0~1之間 2*1

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud # x_origin, z_add, x_add, motion cmd_add


def motion_model(x, u): # motion_model
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])  # DT = delta T

    x = F @ x + B @ u   # 矩陣乘法, @ = .dot

    return x


def observation_model(x, landmark_pos):   # observation_model(laser)
    px = landmark_pos[0]
    py = landmark_pos[1]
    dist = np.sqrt((px - x[0, 0])**2 + (py - x[1, 0])**2)

    z = np.array([[dist],
                [math.atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]])

    # z = H @ x

    return z    # H @ x


def jacob_f(x, u):  # motion conv 
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF


def jacob_h(x, landmark_pos):  # Observation conv
    # Jacobian of Observation Model
    px = landmark_pos[0]
    py = landmark_pos[1]
    hyp = (px - x[0, 0])**2 + (py - x[1, 0])**2
    dist = np.sqrt(hyp)

    jH = np.array(
        [[-(px - x[0, 0]) / dist, -(py - x[1, 0]) / dist, 0, 0],
         [ (py - x[1, 0]) / hyp,  -(px - x[0, 0]) / hyp, -1, 0]])

    return jH


def ekf_estimation(xEst, PEst, z, u, landmark_pos):   # xEst = 4*1, PEst = eye(4)
    #  Predict
    xPred = motion_model(xEst, u)   # x = F @ x + B @ u-----------------------------1
    jF = jacob_f(xEst, u)   #　jF(t = 0), xEst(t = t-1), u = u_add
    PPred = jF @ PEst @ jF.T + Q    # PPred = jF @ PEst @ jF.T + Q------------------2

    #  Update
    jH = jacob_h(xEst, landmark_pos)
    zPred = observation_model(xPred, landmark_pos)    # zPred = H @ x 
    y = z - zPred   # z_add - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S) #  K = PPred @ jH.T @ inv(S = jH @ PPred @ jH.T + R)---3
    xEst = xPred + K @ y    # xEst = xPred + K @ (z - zPred)-----------------------------------4
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred # PEst = (I - K @ jH) @ PPred------------------5
    return xEst, PEst


def plot_covariance_ellipse(xEst, PEst):  # plot covariance橢圓(measure updated)
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--g")


def main():
    print(__file__ + " EKF localization start!")

    time = 0.0

    # obstacles positions [x, y] 
    # landmark_pos = np.array([[5, 2]])
    landmark_pos = np.array([[5, 10], [10, 5], [15, 15]])   
    # landmark_pos = np.array([[-15, 10], [-10, 0], [0, -5], [0, 5], [0, 15], [0, 25], [10, 0], [10, 20]])

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1)) # 4*1
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)    #[[1 0 0], [0 1 0], [0 0 1]...]

    xDR = np.zeros((4, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))   # observation 2*1

    while SIM_TIME >= time: 
        time += DT  # DT = 0.1
        u = calc_input()    # call motion cmd [v, w]

        for lmark in landmark_pos:
            xTrue, z, xDR, ud = observation(xTrue, xDR, u, lmark)  # add noise to x_origin, z_add, x_add, motion cmd_add

            xEst, PEst = ekf_estimation(xEst, PEst, z, ud, lmark)  # return xEst, PEst(xt+1, Pt+1)

            # store data history # hstack: 水平堆疊
            hxEst = np.hstack((hxEst, xEst))    # 此時和上一個時刻之X state from ekf
            hxDR = np.hstack((hxDR, xDR))   #　xTrue, xDR(Dead reckoning)
            hxTrue = np.hstack((hxTrue, xTrue)) # ground truth
            hz = np.hstack((hz, z)) # green point

        if show_animation:
            plt.cla()   # 取消上一時間動作
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            
            plt.plot(landmark_pos[:, 0], landmark_pos[:, 1], "*k")    # obstacles black star

            # plt.plot(hz[0, :], hz[1, :], ".g")  # point, green (the pose of landmark_pos)
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b")  # ground truth blue line(-)
            # plt.plot(hxDR[0, :].flatten(),
            #          hxDR[1, :].flatten(), "-k")    # Dead reckoning black line(-)
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r")   # X state from ekf red line(-)
            plot_covariance_ellipse(xEst, PEst) # plot covariance橢圓
            plt.axis("equal")
            plt.legend(['landmarks','ground truth','EKF estimated', 'covariance ellipse'])
            plt.title('EKF localization') 
            plt.grid(True)
            plt.pause(0.1)


if __name__ == '__main__':
    main()