import math
import matplotlib.pyplot as plt
import numpy as np

# Covariance for EKF simulation
Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2  # Q
R = np.diag([0.5, 0.5]) ** 2  # Observation covariance : std_range, std_bearing

#  Simulation parameter
Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2  # Sensor Noise
R_sim = np.diag([1.0, np.deg2rad(10.0)]) ** 2  # Input Noise

DT = 0.1  # time tick [s]
SIM_TIME = 60.0  # simulation time [s]
MAX_RANGE = 15.0  # maximum observation range
M_DIST_TH = 2.0  # Threshold of distance
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]

show_animation = True


def ekf_slam(xEst, PEst, u, z):
    # Predict
    S = STATE_SIZE  # State size [x,y,yaw]
    # G, Fx, jF = jacob_motion(xEst[0:S], u)
    jF = jacob_f(xEst[0:S], u)
    # x = F @ x + B @ u----------------------------------------------1
    xEst[0:S] = motion_model(xEst[0:S], u)

    # PEst[0:S, 0:S] = G.T @ PEst[0:S, 0:S] @ G + \
    #     Fx.T @ Cx @ Fx  # Cx = Q, G = jF, Fx is an an identity matrix
    # PPred = jF @ PEst @ jF.T + Q-----------------------------------2
    PEst[0:S, 0:S] = jF @ PEst[0:S, 0:S] @ jF.T + Cx

    initP = np.eye(2)

    # Update
    for iz in range(len(z[:, 0])):
        # observation model
        min_id = search_correspond_landmark_id(xEst, PEst, z[iz, 0:2])

        nLM = calc_n_lm(xEst)
        if min_id == nLM:
            print("SLAM!")
            # Extend state and covariance matrix
            xAug = np.vstack((xEst, calc_landmark_position(xEst, z[iz, :])))
            PAug = np.vstack((np.hstack((PEst, np.zeros((len(xEst), LM_SIZE)))),
                              np.hstack((np.zeros((LM_SIZE, len(xEst))), initP))))
            xEst = xAug  # Est with LM
            PEst = PAug
        lm = get_landmark_position_from_state(xEst, min_id)  # [x, y, theta]
        y, S, jH = calc_innovation(lm, xEst, PEst, z[iz, 0:2], min_id)
        # ----------------------------------------------------------------

        # K = PPred @ jH.T @ inv(S = jH @ PPred @ jH.T + R)---3
        K = (PEst @ jH.T) @ np.linalg.inv(S)
        # xEst = xPred + K @ (z - zPred)----------------------4
        xEst = xEst + (K @ y)
        # PEst = (I - K @ jH) @ PPred-------------------------5
        PEst = (np.eye(len(xEst)) - (K @ jH)) @ PEst

    xEst[2] = pi_2_pi(xEst[2])

    return xEst, PEst


def calc_input():  # u = [v, w]
    v = 1.0  # [m/s]
    yaw_rate = 0.1  # [rad/s]
    u = np.array([[v, yaw_rate]]).T
    return u


def observation(xTrue, xd, u, landmark_pos):  # observation_model(laser)
    xTrue = motion_model(xTrue, u)  # x state 3*1

    # init laser
    z = np.zeros((0, 3))
    # z = observation_model(xTrue, landmark_pos) + \
    #     LASER_NOISE @ np.random.randn(2, 1)  # laser

    for i in range(len(landmark_pos[:, 0])):

        dx = landmark_pos[i, 0] - xTrue[0, 0]
        dy = landmark_pos[i, 1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5  # add sensor noise
            angle_n = angle + np.random.randn() * \
                Q_sim[1, 1] ** 0.5  # add sensor noise
            zi = np.array([dn, angle_n, i])
            z = np.vstack((z, zi))
    # ---------------------------------------------

    # add noise to input motion cmd
    # R_sim = Input Noise
    ud = np.array([[
        u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5,
        u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5]]).T

    xd = motion_model(xd, ud)
    return xTrue, z, xd, ud  # x_origin, z_add, x_add, motion cmd_add


# def observation_model(x, landmark_pos):   # observation_model(laser)
#     px = landmark_pos[0]
#     py = landmark_pos[1]
#     dist = np.sqrt((px - x[0, 0])**2 + (py - x[1, 0])**2)

#     z = np.array([[dist],
#                   [math.atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]])

#     # z = H @ x

#     return z    # H @ x


def motion_model(x, u):  # motion_model
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT]])  # DT = delta T

    x = F @ x + B @ u
    return x


def calc_n_lm(x):
    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n    # the number of landmarks n


# def jacob_motion(x, u):
#     # Fx is an an identity matrix of size (STATE_SIZE)
#     Fx = np.hstack((np.eye(STATE_SIZE), np.zeros(
#         (STATE_SIZE, LM_SIZE * calc_n_lm(x)))))

#     jF = np.array([[0.0, 0.0, -DT * u[0, 0] * math.sin(x[2, 0])],
#                    [0.0, 0.0, DT * u[0, 0] * math.cos(x[2, 0])],
#                    [0.0, 0.0, 0.0]], dtype=float)

#     G = np.eye(STATE_SIZE) + Fx.T @ jF @ Fx

#     return G, Fx, jF


def jacob_f(x, u):  # motion conv
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw)],
        [0.0, 0.0, 1.0]])

    return jF


def calc_landmark_position(x, z):
    zp = np.zeros((2, 1))  # 2*1
    # z: [range; bearing]

    zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
    zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])

    return zp


def get_landmark_position_from_state(x, ind):  # LM pose
    lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE +
           LM_SIZE * (ind + 1), :]  # [x, y, theta]

    return lm


def search_correspond_landmark_id(xAug, PAug, zi):  # xEst, PEst, z[iz, 0:2]

    nLM = calc_n_lm(xAug)  # the number of landmarks n we detect

    min_dist = []

    for i in range(nLM):
        lm = get_landmark_position_from_state(xAug, i)  # LM = [x, y, theta]
        y, S, H = calc_innovation(lm, xAug, PAug, zi, i)
        min_dist.append(y.T @ np.linalg.inv(S) @ y)

    min_dist.append(M_DIST_TH)  # prevent generate double new_LM

    min_id = min_dist.index(min(min_dist))

    return min_id


def calc_innovation(lm, xEst, PEst, z, LMid):  # update
    delta = lm - xEst[0:2]  # the error between true LM and EST_LM
    q = (delta.T @ delta)[0, 0]  # 4*2
    z_angle = math.atan2(delta[1, 0], delta[0, 0]) - \
        xEst[2, 0]  # error with angle(true LM and EST_LM)
    zp = np.array([[math.sqrt(q), pi_2_pi(z_angle)]])  # zp = H * X_head
    y = (z - zp).T  # y = z_add - zPred
    y[1] = pi_2_pi(y[1])
    jH = jacob_h(q, delta, xEst, LMid + 1)  # jH
    S = jH @ PEst @ jH.T + R  # S

    return y, S, jH


def jacob_h(q, delta, x, i):  # Jacobian of Observation Model
    sq = math.sqrt(q)  # delta
    G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                  [delta[1, 0], - delta[0, 0], - q, - delta[1, 0], delta[0, 0]]])  # 2*5

    G = G / q
    nLM = calc_n_lm(x)
    F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))  # 3*(3+2*..)
    F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
                    np.eye(2), np.zeros((2, 2 * nLM - 2 * i))))  # 2*...

    F = np.vstack((F1, F2))  # 5*...

    jH = G @ F  # (2*5) @ (5*...)

    return jH


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main():
    print(__file__ + " EKF SLAM start!")

    time = 0.0
    # landmark_pos = np.array([[15.0, 10.0],
    #                          [3.0, 15.0],
    #                          [-5.0, 20.0]])

    landmark_pos = np.array([[10.0, -2.0],
                             [15.0, 10.0],
                             [3.0, 15.0],
                             [-5.0, 20.0],
                             [10.0, 5.0],
                             [-5.0, 5.0],
                             [-5.0, 10.0], ])

    # State Vector [x y yaw v]'
    xEst = np.zeros((STATE_SIZE, 1))  # 3*1
    xTrue = np.zeros((STATE_SIZE, 1))   # 3*1
    PEst = np.eye(STATE_SIZE)  # [[1 0 0], [0 1 0], [0 0 1]]

    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue

    while SIM_TIME >= time:
        time += DT
        u = calc_input()  # call motion cmd [v, w]

        # add noise to x_origin, z_add, x_add, motion cmd_add
        xTrue, z, xDR, ud = observation(xTrue, xDR, u, landmark_pos)

        # return xEst, PEst(xt+1, Pt+1)
        xEst, PEst = ekf_slam(xEst, PEst, ud, z)

        x_state = xEst[0:STATE_SIZE]

        # store data history
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            # obstacles black stars
            plt.plot(landmark_pos[:, 0], landmark_pos[:, 1],
                     "*k", label='landmarks')
            plt.plot(xEst[0], xEst[1], ".r")

            # plot landmark
            for i in range(calc_n_lm(xEst)):
                plt.plot(xEst[STATE_SIZE + i * 2],
                         xEst[STATE_SIZE + i * 2 + 1], "xg")  # Est LM

            plt.plot(hxTrue[0, :],
                     hxTrue[1, :], "-b", label='ground truth')  # ground truth blue line(-)
            # plt.plot(hxDR[0, :],
            #          hxDR[1, :], "-k")
            plt.plot(hxEst[0, :],
                     hxEst[1, :], "-r", label='EKF Est path')  # X state from ekf red line(-)
            plt.axis("equal")
            plt.title('EKF SLAM')
            plt.legend()
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()
