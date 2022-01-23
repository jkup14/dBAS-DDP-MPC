from helper_funcs import *

class MyPlanningPolicy:
    """
    Command accelerations to the drone to follow the subject according to the
    desired settings and avoid the obstacle.
    """

    def __init__(self, params):
        self.params = params
        self.u_prev = None
        self.subject_positions = np.array([[], []])
        self.subject_velocities = np.array([[], []])
        self.obstacle_positions = np.array([[], []])
        self.obstacle_velocities = np.array([[], []])
        self.range = self.params["desired_range_to_subject"]
        self.delta = self.params["obstacle_radius"]

    def getAbar(
            self,
            px: float,
            py: float,
            ox: float,
            oy: float,
            delta: float
    ) -> np.ndarray:
        dBdx = np.array([[(2 * ox - 2 * px) / (-delta ** 2 + (-ox + px) ** 2 + (-oy + py) ** 2) ** 2,
                          (2 * oy - 2 * py) / (-delta ** 2 + (-ox + px) ** 2 + (-oy + py) ** 2) ** 2,
                          0, 0, 0]])
        # dBdx = np.array([[-(-2*ox + 2*px)/(-delta**2 + (-ox + px)**2 + (-oy + py)**2),
        #                   -(-2*oy + 2*py)/(-delta**2 + (-ox + px)**2 + (-oy + py)**2), 0, 0, 0]])
        return dBdx

    def getBAS(
            self,
            drone_position: np.ndarray,
            obstacle_position: np.ndarray,
            delta: float
    ) -> float:
        BAS = 1 / (-delta ** 2 + (drone_position[0] - obstacle_position[0]) ** 2 +
                    (drone_position[1] - obstacle_position[1]) ** 2)

        # h = (-delta ** 2 + (drone_position[0] - obstacle_position[0]) ** 2 + (drone_position[1] - obstacle_position[1]) ** 2)
        # if BAS < 0:
        #     return 100
        # return -np.log(h)
        return BAS

    def fnsimulate(
            self,
            x0: np.ndarray,
            u_new: np.ndarray,
            Horizon: float,
            dt: float,
            obstacle_traj: np.ndarray,
    ) -> np.ndarray:
        x_traj = np.zeros([5, Horizon])
        x_traj[:, [0]] = x0
        for k in range(1, Horizon):
            x_traj[0, [k]] = x_traj[0, [k - 1]] + x_traj[2, [k - 1]] * dt
            x_traj[1, [k]] = x_traj[1, [k - 1]] + x_traj[3, [k - 1]] * dt
            x_traj[2, [k]] = x_traj[2, [k - 1]] + u_new[0, [k - 1]] * dt
            x_traj[3, [k]] = x_traj[3, [k - 1]] + u_new[1, [k - 1]] * dt
            x_traj[4, [k]] = self.getBAS(x_traj[0:4,[k]], obstacle_traj[:, [k]], self.delta)
        return x_traj

    def calcTargetTrajInFrontStraight(
            self,
            subject_position: np.ndarray,
            subject_velocity: np.ndarray,
            obstacle_traj: np.ndarray,
            desired_range: float,
            Horizon: float,
            dt: float
    ) -> np.ndarray:
        vx, vy = subject_velocity
        direction = np.arctan2(vy, vx)
        times = np.linspace(0, Horizon * dt - dt, Horizon)
        x_traj = subject_position[0] + desired_range * np.cos(direction) + vx * times
        y_traj = subject_position[1] + desired_range * np.sin(direction) + vy * times
        vx_traj = vx * np.ones([1, Horizon])
        vy_traj = vy * np.ones([1, Horizon])
        traj = np.vstack([x_traj, y_traj, vx_traj, vy_traj])
        # BAS_traj = np.array([self.getBAS(traj[:, [i]], obstacle_traj[:, [i]], self.delta) for i in range(Horizon)])
        # return np.vstack([traj, BAS_traj.T])
        return np.vstack([traj, np.zeros([1, Horizon])])

    def calcTargetTrajInFrontCurved(
            self,
            subject_positions: np.ndarray,
            subject_velocities: np.ndarray,
            obstacle_traj: np.ndarray,
            desired_range: float,
            Horizon: float,
            dt: float
    ) -> np.ndarray:
        x = subject_positions[0, :]
        y = subject_positions[1, :]
        vx = subject_velocities[0, :]
        vy = subject_velocities[1, :]

        # coordinates of the barycenter
        x_m = np.mean(x)
        y_m = np.mean(y)

        # calculation of the reduced coordinates
        u = x - x_m
        v = y - y_m

        # linear system defining the center (uc, vc) in reduced coordinates:
        #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
        #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
        Suv = sum(u * v)
        Suu = sum(u ** 2)
        Svv = sum(v ** 2)
        Suuv = sum(u ** 2 * v)
        Suvv = sum(u * v ** 2)
        Suuu = sum(u ** 3)
        Svvv = sum(v ** 3)

        # Solving the linear system
        A = np.array([[Suu, Suv], [Suv, Svv]])
        B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
        uc, vc = np.linalg.solve(A, B)

        xc_1 = x_m + uc
        yc_1 = y_m + vc

        # Calculate Radius of Curvature
        Ri_1 = np.sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2)
        R_1 = np.mean(Ri_1)

        # Determine CW (-1) or CCW (1):
        thetaFirst = np.arctan2(y[0] - yc_1, x[0] - xc_1)
        thetaCurr = np.arctan2(y[-1] - yc_1, x[-1] - xc_1)
        direction = -1
        if thetaCurr > thetaFirst:
            direction = 1

        # Mean Velocity:
        v_mean = np.mean(np.sqrt(vx ** 2 + vy ** 2))
        omega = v_mean / R_1
        theta_travelled = omega * Horizon * dt * direction
        theta_final = thetaCurr + theta_travelled
        theta_traj = np.linspace(thetaCurr, theta_final, Horizon)
        x_predicted = xc_1 + R_1 * np.cos(theta_traj)
        y_predicted = yc_1 + R_1 * np.sin(theta_traj)
        heading_traj = theta_traj + direction * np.pi / 2
        vx_predicted = v_mean * np.cos(heading_traj)
        vy_predicted = v_mean * np.sin(heading_traj)
        x_inFront_pred = x_predicted + desired_range * np.cos(heading_traj)
        y_inFront_pred = y_predicted + desired_range * np.sin(heading_traj)
        target_traj_predicted = np.vstack([x_inFront_pred, y_inFront_pred,
                                           vx_predicted, vy_predicted])
        # BAS_traj = np.array([self.getBAS(target_traj_predicted[:, [i]], obstacle_traj[:, [i]], self.delta) for i in range(Horizon)])
        # return np.vstack([target_traj_predicted, BAS_traj.T])
        return np.vstack([target_traj_predicted, np.zeros([1, Horizon])])


    def predictObstacleTrajectoryCurved(
            self,
            obstacle_positions: np.ndarray,
            obstacle_velocities: np.ndarray,
            Horizon: float,
            dt: float
    ) -> np.ndarray:
        x = obstacle_positions[0, :]
        y = obstacle_positions[1, :]
        vx = obstacle_velocities[0, :]
        vy = obstacle_velocities[1, :]

        # coordinates of the barycenter
        x_m = np.mean(x)
        y_m = np.mean(y)

        # calculation of the reduced coordinates
        u = x - x_m
        v = y - y_m

        # linear system defining the center (uc, vc) in reduced coordinates:
        #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
        #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
        Suv = sum(u * v)
        Suu = sum(u ** 2)
        Svv = sum(v ** 2)
        Suuv = sum(u ** 2 * v)
        Suvv = sum(u * v ** 2)
        Suuu = sum(u ** 3)
        Svvv = sum(v ** 3)

        # Solving the linear system
        A = np.array([[Suu, Suv], [Suv, Svv]])
        B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
        uc, vc = np.linalg.solve(A, B)

        xc_1 = x_m + uc
        yc_1 = y_m + vc

        # Calculate Radius of Curvature
        Ri_1 = np.sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2)
        R_1 = np.mean(Ri_1)

        # Determine CW (-1) or CCW (1):
        thetaFirst = np.arctan2(y[0] - yc_1, x[0] - xc_1)
        thetaCurr = np.arctan2(y[-1] - yc_1, x[-1] - xc_1)
        direction = -1
        if thetaCurr > thetaFirst:
            direction = 1

        # Mean Velocity:
        v_mean = np.mean(np.sqrt(vx ** 2 + vy ** 2))
        omega = v_mean / R_1
        theta_travelled = omega * Horizon * dt * direction
        theta_final = thetaCurr + theta_travelled
        theta_traj = np.linspace(thetaCurr, theta_final, Horizon)
        x_predicted = xc_1 + R_1 * np.cos(theta_traj)
        y_predicted = yc_1 + R_1 * np.sin(theta_traj)
        heading_traj = theta_traj + direction * np.pi / 2
        vx_predicted = v_mean * np.cos(heading_traj)
        vy_predicted = v_mean * np.sin(heading_traj)
        obstacle_traj_predicted = np.vstack([x_predicted, y_predicted,
                                             vx_predicted, vy_predicted])
        return obstacle_traj_predicted

    def predictObstacleTrajectoryStraight(
            self,
            obstacle_position: np.ndarray,
            obstacle_velocity: np.ndarray,
            Horizon: float,
            dt: float
    ) -> np.ndarray:
        x, y = obstacle_position
        vx, vy = obstacle_velocity
        direction = np.arctan2(vy, vx)
        times = np.linspace(0, Horizon * dt - dt, Horizon)
        x_traj = x + times * vx
        y_traj = y + times * vy
        vx_traj = vx * np.ones([1, Horizon])
        vy_traj = vy * np.ones([1, Horizon])
        traj = np.vstack([x_traj, y_traj, vx_traj, vy_traj])
        return traj

    def fnCost(
            self,
            x: np.ndarray,
            u: np.ndarray,
            j: int,
            Q: np.ndarray,
            R: np.ndarray,
            dt: float
    ) -> tuple:
        l0 = u.T @ R @ u + x.T @ Q @ x
        l_x = Q @ x
        l_xx = Q
        l_u = R @ u
        l_uu = R
        l_ux = np.zeros([self.nu, self.nx])
        return l0, l_x, l_xx, l_u, l_uu, l_ux

    def compute_accel(
            self,
            t: float,
            drone_position: np.ndarray,
            drone_velocity: np.ndarray,
            subject_position: np.ndarray,
            subject_velocity: np.ndarray,
            obstacle_position: np.ndarray,
            obstacle_velocity: np.ndarray,
            Horizon: int,
            Q_f: np.ndarray,
            R: np.ndarray
    ) -> np.ndarray:
        # Problem Definition:
        self.nx, self.nu = 5, 2
        # State and Control Transition Matrices:
        dfx = np.array([[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])
        dfu = np.array([[0, 0],
                        [0, 0],
                        [1, 0],
                        [0, 1]])

        # Time Horizon:
        # Horizon = 55

        # DDP Iterations:
        num_iter = 10

        # Discretization:
        dt = self.params["dt"]

        # Index of Current Time:
        t_ind = int(t / dt)

        # Weight in Final State:
        # Q_f = np.diag([100, 100, 0, 0, 50000])

        # Weight in the Control:
        # R = 20 * np.eye(self.nu)

        # Initial Configuration:
        B0 = self.getBAS(drone_position, obstacle_position, self.delta)
        x0 = np.array([[drone_position[0]], [drone_position[1]],
                       [drone_velocity[0]], [drone_velocity[1]], [B0]])

        # Initial Controls:
        u_k = None
        if self.u_prev is None:
            u_k = np.zeros([self.nu, Horizon])
        else:
            u_k = np.hstack([self.u_prev[:, 1:], np.zeros([self.nu, 1])])
        du_k = np.zeros([self.nu, Horizon])

        # Initial Trajectory:
        x_traj = np.zeros([self.nx, Horizon])

        # Store Positions and Velocities of Subject and Obstacle:
        prediction_history = 10
        history_length = np.shape(self.subject_positions)[1]
        if history_length >= prediction_history:
            self.subject_positions = np.delete(self.subject_positions, 0, 1)
            self.subject_velocities = np.delete(self.subject_velocities, 0, 1)
            self.obstacle_positions = np.delete(self.obstacle_positions, 0, 1)
            self.obstacle_velocities = np.delete(self.obstacle_velocities, 0, 1)
        self.subject_positions = np.hstack([self.subject_positions, np.array([subject_position]).T])
        self.subject_velocities = np.hstack([self.subject_velocities, np.array([subject_velocity]).T])
        self.obstacle_positions = np.hstack([self.obstacle_positions, np.array([obstacle_position]).T])
        self.obstacle_velocities = np.hstack([self.obstacle_velocities, np.array([obstacle_velocity]).T])

        # Choose Prediction Strategy:
        if history_length < 3:
            obstacle_traj = self.predictObstacleTrajectoryStraight(obstacle_position, obstacle_velocity, Horizon, dt)
            target_traj = self.calcTargetTrajInFrontStraight(subject_position, subject_velocity, obstacle_traj, self.range, Horizon, dt)
        else:
            obstacle_traj = self.predictObstacleTrajectoryCurved(self.obstacle_positions, self.obstacle_velocities, Horizon, dt)
            target_traj = self.calcTargetTrajInFrontCurved(self.subject_positions, self.subject_velocities, obstacle_traj, self.range, Horizon, dt)
        # h = -delta **2 + (px-ox)**2 + (py-oy)**2
        # B = 1/h

        A_ = np.hstack([dfx, np.zeros((4, 1))])
        dBdu = np.array([0, 0])

        # Learning Rate:
        gamma = 0.5
        L = np.zeros([Horizon])
        L_x = np.zeros([self.nx, Horizon])
        L_xx = np.zeros([self.nx, self.nx, Horizon])
        L_u = np.zeros([self.nu, Horizon])
        L_uu = np.zeros([self.nu, self.nu, Horizon])
        L_ux = np.zeros([self.nu, self.nx, Horizon])
        phi = np.zeros([self.nx, self.nx, Horizon])
        B = np.zeros([self.nx, self.nu, Horizon])
        Vxx = np.zeros([self.nx, self.nx, Horizon])
        Vx = np.zeros([self.nx, Horizon])
        V = np.zeros([Horizon])
        l_k = np.zeros([self.nu, Horizon])
        L_k = np.zeros([self.nu, self.nx, Horizon])
        for k in range(num_iter):
            for j in range(Horizon):
                l0, l_x, l_xx, l_u, l_uu, l_ux = self.fnCost(x_traj[:, [j]] - target_traj[:, [j]], u_k[:, [j]], j, Q_f, R, dt)
                L[j] = (dt * l0)
                L_x[:, [j]] = (dt * l_x)
                L_xx[:, :, j] = (dt * l_xx)
                L_u[:, [j]] = (dt * l_u)
                L_uu[:, :, j] = (dt * l_uu)
                L_ux[:, :, j] = (dt * l_ux)
                dBdx = self.getAbar(x_traj[0, j], x_traj[1, j], obstacle_traj[0, j], obstacle_traj[1, j], self.delta)
                Abar = np.vstack([A_, dBdx])
                Bbar = np.vstack([dfu, dBdu])
                phi[:, :, j] = np.eye(self.nx) + Abar * dt
                B[:, :, j] = Bbar * dt
            Vxx[:, :, -1] = Q_f
            Vx[:, [-1]] = Q_f @ (x_traj[:, [-1]] - target_traj[:, [-1]])
            V[-1] = 0.5 * (x_traj[:, [-1]] - target_traj[:, [-1]]).T @ Q_f @ (x_traj[:, [-1]] - target_traj[:, [-1]])

            for j in range(Horizon - 2, -1, -1):
                Q_o = L[j] + V[j + 1]
                Q_u = L_u[:, [j]] + B[:, :, j].T @ Vx[:, [j + 1]]
                Q_x = L_x[:, [j]] + phi[:, :, j].T @ Vx[:, [j + 1]]
                Q_xx = L_xx[:, :, j] + phi[:, :, j].T @ Vxx[:, :, j + 1] @ phi[:, :, j]
                Q_uu = L_uu[:, :, j] + B[:, :, j].T @ Vxx[:, :, j + 1] @ B[:, :, j]
                Q_ux = L_ux[:, :, j] + B[:, :, j].T @ Vxx[:, :, j + 1] @ phi[:, :, j]
                Q_xu = Q_ux.T

                l_k[:, [j]] = -np.linalg.inv(Q_uu) @ Q_u
                L_k[:, :, j] = -np.linalg.inv(Q_uu) @ Q_ux

                Vxx[:, :, j] = Q_xx - Q_xu @ np.linalg.inv(Q_uu) @ Q_ux
                Vx[:, [j]] = Q_x - Q_xu @ np.linalg.inv(Q_uu) @ Q_u
                V[j] = Q_o - 0.5 * Q_u.T @ np.linalg.inv(Q_uu) @ Q_u

            dx = np.zeros([self.nx, 1])
            u_new = np.zeros([self.nu, Horizon])
            for i in range(Horizon):
                du = l_k[:, [i]] + L_k[:, :, i] @ dx
                dx = phi[:, :, i] @ dx + B[:, :, i] @ du
                u_new[:, [i]] = u_k[:, [i]] + gamma * du
            u_k = u_new
            x_traj = self.fnsimulate(x0, u_new, Horizon, dt, obstacle_traj)
            # cost = fnCostComputation(x_traj, u_k, target_traj, dt, Horizon, Q_f, R)
            # print(k, cost)
        self.u_prev = u_k
        return u_k[:, 0]


# Run one scenario for debugging
# scenario = FigureEightCosineObstacle(
#     loop_time=25,
#     loop_radius=5.0,
#     obstacle_period=28
# )
# result = run_scenario_using_planning_policy(scenario, MyPlanningPolicy)
# plot_animation(scenario, result)

scenarios = [
  FigureEightCosineObstacle(loop_time=25, loop_radius=5.0, obstacle_period=28),
  FigureEightCosineObstacle(loop_time=24, loop_radius=4.9, obstacle_period=21),
  FigureEightCosineObstacle(loop_time=28, loop_radius=5.2, obstacle_period=15),
  FigureEightCosineObstacle(loop_time=30, loop_radius=4.4, obstacle_period=15),
  FigureEightCosineObstacle(loop_time=30, loop_radius=6.0, obstacle_period=23)
]

h = 41
Q = np.diag([125, 125, 50, 50, 23000])
R = 41.1 * np.eye(2)
results = [run_scenario_using_planning_policy(s, MyPlanningPolicy, h, Q, R) for s in scenarios]
mean_error = np.mean([r['error'] for r in results])

for i in range(len(scenarios)):
    plot_animation(scenarios[i], results[i])
