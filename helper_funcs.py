import numpy as np
import plotly.express as px
import time
import typing as T

# def fnsimulate(
#         x0: np.ndarray,
#         u_new: np.ndarray,
#         Horizon: float,
#         dt: float
# ) -> np.ndarray:
#     x_traj = np.zeros([4,Horizon])
#     x_traj[:,[0]] = x0
#     for k in range(1,Horizon):
#         x_traj[0, [k]] = x_traj[0,[k-1]] + x_traj[2,[k-1]]*dt
#         x_traj[1, [k]] = x_traj[1, [k - 1]] + x_traj[3, [k - 1]] * dt
#         x_traj[2, [k]] = x_traj[2, [k - 1]] + u_new[0, [k-1]] * dt
#         x_traj[3, [k]] = x_traj[3, [k - 1]] + u_new[1, [k - 1]] * dt
#     return x_traj

# def fnCostComputation(
#         x_traj: np.ndarray,
#         u_new: np.ndarray,
#         target_traj: np.ndarray,
#         dt: float,
#         Horizon: float,
#         Q_f: np.ndarray,
#         R: np.ndarray
# ) -> float:
#     cost = 0
#     for j in range(Horizon):
#         cost += 0.5*u_new[:,[j]].T @ R @ u_new[:,[j]] * dt + 0.5*(x_traj[:,[j]]-target_traj[:,[j]]).T @ Q_f @ (x_traj[:,[j]]-target_traj[:,[j]]) * dt
#     cost += (x_traj[:,[-1]]-target_traj[:,[-1]]).T @ Q_f @ (x_traj[:,[-1]]-target_traj[:,[-1]])
#     return cost

def simulate_drone_dynamics(
    dt : float,
    position : np.ndarray,
    velocity : np.ndarray,
    accel_command : np.ndarray,
    max_accel : float,
    max_speed : float
  ) -> np.ndarray:
    """
    Simulate the drone's dynamics forward given a control input.
    """
    def saturated(vec : np.ndarray, max_mag : float) -> np.ndarray:
        mag = np.sqrt(np.sum(vec**2) + 1e-6)
        return vec * np.minimum(mag, max_mag) / mag

    # Basic forward euler integration
    accel_next = saturated(accel_command, max_accel)
    velocity_next = saturated(velocity + dt * accel_next, max_speed)
    position_next = position + dt * velocity_next

    return position_next, velocity_next, accel_next

def run_simulation(
    times : np.ndarray,
    subject_positions : np.ndarray,
    obstacle_positions : np.ndarray,
    planning_policy : T.Any,
    params : T.Dict[str, T.Any],
    h : float,
    Q: np.ndarray,
    R: np.ndarray
  ) -> T.Tuple[np.ndarray]:
    """
    Compute the drone's trajectory by stepping along time, computing a commmand,
    and simulating the drone's dynamics.
    """
    policy = planning_policy(params=params)
    drone_positions = [params['drone_initial_position']]
    drone_velocities = [np.array([0, 0])]
    drone_accels = [np.array([0, 0])]
    subject_velocities = [np.array([0, 0])]

    for i in range(len(times) - 1):

        # Get current state
        t, t_next = times[i], times[i + 1]
        # print(t)
        dt = t_next - t
        drone_position, drone_velocity = drone_positions[i], drone_velocities[i]
        subject_position = subject_positions[i]
        subject_velocity = (subject_positions[i + 1] - subject_position) / dt
        obstacle_position = obstacle_positions[i]
        obstacle_velocity = (obstacle_positions[i + 1] - obstacle_positions[i]) / dt

        # Compute the command
        accel_command = policy.compute_accel(
            t=t,
            drone_position=drone_position,
            drone_velocity=drone_velocity,
            subject_position=subject_position,
            subject_velocity=subject_velocity,
            obstacle_position=obstacle_position,
            obstacle_velocity=obstacle_velocity,
            Horizon=h,
            Q_f=Q,
            R=R
        )

        # Simulate the next drone position
        drone_position_next, drone_velocity_next, drone_accel_next =\
            simulate_drone_dynamics(
                dt=t_next - t,
                position=drone_position,
                velocity=drone_velocity,
                accel_command=accel_command,
                max_accel=params['drone_max_accel'],
                max_speed=params['drone_max_speed']
            )
        drone_positions.append(drone_position_next)
        drone_velocities.append(drone_velocity_next)
        drone_accels.append(drone_accel_next)
        subject_velocities.append(subject_velocity)

    return np.vstack(drone_positions), np.vstack(drone_velocities),\
        np.vstack(drone_accels), np.vstack(subject_velocities)

def mod_2_pi(x : np.ndarray) -> np.ndarray:
    """
    Convert an angle in radians into range [-pi, pi]
    """
    return np.mod(x + np.pi, 2 * np.pi) - np.pi

def compute_error_for_scenario(
    subject_positions : np.ndarray,
    subject_velocities : np.ndarray,
    drone_positions : np.ndarray,
    drone_velocities : np.ndarray,
    drone_accels : np.ndarray,
    obstacle_positions : np.ndarray,
    params : T.Dict[str, T.Any],
    verbose : bool = True
  ):
    """
    Score the planning policy's performance with a multi-objective metric,
    where lower is better.

    Current cost terms:
      #1) Drone is at a desired range from the subject
      #2) Drone is at a desired angle relative to the subject's velocity vector
      #3) Drone is penalized in a soft way for being inside the obstacle
    """
    error = 0.0

    # Error term for the range of the drone from the subject
    range_cost = params['desired_range_to_subject_cost']
    diff = drone_positions - subject_positions
    actual_range = np.linalg.norm(diff, axis=1)
    range_residual = params['desired_range_to_subject'] - actual_range
    range_error = 0.5 * params['dt'] * range_cost * range_residual.dot(range_residual)
    error += range_error
    if verbose:
        print(f'Subject range error: {range_error:0.1f}')

    # Error term for the angle to the subject
    angle_cost = params['desired_angle_to_subject_velocity_cost']
    actual_angle = np.arctan2(diff[:, 1], diff[:, 0])
    subject_vel_angle = np.arctan2(subject_velocities[:, 1], subject_velocities[:, 0] + 1e-6)
    angle_residual = mod_2_pi(params['desired_angle_to_subject_velocity'] +\
                              subject_vel_angle - actual_angle)
    angle_error = 0.5 * params['dt'] * angle_cost * angle_residual.dot(angle_residual)
    error += angle_error
    if verbose:
        print(f'Subject angle error: {angle_error:0.1f}')

    # Error term for going inside a moving obstacle radius
    obstacle_cost = params['obstacle_cost']
    range_to_obstacle = np.linalg.norm(drone_positions - obstacle_positions, axis=1)
    infringement = np.maximum(0, params['obstacle_radius'] - range_to_obstacle)
    obstacle_error = 0.5 * params['dt'] * obstacle_cost * infringement.dot(infringement)
    error += obstacle_error
    if verbose:
        print(f'Obstacle error: {obstacle_error:0.1f}')

    print(f'Total error: {error:0.1f}')
    return error


class Scenario:
    """
    Base class for describing the test scenario.
    """

    def __init__(self):
        pass

    def params(self):
        """
        Return static parameters for the scenario
        """
        return dict(
            dt=0.025,  # [s]
            total_time=30.0,  # [s]

            drone_initial_position=np.array([-3.0, 0.0]),  # [m]
            drone_max_speed=10.0,  # [m/s]
            drone_max_accel=2.5,  # [m/s^2]

            obstacle_radius=2.0,  # [m]

            desired_range_to_subject_cost=10.0,
            desired_range_to_subject=3.0,  # [m]

            desired_angle_to_subject_velocity_cost=10.0,
            desired_angle_to_subject_velocity=0.0 * np.pi,  # [rad]

            obstacle_cost=500.0
        )

    def subject_position(self, t: float) -> np.ndarray:
        """
        Return the subject's (x, y) position given a time.
        """
        return np.array([0.0, 0.0])

    def obstacle_position(self, t: float) -> np.ndarray:
        """
        Return the obstacle's (x, y) position given a time.
        """
        return np.array([10.0, 10.0])


class FigureEightCosineObstacle(Scenario):
    """
    The subject moves along a figure eight curve:
        https://mathworld.wolfram.com/EightCurve.html

    Obstacle moves along a cosine wave.
    """

    def __init__(self,
                 loop_time: float = 25.0,  # [s]
                 loop_radius: float = 5.0,  # [m]
                 obstacle_period=4.14):
        self.loop_time = loop_time
        self.loop_radius = loop_radius
        self.obstacle_period = obstacle_period

    def subject_position(self, t: float) -> np.ndarray:
        t_angle = t / self.loop_time * (2 * np.pi)
        x = self.loop_radius * np.sin(t_angle)
        y = x * np.cos(t_angle)
        return np.array([x, y]) + np.array([0, 1])

    def obstacle_position(self, t: float) -> np.ndarray:
        return np.array([11, -1]) + (t / 25) * np.array([-21.1, 0]) + \
               np.cos(t * 2 * np.pi / self.obstacle_period) * np.array([0, 5])


def plot_animation(scenario: Scenario, result: T.Dict[str, T.Any],
                   range_dims: T.Tuple[float, float] = (10, 6),
                   drawing_dt: float = 0.5) -> None:
    """
    Draw the experiment results as an interactive widget.
    """
    title = f'{scenario.__class__.__name__} (error = {result["error"]:0.1f})'
    times = result['times']
    subject_positions = result['subject_positions']
    drone_positions = result['drone_positions']
    obstacle_positions = result['obstacle_positions']
    obstacle_radius = scenario.params()['obstacle_radius']

    # Downsample for drawing sanity, also grabbing last frame
    dt = times[1] - times[0]
    step = int(drawing_dt / dt)
    indices = np.array(list(range(len(times))[::step]) + [len(times) - 1])

    # Assemble the data frame by stacking subject, drone, obstacle data
    data = dict()
    data['t'] = np.hstack([times[indices], times[indices], times[indices]])
    data['x'] = np.hstack([subject_positions[indices, 0],
                           drone_positions[indices, 0],
                           obstacle_positions[indices, 0]])
    data['y'] = np.hstack([subject_positions[indices, 1],
                           drone_positions[indices, 1],
                           obstacle_positions[indices, 1]])
    data['type'] = ['subject'] * len(indices) + \
                   ['drone'] * len(indices) + \
                   ['obstacle'] * len(indices)
    data['size'] = [5] * len(indices) + [5] * len(indices) + \
                   [53 * obstacle_radius] * len(indices)

    # Make the animated trace
    fig = px.scatter(
        data,
        x='x',
        y='y',
        animation_frame='t',
        animation_group='type',
        color='type',
        category_orders={'type': ['obstacle', 'subject', 'drone']},
        color_discrete_sequence=('#FF5555', '#CCCC00', '#5555FF'),
        size='size',
        size_max=data['size'][-1],
        hover_name='type',
        template='plotly_dark',
        range_x=(-range_dims[0], range_dims[0]),
        range_y=(-range_dims[1], range_dims[1]),
        height=700,
        title=title
    )

    # Make equal one meter grid
    fig.update_xaxes(
        dtick=1.0,
        showline=False
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        showline=False,
        dtick=1.0
    )

    # Draw full curve of the subject's path
    subject_line = px.line(
        x=subject_positions[:, 0],
        y=subject_positions[:, 1]
    ).data[0]
    subject_line.line['color'] = '#FFFF55'
    subject_line.line['width'] = 1
    fig.add_trace(subject_line)

    # Draw full curve of the drone's path
    drone_line = px.line(
        x=drone_positions[:, 0],
        y=drone_positions[:, 1]
    ).data[0]
    drone_line.line['color'] = '#AAAAFF'
    drone_line.line['width'] = 1
    fig.add_trace(drone_line)

    # Draw full curve of the obstacle's path
    drone_line = px.line(
        x=obstacle_positions[:, 0],
        y=obstacle_positions[:, 1]
    ).data[0]
    drone_line.line['color'] = '#FFAAAA'
    drone_line.line['width'] = 1
    fig.add_trace(drone_line)

    fig.show()

def run_scenario_using_planning_policy(scenario : Scenario,
                                       planning_policy : T.Any,
                                       h: float,
                                       Q: np.ndarray,
                                       R: np.ndarray) -> T.Dict:
    """
    Simulate the given planning policy and score its performance.
    """
    params = scenario.params()
    times = np.arange(0, params['total_time'], params['dt'])

    # Compute the subject and obstacle trajectory
    subject_positions = np.array([scenario.subject_position(t) for t in times])
    obstacle_positions = np.array([scenario.obstacle_position(t) for t in times])

    # Compute the drone trajectory
    start_time = time.time()
    drone_positions, drone_velocities, drone_accels, subject_velocities =\
        run_simulation(
            times=times,
            subject_positions=subject_positions,
            obstacle_positions=obstacle_positions,
            planning_policy=planning_policy,
            params=params,
            h = h,
            Q = Q,
            R = R
        )
    print(f'\nScenario took {time.time() - start_time:6.2f} s.')

    # See how well the policy did
    error = compute_error_for_scenario(
        subject_positions=subject_positions,
        subject_velocities=subject_velocities,
        drone_positions=drone_positions,
        drone_velocities=drone_velocities,
        drone_accels=drone_accels,
        obstacle_positions=obstacle_positions,
        params=params,
        verbose=True
    )

    return dict(
        times=times,
        subject_positions=subject_positions,
        drone_positions=drone_positions,
        obstacle_positions=obstacle_positions,
        error=error
    )