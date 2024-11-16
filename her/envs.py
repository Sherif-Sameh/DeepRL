import numpy as np

### Define a sparse reward function for the pendulum CC environment
def sparse_rew_pendulum(obs, act, goal_pos, eps=0.1):
    # Ensure that inputs are at least 2D
    obs, act, goal_pos = np.atleast_2d(obs, act, goal_pos)
    
    # Calculate abs error in theta [0, pi]
    goal_theta = np.arctan2(goal_pos[:, 1], goal_pos[:, 0])
    obs_theta = np.arctan2(obs[:, 1], obs[:, 0])
    error = wrap_abs_angle(goal_theta - obs_theta)

    # Calculate rewards
    dense_rew = -(0.1 * obs[:, 2]**2 + 0.001 * act[:, 0]**2)
    sparse_rew = np.where(error < eps, 0.0, -1.0)

    return sparse_rew + dense_rew


### Define a function to sample goal states for the pendulum CC environment
def sample_goal_pendulum():
    theta = np.random.choice(np.linspace(-np.pi, np.pi, 32))    
    return np.array([np.cos(theta), np.sin(theta)])


def wrap_abs_angle(angles):
    angles = np.abs(angles)
    angles = angles % (2 * np.pi)
    wrapped_angles = np.where(angles > np.pi, 2 * np.pi - angles, angles)
    
    return wrapped_angles