import gymnasium as gym

''' Checks the given environment name and returns true if it starts with Vizdoom'''
def is_vizdoom_env(env_name: str):
    return 'Vizdoom' in env_name

''' Checks whether the given environment is a valid multi-goal environment '''
def is_multigoal_env(env_name):
    env = gym.make(env_name)

    # The following methods must be defined for a multi-goal environment
    assert hasattr(env.unwrapped, 'compute_reward')
    assert hasattr(env.unwrapped, 'compute_truncated')
    assert hasattr(env.unwrapped, 'compute_terminated')

    # The following results must hold for a multi-goal environment
    env.reset()
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    assert rew == env.unwrapped.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
    assert truncated == env.unwrapped.compute_truncated(obs["achieved_goal"], obs["desired_goal"], info)
    assert terminated == env.unwrapped.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info)
    env.close()
    del env

    return True