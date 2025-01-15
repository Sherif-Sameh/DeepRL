import os
from gymnasium.envs.registration import register

''' Checks the given environment name and returns true if it starts with Vizdoom'''
def is_vizdoom_env(env_name: str):
    
    return 'Vizdoom' in env_name
