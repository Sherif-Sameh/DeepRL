import gym
import spinup.algos.pytorch.vpg.core as core
from spinup.utils.test_policy import load_policy_and_env, run_policy
from models import MLPActorCritic

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/vpg/vpg_spinup_discrete_1')
    args = parser.parse_args()

    env, get_action = load_policy_and_env(args.data)
    run_policy(env, get_action)