import gym
import spinup.algos.tf1.trpo.core as core
from spinup.utils.test_policy import load_policy_and_env, run_policy
from models import MLPActorCritic


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/trpo/trpo_custom_discrete')
    parser.add_argument('--itr', type=int, default=-1)
    args = parser.parse_args()

    itr = 'last' if args.itr == -1 else args.itr
    env, get_action = load_policy_and_env(args.data, itr=itr)
    run_policy(env, get_action)