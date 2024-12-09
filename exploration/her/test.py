import numpy as np
from spinup.utils.test_policy import load_policy_and_env, run_policy

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/her/her_td3_custom')
    parser.add_argument('--theta', type=float, default=90)
    parser.add_argument('--itr', type=int, default=-1)
    args = parser.parse_args()

    itr = 'last' if args.itr == -1 else args.itr
    env, get_action = load_policy_and_env(args.data, itr=itr)
    goal = np.array([np.cos(np.deg2rad(args.theta)), np.sin(np.deg2rad(args.theta))]) if args.theta is not None else None
    run_policy(env, get_action, goal=goal)