from spinup.utils.test_policy import load_policy_and_env, run_policy

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/td3/td3_custom')
    args = parser.parse_args()

    env, get_action = load_policy_and_env(args.data)
    run_policy(env, get_action)