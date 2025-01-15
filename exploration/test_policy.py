import numpy as np
import torch
from core.test_policy import get_parser as get_parser_core
from core.test_policy import setup_env_and_model

if __name__ == '__main__':
    parser = get_parser_core()
    args = parser.parse_args()
    env, model, device = setup_env_and_model(args)

    # Run policy in environment for the set number of episodes
    for ep in range(args.num_episodes):
        (obs, _), done = env.reset(), False
        ep_ret, ep_len = 0, 0
        if hasattr(model, 'reset_hidden_states'):
            model.reset_hidden_states(device)
        while not done:
            if args.record == True:
                env.render()
            act = model.act(torch.as_tensor(obs, dtype=torch.float32).to(device),
                            deterministic=args.deterministic)
            if act.dtype == np.int64: act = int(act) # Deals with a bug in Vizdoom
            obs, rew, terminated, truncated, _ = env.step(act)
            ep_ret, ep_len = ep_ret + rew, ep_len + 1
            done = terminated or truncated
        print(f'Episode {ep}: EpRet = {ep_ret} \t EpLen = {ep_len}')

    env.close()