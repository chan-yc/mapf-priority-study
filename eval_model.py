import os
import json

import numpy as np
import torch
import wandb

from alg_parameters import *
from episodic_buffer import EpisodicBuffer
from mapf_gym import MAPFEnv
from model import Model
from util import reset_env, make_gif, set_global_seeds


# TODO: use `argparse` to pass the parameters

NUM_TIMES = 100
# CASE = [[8, 10, 0], [8, 10, 0.15], [8, 10, 0.3], [16, 20, 0.0], [16, 20, 0.15], [16, 20, 0.3], [32, 30, 0.0],
#         [32, 30, 0.15], [32, 30, 0.3], [64, 40, 0.0], [64, 40, 0.15], [64, 40, 0.3], [128, 40, 0.0],
#         [128, 40, 0.15], [128, 40, 0.3]]
CASE = [[8, 10, 0], [8, 10, 0.15], [8, 10, 0.3]]
set_global_seeds(SetupParameters.SEED)


def one_step(env0, actions, model0, pre_value, input_state, ps, episode_perf,
             message, episodic_buffer0):
    obs, vector, reward, done, _, on_goal, _, _, _, _, _, max_on_goal, \
        num_collide, _, modify_actions \
        = env0.joint_step(actions, episode_perf['episode_len'], model0, pre_value, input_state,
                          ps, no_reward=False, message=message, episodic_buffer=episodic_buffer0)

    vector[:, :, -1] = modify_actions
    episode_perf['episode_len'] += 1
    episode_perf['collide'] += num_collide
    return reward, obs, vector, done, episode_perf, max_on_goal, on_goal


def evaluate(eval_env, model0, device, episodic_buffer0, num_agent, save_gif0):
    """Evaluate Model."""
    episode_frames = []

    # Reset environment
    done, _, obs, vector, _ = reset_env(eval_env, num_agent)
    message = Model.init_message(num_agent, torch.device('cpu'))
    hidden_state = Model.init_hidden_state(num_agent, torch.device('cpu'))

    # Reset buffer
    episodic_buffer0.reset(2e6, num_agent)
    new_xy = eval_env.get_positions()
    episodic_buffer0.batch_add(new_xy)

    episode_perf = {'episode_len': 0, 'max_goals': 0, 'collide': 0, 'success_rate': 0}

    # Run episode
    while not done:
        if save_gif0:
            episode_frames.append(eval_env._render())

        # Predict
        actions, hidden_state, v_all, ps, message \
            = model0.final_evaluate(obs, vector, hidden_state, message, num_agent)

        # Move
        rewards, obs, vector, done, episode_perf, max_on_goals, on_goal \
            = one_step(eval_env, actions, model0, v_all, hidden_state, ps,
                       episode_perf, message, episodic_buffer0)

        # Compute intrinsic rewards
        new_xy = eval_env.get_positions()
        processed_rewards, _, intrinsic_reward, min_dist \
            = episodic_buffer0.if_reward(new_xy, rewards, done, on_goal)

        vector[:, :, 3] = rewards
        vector[:, :, 4] = intrinsic_reward
        vector[:, :, 5] = min_dist

    # Compute episode performance
    if episode_perf['episode_len'] < EnvParameters.EPISODE_LEN - 1:
        episode_perf['success_rate'] = 1
    episode_perf['max_goals'] = max_on_goals
    episode_perf['collide'] = (episode_perf['collide'] / num_agent
                               / (episode_perf['episode_len'] + 1))
    # Save GIF
    if save_gif0:
        if not os.path.exists(RecordingParameters.GIFS_PATH):
            os.makedirs(RecordingParameters.GIFS_PATH)
        episode_frames.append(eval_env._render())
        images = np.array(episode_frames)
        make_gif(images, f'{RecordingParameters.GIFS_PATH}/evaluation.gif')

    return episode_perf


def main():
    # Get the trained model
    model_path = './final'
    path_checkpoint = model_path + "/net_checkpoint.pkl"
    net_dict = torch.load(path_checkpoint, map_location=torch.device('cpu'))
    model = Model(0, torch.device('cpu'))
    model.network.load_state_dict(net_dict['model'])

    # Recording
    wandb_id = wandb.util.generate_id()
    wandb.init(project='MAPF_evaluation',
               name='evaluation_global_SCRIMP',
               # entity=RecordingParameters.ENTITY,
               notes=f'Training state: {json.dumps(net_dict["training_state"])}',
               config=net_dict['all_configs'],
               id=wandb_id,
               resume='allow')
    print(f'Launched wandb (ID: {wandb_id})')

    # Start evaluation for each experiment case
    print('Start evaluation.')
    print('-' * 70)
    for case_params in CASE:

        save_gif = True  # Save one GIF for each case

        num_agent, world_size, obstacle_prob = case_params
        env = MAPFEnv(num_agent, world_size, obstacle_prob, mode='eval')
        episodic_buffer = EpisodicBuffer(total_step=2e6, num_agent=num_agent)

        all_perf = {'episode_len': [], 'max_goals': [],
                    'collide': [], 'success_rate': []}

        print(f'Agent: {case_params[0]}  World: {case_params[1]}  Obstacle: {case_params[2]}')

        # Evaluation loop
        for j in range(NUM_TIMES):
            # Evaluation
            episode_perf = evaluate(env, model, torch.device('cpu'),
                                    episodic_buffer, num_agent, save_gif)
            # Record metrics of one episode
            for metric in episode_perf.keys():
                if metric == 'episode_len' and episode_perf['success_rate'] == 1:
                    # Record only successful episodes for episode_len
                    all_perf[metric].append(episode_perf[metric])
                else:
                    all_perf[metric].append(episode_perf[metric])

            # Record GIF only for the first episode
            save_gif = False
            if (j+1) % 20 == 0:
                print(f'Finished {j+1} episodes.')

        # Compute log messages of mean metrics
        perf_log = {}
        for i in all_perf.keys():  # for all episodes
            mean = round(np.nanmean(all_perf[i]), 2)
            if i == 'success_rate':
                perf_log[i] = f'{mean}'
            else:
                stdv = round(np.nanstd(all_perf[i]), 2)
                perf_log[i] = f'{mean} ({stdv})'

        # Log results
        mean_log = f"EL: {perf_log['episode_len']: <10}  " \
                   f"MR: {perf_log['max_goals']: <10}  " \
                   f"CO: {perf_log['collide']: <10}  " \
                   f"SR: {perf_log['success_rate']: <10}"

        print(mean_log)
        print('-' * 70)

    print('Completed evaluation.')
    wandb.finish()


if __name__ == "__main__":
    main()
