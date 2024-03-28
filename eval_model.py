import os
import json
import argparse

import numpy as np
import torch
import wandb

from alg_parameters import *
from episodic_buffer import EpisodicBuffer
from mapf_gym import MAPFEnv
from model import Model
from util import reset_env, make_gif, set_global_seeds, get_torch_device


NUM_TIMES = 100
# CASE = [[8, 10, 0], [8, 10, 0.15], [8, 10, 0.3], [16, 20, 0.0], [16, 20, 0.15], [16, 20, 0.3], [32, 30, 0.0],
#         [32, 30, 0.15], [32, 30, 0.3], [64, 40, 0.0], [64, 40, 0.15], [64, 40, 0.3], [128, 40, 0.0],
#         [128, 40, 0.15], [128, 40, 0.3]]
CASE = [[8, 10, 0], [8, 10, 0.15], [8, 10, 0.3]]
set_global_seeds(SetupParameters.SEED)


def one_step(env0, actions, model0, pre_value, input_state, ps, episode_perf,
             message, episodic_buffer0):
    """Run one step of the environment"""
    obs, vector, reward, done, _, on_goal, _, _, _, _, _, max_on_goal, \
        num_collide, _, modify_actions \
        = env0.joint_step(actions, episode_perf['episode_len'], model0, pre_value, input_state,
                          ps, no_reward=False, message=message, episodic_buffer=episodic_buffer0)

    vector[:, :, -1] = modify_actions
    episode_perf['episode_len'] += 1
    episode_perf['collide'] += num_collide
    return reward, obs, vector, done, episode_perf, max_on_goal, on_goal


def eval_episode(eval_env, model0, device, episodic_buffer0, num_agent, save_gif0):
    """Evaluate one episode of the trained model"""
    episode_frames = []

    # Reset environment
    done, _, obs, vector, _ = reset_env(eval_env, num_agent)
    message = Model.init_message(num_agent, device)
    hidden_state = Model.init_hidden_state(num_agent, device)

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


def eval_model(net_checkpoint=None, device=torch.device('cpu')):
    """Evaluate the trained model"""

    # Get the trained model
    net_checkpoint = net_checkpoint or os.path.join('final', 'net_checkpoint.pkl')
    if not os.path.exists(net_checkpoint):
        raise FileNotFoundError(f"'{net_checkpoint}' does not exist!")
    net_dict = torch.load(net_checkpoint, map_location=device)
    model = Model(0, device)
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
    print(f'Launched wandb. (ID: {wandb_id})\n')

    # Start evaluation for each experiment case
    print('Start evaluation.\n')
    print('-' * 70)
    for n, case_params in enumerate(CASE):
        print(f'[Case: {n+1}/{len(CASE)}]')

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
            episode_perf = eval_episode(env, model, device,  episodic_buffer,
                                        num_agent, save_gif)
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
                print(f'Finished {j+1}/{NUM_TIMES} episodes.')
        print(f'Finished all {NUM_TIMES} episodes.')

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

    # Create the parser
    parser = argparse.ArgumentParser(description='Evaluate a trained model.')
    # Model path argument
    parser.add_argument('model_path', type=str, nargs='?', default='final',
                        help='directory of the trained model, defaults to \'./final\'')
    # GPU argument
    parser.add_argument('-g', '--gpu', action='store_true', help='use GPU if specified')
    # Parse the arguments
    args = parser.parse_args()

    # Check if the provided path is a directory
    if not os.path.isdir(args.model_path):
        raise ValueError('The provided model path is not a directory!')

    net_checkpoint = os.path.join(args.model_path, 'net_checkpoint.pkl')
    device = get_torch_device(use_gpu=args.gpu)

    eval_model(net_checkpoint, device)
