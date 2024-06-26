import os
import random

import imageio
import numpy as np
import torch
import wandb

from alg_parameters import *


def set_global_seeds(i):
    """set seed for fair comparison"""
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)
    torch.backends.cudnn.deterministic = True


def write_to_tensorboard(global_summary, step, performance_dict=None, mb_loss=None, imitation_loss=None, evaluate=True,
                         greedy=True):
    """record performance using tensorboard"""
    if imitation_loss is not None:
        global_summary.add_scalar(tag='Loss/Imitation_loss', scalar_value=imitation_loss[0], global_step=step)
        global_summary.add_scalar(tag='Grad/Imitation_grad', scalar_value=imitation_loss[1], global_step=step)

        global_summary.flush()
        return
    if evaluate:
        if greedy:
            global_summary.add_scalar(tag='Perf_greedy_eval/Reward', scalar_value=performance_dict['per_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/In_Reward', scalar_value=performance_dict['per_in_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Ex_Reward', scalar_value=performance_dict['per_ex_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Valid_rate', scalar_value=performance_dict['per_valid_rate'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Episode_length', scalar_value=performance_dict['per_episode_len'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Num_block', scalar_value=performance_dict['per_block'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Num_leave_goal',scalar_value=performance_dict['per_leave_goal'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Final_goals', scalar_value=performance_dict['per_final_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Half_goals', scalar_value=performance_dict['per_half_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Block_accuracy', scalar_value=performance_dict['per_block_acc'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Max_goals', scalar_value=performance_dict['per_max_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Num_collide', scalar_value=performance_dict['per_num_collide'], global_step=step)

        else:
            global_summary.add_scalar(tag='Perf_random_eval/Reward', scalar_value=performance_dict['per_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/In_Reward', scalar_value=performance_dict['per_in_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Ex_Reward', scalar_value=performance_dict['per_ex_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Valid_rate',scalar_value=performance_dict['per_valid_rate'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Episode_length',scalar_value=performance_dict['per_episode_len'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Num_block', scalar_value=performance_dict['per_block'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Num_leave_goal', scalar_value=performance_dict['per_leave_goal'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Final_goals', scalar_value=performance_dict['per_final_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Half_goals', scalar_value=performance_dict['per_half_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Block_accuracy', scalar_value=performance_dict['per_block_acc'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Max_goals', scalar_value=performance_dict['per_max_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Num_collide', scalar_value=performance_dict['per_num_collide'], global_step=step)

    else:
        loss_vals = np.nanmean(mb_loss, axis=0)
        global_summary.add_scalar(tag='Perf/Reward', scalar_value=performance_dict['per_r'], global_step=step)
        global_summary.add_scalar(tag='Perf/In_Reward', scalar_value=performance_dict['per_in_r'], global_step=step)
        global_summary.add_scalar(tag='Perf/Ex_Reward', scalar_value=performance_dict['per_ex_r'], global_step=step)
        global_summary.add_scalar(tag='Perf/Valid_rate', scalar_value=performance_dict['per_valid_rate'], global_step=step)
        global_summary.add_scalar(tag='Perf/Episode_length',scalar_value=performance_dict['per_episode_len'], global_step=step)
        global_summary.add_scalar(tag='Perf/Num_block', scalar_value=performance_dict['per_block'], global_step=step)
        global_summary.add_scalar(tag='Perf/Num_leave_goal', scalar_value=performance_dict['per_leave_goal'], global_step=step)
        global_summary.add_scalar(tag='Perf/Final_goals', scalar_value=performance_dict['per_final_goals'], global_step=step)
        global_summary.add_scalar(tag='Perf/Half_goals', scalar_value=performance_dict['per_half_goals'], global_step=step)
        global_summary.add_scalar(tag='Perf/Block_accuracy', scalar_value=performance_dict['per_block_acc'], global_step=step)
        global_summary.add_scalar(tag='Perf/Max_goals', scalar_value=performance_dict['per_max_goals'], global_step=step)
        global_summary.add_scalar(tag='Perf/Num_collide', scalar_value=performance_dict['per_num_collide'], global_step=step)
        global_summary.add_scalar(tag='Perf/Rewarded_rate', scalar_value=performance_dict['rewarded_rate'], global_step=step)

        for (val, name) in zip(loss_vals, RecordingParameters.LOSS_NAME):
            if name == 'grad_norm':
                global_summary.add_scalar(tag='Grad/' + name, scalar_value=val, global_step=step)
            else:
                global_summary.add_scalar(tag='Loss/' + name, scalar_value=val, global_step=step)

    global_summary.flush()


def write_to_wandb(step, performance_dict=None, mb_loss=None, imitation_loss=None,
                   evaluate=True, greedy=True):
    """record performance using wandb"""
    if imitation_loss is not None:
        wandb.log({'Loss/Imitation_loss': imitation_loss[0]}, step=step)
        wandb.log({'Grad/Imitation_grad': imitation_loss[1]}, step=step)
        return
    if evaluate:
        if greedy:
            wandb.log({'Perf_greedy_eval/Reward': performance_dict['per_r']}, step=step)
            wandb.log({'Perf_greedy_eval/In_Reward': performance_dict['per_in_r']}, step=step)
            wandb.log({'Perf_greedy_eval/Ex_Reward': performance_dict['per_ex_r']}, step=step)
            wandb.log({'Perf_greedy_eval/Valid_rate': performance_dict['per_valid_rate']}, step=step)
            wandb.log({'Perf_greedy_eval/Episode_length': performance_dict['per_episode_len']}, step=step)
            wandb.log({'Perf_greedy_eval/Num_block': performance_dict['per_block']}, step=step)
            wandb.log({'Perf_greedy_eval/Num_leave_goal': performance_dict['per_leave_goal']}, step=step)
            wandb.log({'Perf_greedy_eval/Final_goals': performance_dict['per_final_goals']}, step=step)
            wandb.log({'Perf_greedy_eval/Half_goals': performance_dict['per_half_goals']}, step=step)
            wandb.log({'Perf_greedy_eval/Block_accuracy': performance_dict['per_block_acc']}, step=step)
            wandb.log({'Perf_greedy_eval/Max_goals': performance_dict['per_max_goals']}, step=step)
            wandb.log({'Perf_greedy_eval/Num_collide': performance_dict['per_num_collide']}, step=step)

        else:
            wandb.log({'Perf_random_eval/Reward': performance_dict['per_r']}, step=step)
            wandb.log({'Perf_random_eval/In_Reward': performance_dict['per_in_r']}, step=step)
            wandb.log({'Perf_random_eval/Ex_Reward': performance_dict['per_ex_r']}, step=step)
            wandb.log({'Perf_random_eval/Valid_rate': performance_dict['per_valid_rate']}, step=step)
            wandb.log({'Perf_random_eval/Episode_length': performance_dict['per_episode_len']}, step=step)
            wandb.log({'Perf_random_eval/Num_block': performance_dict['per_block']}, step=step)
            wandb.log({'Perf_random_eval/Num_leave_goal': performance_dict['per_leave_goal']}, step=step)
            wandb.log({'Perf_random_eval/Final_goals': performance_dict['per_final_goals']}, step=step)
            wandb.log({'Perf_random_eval/Half_goals': performance_dict['per_half_goals']}, step=step)
            wandb.log({'Perf_random_eval/Block_accuracy': performance_dict['per_block_acc']}, step=step)
            wandb.log({'Perf_random_eval/Max_goals': performance_dict['per_max_goals']}, step=step)
            wandb.log({'Perf_random_eval/Num_collide': performance_dict['per_num_collide']}, step=step)

    else:
        loss_vals = np.nanmean(mb_loss, axis=0)
        wandb.log({'Perf/Reward': performance_dict['per_r']}, step=step)
        wandb.log({'Perf/In_Reward': performance_dict['per_in_r']}, step=step)
        wandb.log({'Perf/Ex_Reward': performance_dict['per_ex_r']}, step=step)
        wandb.log({'Perf/Valid_rate': performance_dict['per_valid_rate']}, step=step)
        wandb.log({'Perf/Episode_length': performance_dict['per_episode_len']}, step=step)
        wandb.log({'Perf/Num_block': performance_dict['per_block']}, step=step)
        wandb.log({'Perf/Num_leave_goal': performance_dict['per_leave_goal']}, step=step)
        wandb.log({'Perf/Final_goals': performance_dict['per_final_goals']}, step=step)
        wandb.log({'Perf/Half_goals': performance_dict['per_half_goals']}, step=step)
        wandb.log({'Perf/Block_accuracy': performance_dict['per_block_acc']}, step=step)
        wandb.log({'Perf/Max_goals': performance_dict['per_max_goals']}, step=step)
        wandb.log({'Perf/Num_collide': performance_dict['per_num_collide']},
                  step=step)
        wandb.log({'Perf/Rewarded_rate': performance_dict['rewarded_rate']},
                  step=step)

        for (val, name) in zip(loss_vals, RecordingParameters.LOSS_NAME):
            if name == 'grad_norm':
                wandb.log({'Grad/' + name: val}, step=step)
            else:
                wandb.log({'Loss/' + name: val}, step=step)


def make_gif(images, file_name):
    """record gif"""
    print('Saving GIF ...')
    imageio.mimwrite(file_name, images, subrectangles=True)
    print(f'Saved GIF to {file_name}\n')


def reset_env(env, num_agent):
    """reset environment"""
    done = env._reset(num_agent)
    prev_action = np.zeros(num_agent)
    valid_actions = []
    obs = np.zeros((1, num_agent, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE,
                    EnvParameters.FOV_SIZE), dtype=np.float32)
    vector = np.zeros((1, num_agent, NetParameters.VECTOR_LEN), dtype=np.float32)
    train_valid = np.zeros((num_agent, EnvParameters.N_ACTIONS), dtype=np.float32)

    for i in range(num_agent):
        valid_action = env.list_next_valid_actions(i + 1)
        s = env.observe(i + 1)
        obs[:, i, :, :, :] = s[0]
        vector[:, i, : 3] = s[1]
        vector[:, i, -1] = prev_action[i]
        valid_actions.append(valid_action)
        train_valid[i, valid_action] = 1
    return done, valid_actions, obs, vector, train_valid


def one_step(env, one_episode_perf, actions, pre_block, model, pre_value,
             input_state, ps, no_reward, message, episodic_buffer, num_agent):
    """run one step"""
    train_valid = np.zeros((num_agent, EnvParameters.N_ACTIONS), dtype=np.float32)
    obs, vector, rewards, done, next_valid_actions, on_goal, blockings, valid_actions, num_blockings, leave_goals, \
        num_on_goal, max_on_goal, num_collide, action_status, modify_actions \
        = env.joint_step(actions, one_episode_perf['num_step'], model, pre_value,
                         input_state, ps, no_reward, message, pre_block, episodic_buffer)

    one_episode_perf['block'] += num_blockings
    one_episode_perf['num_leave_goal'] += leave_goals
    one_episode_perf['num_collide'] += num_collide
    vector[:, :, -1] = modify_actions
    for i in range(num_agent):
        train_valid[i, next_valid_actions[i]] = 1
        if (pre_block[i] < 0.5) == blockings[:, i]:
            one_episode_perf['wrong_blocking'] += 1
    one_episode_perf['num_step'] += 1
    return rewards, next_valid_actions, obs, vector, train_valid, done, blockings, num_on_goal, one_episode_perf, \
        max_on_goal, action_status, modify_actions, on_goal


def update_perf(one_episode_perf, performance_dict, num_on_goals, max_on_goals, num_agent):
    """record batch performance"""
    performance_dict['per_ex_r'].append(one_episode_perf['ex_reward'])
    performance_dict['per_in_r'].append(one_episode_perf['in_reward'])
    performance_dict['per_r'].append(one_episode_perf['episode_reward'])
    performance_dict['per_valid_rate'].append(
        ((one_episode_perf['num_step'] * num_agent) - one_episode_perf['invalid']) / (
                one_episode_perf['num_step'] * num_agent))
    performance_dict['per_episode_len'].append(one_episode_perf['num_step'])
    performance_dict['per_block'].append(one_episode_perf['block'])
    performance_dict['per_leave_goal'].append(one_episode_perf['num_leave_goal'])
    performance_dict['per_num_collide'].append(one_episode_perf['num_collide'])
    performance_dict['per_final_goals'].append(num_on_goals)
    performance_dict['per_block_acc'].append(
        ((one_episode_perf['num_step'] * num_agent) - one_episode_perf['wrong_blocking']) / (
                one_episode_perf['num_step'] * num_agent))
    performance_dict['per_max_goals'].append(max_on_goals)
    performance_dict['rewarded_rate'].append(
        one_episode_perf['reward_count'] / (one_episode_perf['num_step'] * num_agent))
    return performance_dict


def get_torch_device(use_gpu=False):
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
    return torch.device('cpu')


def interval_has_elapsed(curr_step, last_step, interval):
    """Check if the interval has elapsed"""
    return curr_step - last_step >= interval


def ensure_directory(dir_path):
    """Ensure the directory exists"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_net(net_dir, model, curr_steps, curr_episodes, performance):
    ensure_directory(net_dir)
    state_log = f"Episodes: {curr_episodes: <8}  " \
                f"Steps: {curr_steps: <8}  " \
                f"Episode reward: {round(performance['per_r'], 2): <8}"
    print(state_log)
    net_path = os.path.join(net_dir, RecordingParameters.MODEL_SAVE)
    net_checkpoint = {"model": model.network.state_dict(),
                      "optimizer": model.net_optimizer.state_dict(),
                      "all_configs": get_all_configs(),
                      "training_state": {"step": curr_steps,
                                         "episode": curr_episodes,
                                         "reward": performance['per_r'],
                                         }
                      }
    torch.save(net_checkpoint, net_path)
    print(f"Saved model to {net_path}\n")
    return net_path


def wandb_eval_log(eval_data, all_configs):
    """Log evaluation data to wandb table"""
    columns = ['Agent', 'World', 'Obstacle', 'EL_mean', 'EL_std', 'MR_mean',
               'MR_std', 'CO_mean', 'CO_std', 'SR_mean', 'Dist_factor',
               'Block_factor', 'Congestion_factor']
    table = wandb.Table(columns=columns)
    for record in eval_data:
        table.add_data(
            record['eval_params'][0],  # Agent
            record['eval_params'][1],  # World size
            record['eval_params'][2],  # Obstacle density
            record['perf_mean']['episode_len'],
            record['perf_std']['episode_len'],
            record['perf_mean']['max_goals'],
            record['perf_std']['max_goals'],
            record['perf_mean']['collide'],
            record['perf_std']['collide'],
            record['perf_mean']['success_rate'],
            all_configs['DIST_FACTOR'],
            all_configs['BLOCK_FACTOR'],
            all_configs['CONGESTION_FACTOR']
        )
    wandb.log({'eval_results': table})
