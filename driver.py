import os
import os.path as osp

import numpy as np
import ray
import setproctitle
from torch.utils.tensorboard import SummaryWriter
import torch
import wandb

from alg_parameters import (SetupParameters, EnvParameters, TrainingParameters,
                            NetParameters, RecordingParameters, all_args)
from episodic_buffer import EpisodicBuffer
from mapf_gym import MAPFEnv
from model import Model
from runner import Runner
from util import (set_global_seeds, write_to_tensorboard, write_to_wandb,
                  make_gif, reset_env, one_step, update_perf, get_torch_device)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ray.init(num_gpus=SetupParameters.NUM_GPU)
print("Welcome to SCRIMP on MAPF!\n")


def main():
    """main code"""

    # Prepare for training
    if RecordingParameters.RETRAIN:
        restore_path = './local_model'
        net_path_checkpoint = restore_path + "/net_checkpoint.pkl"
        net_dict = torch.load(net_path_checkpoint)

    if RecordingParameters.WANDB:
        if RecordingParameters.RETRAIN:
            wandb_id = None
        else:
            wandb_id = wandb.util.generate_id()
        wandb.init(project=RecordingParameters.EXPERIMENT_PROJECT,
                   name=RecordingParameters.EXPERIMENT_NAME,
                   entity=RecordingParameters.ENTITY,
                   notes=RecordingParameters.EXPERIMENT_NOTE,
                   config=all_args,
                   id=wandb_id,
                   resume='allow')
        print('id is:{}'.format(wandb_id))
        print('Launching wandb...\n')

    if RecordingParameters.TENSORBOARD:
        if RecordingParameters.RETRAIN:
            summary_path = ''
        else:
            summary_path = RecordingParameters.SUMMARY_PATH
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        global_summary = SummaryWriter(summary_path)
        print('Launching tensorboard...\n')

        if RecordingParameters.TXT_WRITER:
            txt_path = summary_path + '/' + RecordingParameters.TXT_NAME
            with open(txt_path, "w") as f:
                f.write(str(all_args))
            print('Logging txt...\n')

    set_global_seeds(SetupParameters.SEED)
    setproctitle.setproctitle(RecordingParameters.EXPERIMENT_PROJECT
                              + RecordingParameters.EXPERIMENT_NAME + "@"
                              + RecordingParameters.ENTITY)

    # Create classes
    global_device = get_torch_device(use_gpu=SetupParameters.USE_GPU_GLOBAL)
    local_device = get_torch_device(use_gpu=SetupParameters.USE_GPU_LOCAL)
    global_model = Model(0, global_device, True)

    if RecordingParameters.RETRAIN:
        global_model.network.load_state_dict(net_dict['model'])
        global_model.net_optimizer.load_state_dict(net_dict['optimizer'])

    envs = [Runner.remote(i + 1) for i in range(TrainingParameters.N_ENVS)]
    eval_env = MAPFEnv(num_agents=EnvParameters.N_AGENTS, mode='eval')
    eval_memory = EpisodicBuffer(0, EnvParameters.N_AGENTS)

    if RecordingParameters.RETRAIN:
        curr_steps = net_dict["step"]
        curr_episodes = net_dict["episode"]
        best_perf = net_dict["reward"]
    else:
        curr_steps, curr_episodes, best_perf = 0, 0, 0

    last_test_t = -RecordingParameters.EVAL_INTERVAL - 1
    last_model_t = -RecordingParameters.SAVE_INTERVAL - 1
    last_best_t = -RecordingParameters.BEST_INTERVAL - 1
    last_gif_t = -RecordingParameters.GIF_INTERVAL - 1

    # Start training
    try:
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            # Collect network weights
            if global_device != local_device:
                net_weights = global_model.network.to(local_device).state_dict()
                global_model.network.to(global_device)
            else:
                net_weights = global_model.network.state_dict()
            net_weights_id = ray.put(net_weights)
            curr_steps_id = ray.put(curr_steps)

            # Decide whether to use imitation learning for this iteration
            imitation = np.random.rand() < TrainingParameters.IMITATION_LEARNING_RATE
            if imitation:  # Compute imitation learning data using ODrM*
                jobs = [env.imitation.remote(net_weights_id, curr_steps_id) for env in envs]
            else:  # Compute reinforcement learning data
                jobs = [env.run.remote(net_weights_id, curr_steps_id) for env in envs]

            # Wait for all jobs to finish and collect results
            done_jobs, _ = ray.wait(jobs, num_returns=TrainingParameters.N_ENVS)
            job_results = ray.get(done_jobs)

            # Get imitation learning data
            if imitation:
                # Mini-batch imitation data
                # obs, vector, actions, hid_states, message
                mb_imit_data = [[] for _ in range(5)]

                # Append mini-batch data
                for result in job_results:
                    for i in range(len(mb_imit_data)):
                        mb_imit_data[i].append(result[i])
                    curr_episodes += result[-2]
                    curr_steps += result[-1]

                # Concatenate mini-batch data
                for i in range(len(mb_imit_data)):
                    mb_imit_data[i] = np.concatenate(mb_imit_data[i], axis=0)

                # Training using imitation learning data
                data_len = len(mb_imit_data[0])
                mb_imitation_loss = []
                for start in range(0, data_len, TrainingParameters.MINIBATCH_SIZE):
                    end = start + TrainingParameters.MINIBATCH_SIZE
                    batch_data = [arr[start:end] for arr in mb_imit_data]
                    loss = global_model.imitation_train(*batch_data)
                    mb_imitation_loss.append(loss)
                mb_imitation_loss = np.nanmean(mb_imitation_loss, axis=0)

                # Record training result
                if RecordingParameters.WANDB:
                    write_to_wandb(curr_steps, imitation_loss=mb_imitation_loss, evaluate=False)
                if RecordingParameters.TENSORBOARD:
                    write_to_tensorboard(global_summary, curr_steps, imitation_loss=mb_imitation_loss, evaluate=False)

            # Get reinforcement learning data
            else:
                # Mini-batch RL data
                # obs, vector, returns_in, returns_ex, returns_all, values_in,
                # values_ex, values_all, actions, ps, hid_states, train_valid,
                # blocking, message
                mb_rl_data = [[] for _ in range(14)]
                metrics = {
                    'per_r': [], 'per_in_r': [], 'per_ex_r': [], 'per_valid_rate': [],
                    'per_episode_len': [], 'per_block': [], 'per_leave_goal': [],
                    'per_final_goals': [], 'per_half_goals': [], 'per_block_acc': [],
                    'per_max_goals': [], 'per_num_collide': [], 'rewarded_rate': []
                }

                # Append mini-batch data
                for result in job_results:
                    for i in range(len(mb_rl_data)):
                        mb_rl_data[i].append(result[i])
                    curr_episodes += result[-2]
                    for metric in metrics.keys():
                        metrics[metric].append(np.nanmean(result[-1][metric]))

                for i in metrics.keys():
                    metrics[i] = np.nanmean(metrics[i])

                curr_steps += len(done_jobs) * TrainingParameters.N_STEPS

                # Concatenate mini-batch data
                for i in range(len(mb_rl_data)):
                    mb_rl_data[i] = np.concatenate(mb_rl_data[i], axis=0)

                # Training using reinforcement learning data
                mb_loss = []
                data_len = len(done_jobs) * TrainingParameters.N_STEPS
                for _ in range(TrainingParameters.N_EPOCHS):
                    # Shuffle data sequence
                    idx = np.random.choice(data_len, size=data_len, replace=False)
                    for start in range(0, data_len, TrainingParameters.MINIBATCH_SIZE):
                        batch_idx = idx[start:start+TrainingParameters.MINIBATCH_SIZE]
                        batch_data = [arr[batch_idx] for arr in mb_rl_data]
                        mb_loss.append(global_model.train(*batch_data))

                # Record training result
                if RecordingParameters.WANDB:
                    write_to_wandb(curr_steps, metrics, mb_loss, evaluate=False)
                if RecordingParameters.TENSORBOARD:
                    write_to_tensorboard(global_summary, curr_steps, metrics, mb_loss, evaluate=False)

            # Evaluate model
            if (curr_steps - last_test_t) / RecordingParameters.EVAL_INTERVAL >= 1.0:
                # if save gif
                if (curr_steps - last_gif_t) / RecordingParameters.GIF_INTERVAL >= 1.0:
                    save_gif = True
                    last_gif_t = curr_steps
                else:
                    save_gif = False

                # evaluate training model
                last_test_t = curr_steps
                with torch.no_grad():
                    # greedy_eval_performance_dict = evaluate(eval_env,eval_memory, global_model,
                    # global_device, save_gif, curr_steps, True)
                    eval_performance_dict = evaluate(eval_env, eval_memory, global_model, global_device, save_gif,
                                                     curr_steps, False)
                # record evaluation result
                if RecordingParameters.WANDB:
                    # write_to_wandb(curr_steps, greedy_eval_performance_dict, evaluate=True, greedy=True)
                    write_to_wandb(curr_steps, eval_performance_dict, evaluate=True, greedy=False)
                if RecordingParameters.TENSORBOARD:
                    # write_to_tensorboard(global_summary, curr_steps, greedy_eval_performance_dict, evaluate=True,
                    #                      greedy=True)
                    write_to_tensorboard(global_summary, curr_steps, eval_performance_dict, evaluate=True, greedy=False,
                                         )

                print('episodes: {}, step: {},episode reward: {}, final goals: {} \n'.format(
                    curr_episodes, curr_steps, eval_performance_dict['per_r'],
                    eval_performance_dict['per_final_goals']))
                # save model with the best performance
                if RecordingParameters.RECORD_BEST:
                    if eval_performance_dict['per_r'] > best_perf and (
                            curr_steps - last_best_t) / RecordingParameters.BEST_INTERVAL >= 1.0:
                        best_perf = eval_performance_dict['per_r']
                        last_best_t = curr_steps
                        print('Saving best model \n')
                        model_path = osp.join(RecordingParameters.MODEL_PATH, 'best_model')
                        if not os.path.exists(model_path):
                            os.makedirs(model_path)
                        path_checkpoint = model_path + "/net_checkpoint.pkl"
                        net_checkpoint = {"model": global_model.network.state_dict(),
                                          "optimizer": global_model.net_optimizer.state_dict(),
                                          "step": curr_steps,
                                          "episode": curr_episodes,
                                          "reward": best_perf}
                        torch.save(net_checkpoint, path_checkpoint)

            # Save model
            if (curr_steps - last_model_t) / RecordingParameters.SAVE_INTERVAL >= 1.0:
                last_model_t = curr_steps
                print('Saving Model !\n')
                model_path = osp.join(RecordingParameters.MODEL_PATH, '%.5i' % curr_steps)
                os.makedirs(model_path)
                path_checkpoint = model_path + "/net_checkpoint.pkl"
                net_checkpoint = {"model": global_model.network.state_dict(),
                                  "optimizer": global_model.net_optimizer.state_dict(),
                                  "step": curr_steps,
                                  "episode": curr_episodes,
                                  "reward": eval_performance_dict['per_r']}
                torch.save(net_checkpoint, path_checkpoint)

    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")
    finally:
        # save final model
        print('Saving Final Model !\n')
        model_path = RecordingParameters.MODEL_PATH + '/final'
        os.makedirs(model_path)
        path_checkpoint = model_path + "/net_checkpoint.pkl"
        net_checkpoint = {"model": global_model.network.state_dict(),
                          "optimizer": global_model.net_optimizer.state_dict(),
                          "step": curr_steps,
                          "episode": curr_episodes,
                          "reward": eval_performance_dict['per_r']}
        torch.save(net_checkpoint, path_checkpoint)
        if RecordingParameters.TENSORBOARD:
            global_summary.close()
        # killing
        for e in envs:
            ray.kill(e)
        if RecordingParameters.WANDB:
            wandb.finish()


def evaluate(eval_env, episodic_buffer, model, device, save_gif, curr_steps, greedy):
    """Evaluate Model."""
    eval_performance_dict = {'per_r': [], 'per_ex_r': [], 'per_in_r': [], 'per_valid_rate': [], 'per_episode_len': [],
                             'per_block': [], 'per_leave_goal': [], 'per_final_goals': [], 'per_half_goals': [],
                             'per_block_acc': [], 'per_max_goals': [], 'per_num_collide': [], 'rewarded_rate': []}
    episode_frames = []

    for i in range(RecordingParameters.EVAL_EPISODES):
        num_agent = EnvParameters.N_AGENTS

        # reset environment and buffer
        message = torch.zeros((1, num_agent, NetParameters.NET_SIZE)).to(device)
        hidden_state = (torch.zeros((num_agent, NetParameters.NET_SIZE // 2)).to(device),
                        torch.zeros((num_agent, NetParameters.NET_SIZE // 2)).to(device))

        done, valid_actions, obs, vector, _ = reset_env(eval_env, num_agent)
        episodic_buffer.reset(curr_steps, num_agent)
        new_xy = eval_env.get_positions()
        episodic_buffer.batch_add(new_xy)

        one_episode_perf = {'num_step': 0, 'episode_reward': 0, 'invalid': 0, 'block': 0,
                            'num_leave_goal': 0, 'wrong_blocking': 0, 'num_collide': 0, 'reward_count': 0,
                            'ex_reward': 0, 'in_reward': 0}
        if save_gif:
            episode_frames.append(eval_env._render(mode='rgb_array', screen_width=900, screen_height=900))

        # stepping
        while not done:
            # predict
            actions, pre_block, hidden_state, num_invalid, v_all, ps, message = model.evaluate(obs, vector,
                                                                                               valid_actions,
                                                                                               hidden_state,
                                                                                               greedy,
                                                                                               episodic_buffer.no_reward,
                                                                                               message, num_agent)
            one_episode_perf['invalid'] += num_invalid

            # move
            rewards, valid_actions, obs, vector, _, done, _, num_on_goals, one_episode_perf, max_on_goals, \
                _, _, on_goal = one_step(eval_env, one_episode_perf, actions, pre_block, model, v_all, hidden_state,
                                         ps, episodic_buffer.no_reward, message, episodic_buffer, num_agent)

            new_xy = eval_env.get_positions()
            processed_rewards, be_rewarded, intrinsic_reward, min_dist = episodic_buffer.if_reward(new_xy, rewards,
                                                                                                   done, on_goal)
            one_episode_perf['reward_count'] += be_rewarded
            vector[:, :, 3] = rewards
            vector[:, :, 4] = intrinsic_reward
            vector[:, :, 5] = min_dist

            if save_gif:
                episode_frames.append(eval_env._render(mode='rgb_array', screen_width=900, screen_height=900))

            one_episode_perf['episode_reward'] += np.sum(processed_rewards)
            one_episode_perf['ex_reward'] += np.sum(rewards)
            one_episode_perf['in_reward'] += np.sum(intrinsic_reward)
            if one_episode_perf['num_step'] == EnvParameters.EPISODE_LEN // 2:
                eval_performance_dict['per_half_goals'].append(num_on_goals)

            if done:
                # save gif
                if save_gif:
                    if not os.path.exists(RecordingParameters.GIFS_PATH):
                        os.makedirs(RecordingParameters.GIFS_PATH)
                    images = np.array(episode_frames)
                    make_gif(images,
                             '{}/steps_{:d}_reward{:.1f}_final_goals{:.1f}_greedy{:d}.gif'.format(
                                 RecordingParameters.GIFS_PATH,
                                 curr_steps, one_episode_perf[
                                     'episode_reward'],
                                 num_on_goals, greedy))
                    save_gif = False

                eval_performance_dict = update_perf(one_episode_perf, eval_performance_dict, num_on_goals, max_on_goals,
                                                    num_agent)

    # average performance of multiple episodes
    for i in eval_performance_dict.keys():
        eval_performance_dict[i] = np.nanmean(eval_performance_dict[i])

    return eval_performance_dict


if __name__ == "__main__":
    main()
