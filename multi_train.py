import os
import argparse
import datetime

from train_model import train_model
from alg_parameters import (SetupParameters, NetParameters, EnvParameters,
                            TrainingParameters, TieBreakingParameters,
                            RecordingParameters)


# Define multiple configurations
# Change the parameters as needed
# Must provide a unique expt_name for each configuration
CONFIG_SETS = [
    {
        'expt_name': 'expt_1',
        'params': [
            [RecordingParameters, 'DIST_FACTOR', 0.1],
            [RecordingParameters, 'BLOCK_FACTOR', 0.1],
            [RecordingParameters, 'CONGESTION_FACTOR', 0.1],
        ]
    },
    {
        'expt_name': 'expt_2',
        'params': [
            [RecordingParameters, 'DIST_FACTOR', 0.1],
            [RecordingParameters, 'BLOCK_FACTOR', 0.2],
            [RecordingParameters, 'CONGESTION_FACTOR', 0.3],
        ]
    },
]


def multi_train(wandb_id=None, retrain_path=None):
    """Train the model with multiple configurations"""
    print(f'Running {len(CONFIG_SETS)} experiments.\n')

    for n, configs in enumerate(CONFIG_SETS):

        print(f'[Experiment: {n+1}/{len(CONFIG_SETS)}]')

        # Update parameters
        update_expt_info(configs['expt_name'])
        for param_cls, param_name, param_val in configs['params']:
            setattr(param_cls, param_name, param_val)

        # Full model training
        if retrain_path is not None:
            RecordingParameters.RETRAIN = True
        else:
            RecordingParameters.RETRAIN = False

        for _ in range(10):  # retry 10 times
            status, wandb_id = train_model(wandb_id, retrain_path)
            if status != 'failed':
                break
            else:
                RecordingParameters.RETRAIN = True
                retrain_path = os.path.join(RecordingParameters.MODEL_PATH, 'final')


def update_expt_info(name):
    """Update experiment information"""
    project = RecordingParameters.EXPERIMENT_PROJECT
    expt_time = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    RecordingParameters.TIME = expt_time
    RecordingParameters.EXPERIMENT_NAME = name
    RecordingParameters.MODEL_PATH = os.path.join('models', project, name + '_' + expt_time)
    RecordingParameters.GIFS_PATH = os.path.join('gifs', project, name + '_' + expt_time)
    RecordingParameters.SUMMARY_PATH = os.path.join('summaries', project, name + '_' + expt_time)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--retrain-path', type=str, default=None, help='Retrain the model')
    parser.add_argument('-w', '--wandb-id', type=str, default=None, help='Use wandb')

    args = parser.parse_args()

    multi_train(args.wandb_id, args.retrain_path)
