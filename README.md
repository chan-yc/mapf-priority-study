# Project Title: MAPF Agent's Priority Study

This repository is part of a group project for the module COMP0124 Multi-agent Artificial Intelligence (2023/24) of University College London. The project expands upon the value-based tie breaking mechanism introduced in the paper titled "[SCRIMP: Scalable Communication for Reinforcement- and Imitation-Learning-Based Multi-Agent Pathfinding](https://ieeexplore.ieee.org/abstract/document/10342305/)" (Wang et al., 2023). This repository was forked from the authors' repository [SCRIMP](https://github.com/marmotlab/SCRIMP) and contains modifications to the original code to align with our analysis.

## Major Modifications

1. Add conda environment file for python 3.9.
2. Renamed `driver.py` to `train_model.py`.
3. Add `multi_train.py` for training multiple models consecutively.
4. Add argument parser for model training and evaluation scripts.
5. Upload final model to wandb automatically.
6. Log model evaluation result to wandb's table.
7. Add block factor and congestion factor to the probability construction in the tie breaking mechanism.

## Usage

### Prerequisite

- **Install `python==3.9` and the project dependencies**

  - Using conda, or

        $ conda env create -f environment.yml
        $ conda activate maai

  - pip

        $ pip install -r requirement.txt

- **Setup the OdrM\* package**

  - Build the package

        $ cd od_mstar3
        $ python setup.py build_ext --inplace
        $ cd ..

  - Testing

        $ python
        >>> import od_mstar3.cpp_mstar  # should be done without any error

- **Setup wandb for real-time training monitoring and evaluation result**

  - Register an account https://wandb.ai/site

  - Login to wandb on the machine

        $ conda activate maai  # make sure the environment is on
        $ wandb login          # then follows the instructions

### Model Training

- **Train a single model**

    1. Set training parameters in `alg_parameters.py`.
    2. Run the single training script.

           $ python train_model.py

    3. Trained models will be stored in the corresponding experiment directory in `models/MAPF/` as `net_checkpoint.pkl` and uploaded in wandb if 1RecordingParameters.wandb` is set to `True`.

- **Train multiple models**

    1. Set multiple sets of training configs using `CONFIG_SETS` in `multi_train.py`.
    2. Run the multi training script to train the models one by one.

           $ python multi_train.py

    3. Trained models will be stored in the corresponding experiment directory in `models/MAPF/` as `net_checkpoint.pkl` and uploaded in wandb if 1RecordingParameters.wandb` is set to `True`.

### Model Evaluation

- **Evaluate a single model**

    1. Locate the model's path, e.g. `models/MAPF/expt1/final/net_checkpoint.pkl`.
    2. Run the evaluation script.

           $ python eval_model.py models/MAPF/expt1/final/ -n expt --gpu

        Notes:
          - The model's directory is used instead of the path to `net_checkpoint.pkl`.
          - The argument after `-n` specifies the name of experiment.
          - The `--gpu` specifies the use of gpu for evaluation.

    3. Evaluation results are printed in the terminal and uploaded to wandb.

## Key Files

`alg_parameters.py` - Training parameters.

`train_model.py` - Single model training program. Holds global training network for PPO.

`multi_train.py` - Multi models training program. Allow setting multiple sets of training parameters.

`runner.py` - A single process for collecting training data.

`eval_model.py` - Single model evaluation program.

`mapf_gym.py` - Defines the classical Reinforcement Learning environment of Multi-Agent Pathfinding.

`episodic_buffer.py` - Defines the episodic buffer used to generate intrinsic rewards.

`model.py` - Defines the neural network-based operation model.

`net.py` - Defines network architecture.

## Group Members

Tian Ruen Woon ([tianruen](https://github.com/tianruen))

Ruibo Zhang ([RuiboZhang1](https://github.com/RuiboZhang1))

Yuen Chung Chan ([chan-yc](https://github.com/chan-yc))


## References

[Y. Wang, B. Xiang, S. Huang and G. Sartoretti, "SCRIMP: Scalable Communication for Reinforcement- and Imitation-Learning-Based Multi-Agent Pathfinding," 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Detroit, MI, USA, 2023, pp. 9301-9308, doi: 10.1109/IROS55552.2023.10342305.](https://ieeexplore.ieee.org/abstract/document/10342305)
