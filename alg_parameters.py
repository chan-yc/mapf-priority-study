import datetime

""" Hyperparameters of SCRIMP!"""


class EnvParameters:
    N_AGENTS = 8  # number of agents used in training
    N_ACTIONS = 5
    EPISODE_LEN = 256  # maximum episode length in training
    FOV_SIZE = 3
    WORLD_SIZE = (10, 40)  # must be tuple
    OBSTACLE_PROB = (0.0, 0.5)  # must be tuple
    ACTION_COST = -0.3
    IDLE_COST = -0.3
    GOAL_REWARD = 0.0
    COLLISION_COST = -2
    BLOCKING_COST = -1


class TrainingParameters:
    lr = 1e-5
    GAMMA = 0.95  # discount factor
    LAM = 0.95  # For GAE
    CLIP_RANGE = 0.2
    MAX_GRAD_NORM = 10
    ENTROPY_COEF = 0.01
    IN_VALUE_COEF = 0.08
    EX_VALUE_COEF = 0.08
    POLICY_COEF = 10
    VALID_COEF = 0.5
    BLOCK_COEF = 0.5
    N_EPOCHS = 5
    N_ENVS = 2  # number of processes
    N_MAX_STEPS = 1e7  # maximum number of time steps used in training
    N_STEPS = 2 ** 10  # number of time steps per process per data collection
    MINIBATCH_SIZE = int(2 ** 10)
    IMITATION_LEARNING_RATE = 0.1  # imitation learning rate


class NetParameters:
    NET_SIZE = 512
    NUM_CHANNEL = 8  # number of channels of observations -[FOV_SIZE x FOV_SIZEx NUM_CHANNEL]
    GOAL_REPR_SIZE = 12
    VECTOR_LEN = 7  # [dx, dy, d total,extrinsic rewards,intrinsic reward, min dist respect to buffer, action t-1]
    N_POSITION = 1024  # maximum number of unique ID
    D_MODEL = NET_SIZE  # for input and inner feature of attention
    D_HIDDEN = 1024  # for feed-forward network
    N_LAYERS = 1  # number of computation block
    N_HEAD = 8
    D_K = 32
    D_V = 32


# TODO Examinate the distance factor! Add more factors if needed
class TieBreakingParameters:
    DIST_FACTOR = 0.1


class IntrinsicParameters:
    K = 3  # threshold for obtaining intrinsic reward
    CAPACITY = 80
    ADD_THRESHOLD = 3
    N_ADD_INTRINSIC = 1e6  # number of steps to start giving intrinsic reward
    SURROGATE1 = 0.2
    SURROGATE2 = 1


class SetupParameters:
    SEED = 1234
    USE_GPU_LOCAL = True
    USE_GPU_GLOBAL = True
    NUM_GPU = 1


class RecordingParameters:
    RETRAIN = False
    WANDB =  True
    TENSORBOARD = False
    JSON_WRITER = True
    ENTITY = 'group'
    TIME = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    EXPERIMENT_PROJECT = 'MAPF'
    EXPERIMENT_NAME = 'SCRIMP_UPDATE'
    EXPERIMENT_NOTE = ''
    SAVE_INTERVAL = 5e5  # interval of saving model
    BEST_INTERVAL = 0  # interval of saving model with the best performance
    GIF_INTERVAL = 1e6  # interval of saving gif
    EVAL_INTERVAL = TrainingParameters.N_ENVS * TrainingParameters.N_STEPS  # interval of evaluating training model0
    EVAL_EPISODES = 1  # number of episode used in evaluation
    RECORD_BEST = False
    MODEL_PATH = './models' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + '_' + TIME
    GIFS_PATH = './gifs' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + '_' + TIME
    SUMMARY_PATH = './summaries' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + '_' + TIME
    JSON_NAME = 'alg.json'
    LOSS_NAME = ['all_loss', 'policy_loss', 'policy_entropy', 'critic_loss_in',
                 'critic_loss_ex', 'valid_loss', 'blocking_loss', 'clipfrac',
                 'grad_norm', 'advantage']


param_classes = [EnvParameters, TrainingParameters, NetParameters,
                 TieBreakingParameters, RecordingParameters]
all_configs = {}
for params in param_classes:
    all_configs[params.__name__] = {k: v for k, v in params.__dict__.items()
                                    if not k.startswith('_')}
