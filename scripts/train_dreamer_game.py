import warnings
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Relative paths in the env configs (config / image_string / inv_img_path) are
# resolved against the repo root, so run from there regardless of launch cwd.
os.chdir(ROOT)

import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym, from_dm, atari
from environments import MicrorobotEnvGameRayWrappedCont, MicrorobotEnvContGame, MicrorobotEnvGameNoCollision, MicrorobotEnvGameByTheWall, MicrorobotEnvGame8Act
import yaml
from environments.costum_wrappers.MaxandSkip import MaxAndSkipEnv
from environments.costum_wrappers.RateTargetReached import RateTargetReachedWrapper, RewardWrapper


EXP_NAME = 'vascular_large_collision_smaller_bubbles'
# With num_envs>1 the runner spawns worker processes that RE-IMPORT this module
# (spawn start method). If each worker recomputed the auto-incremented LOGDIR it
# would land on a different / uncreated dir and crash writing its CSV. So compute
# LOGDIR once in the parent and pin it via an env var that spawned children inherit.
if 'DREAMER_GAME_LOGDIR' in os.environ:
    LOGDIR = os.environ['DREAMER_GAME_LOGDIR']
else:
    _logdir_base = ROOT / 'logdir'
    _logdir_base.mkdir(parents=True, exist_ok=True)
    _candidate = _logdir_base / EXP_NAME
    _suffix = 1
    while _candidate.exists():
        _candidate = _logdir_base / f"{EXP_NAME}_{_suffix}"
        _suffix += 1
    LOGDIR = str(_candidate)
    os.makedirs(LOGDIR, exist_ok=True)
    os.environ['DREAMER_GAME_LOGDIR'] = LOGDIR
num_envs = 8  # 16-core CPU: 8 parallel env processes, leaves cores for learner/actor
yaml_config = "config_sim_by_wall"

env_config_racetrack={
    "name": (0,0),
    "config": f"scripts/{yaml_config}.yaml",
    "max_envs": 1,
    "render_mode": None,  # was "human": cv2.imshow per step x8 procs crashes/wastes CPU during training
    "image_string": 'binary_images/default_mask_racetrack_segmented.png', # binary_images/closing1.png
    }
# env_config_racetrack["inv_img_path"] = 'binary_images/default_mask_racetrack_segmented_inv_dil.png'

env_config_4_squares=env_config_racetrack.copy()
env_config_4_squares["image_string"] = 'binary_images/4_squares.png'
env_config_4_squares["inv_img_path"] = 'binary_images/4_squares_inv_dil.png'
env_config_4_squares["name"] = (0,0)

env_config_simple=env_config_racetrack.copy()
env_config_simple["image_string"] = 'binary_images/simple.png'
env_config_simple["inv_img_path"] = 'binary_images/simple_inv_dil.png'
env_config_simple["name"] = (0,0)

env_config_vascular=env_config_racetrack.copy()
env_config_vascular["image_string"] = 'binary_images/default_segmentation_vascular.png'
env_config_vascular["name"] = (0,0)

env_config_vascular_real=env_config_racetrack.copy()
env_config_vascular_real["image_string"] = 'binary_images/default_segmentation_vascular_4_closed.png'
env_config_vascular_real["name"] = (0,0)

env_config_vascular_bis=env_config_racetrack.copy()
env_config_vascular_bis["image_string"] = 'binary_images/default_segmentation_vascular_fake.png'
env_config_vascular_bis["inv_img_path"] = 'binary_images/default_segmentation_vascular_fake_inv_dil.png'
env_config_vascular_bis["name"] =(0,0)

env_config_9_squares=env_config_racetrack.copy()
env_config_9_squares["image_string"] = 'binary_images/closing_cleaned.png'
env_config_9_squares["inv_img_path"] = 'binary_images/closing_cleaned_inv_dil.png'
env_config_9_squares["name"] = (0,0)

env_config_maze_1=env_config_racetrack.copy()
env_config_maze_1["image_string"] = 'binary_images/maze_1.png'
env_config_maze_1["inv_img_path"] = 'binary_images/maze_1_inv_dil.png'
env_config_maze_1["name"] = (0,0)

env_config_maze_2=env_config_racetrack.copy()
env_config_maze_2["image_string"] = 'binary_images/maze_2.png'
env_config_maze_2["inv_img_path"] = 'binary_images/maze_2_inv_dil.png'
env_config_maze_2["name"] = (0,0)

env_config_maze_3=env_config_racetrack.copy()
env_config_maze_3["image_string"] = 'binary_images/maze_3.png'
env_config_maze_3["inv_img_path"] = 'binary_images/maze_3_inv_dil.png'
env_config_maze_3["name"] = (0,0)

env_config_maze_4=env_config_racetrack.copy()
env_config_maze_4["image_string"] = 'binary_images/maze_4.png'
env_config_maze_4["inv_img_path"] = 'binary_images/maze_4_inv_dil.png'
env_config_maze_4["name"] = (0,0)

env_config_large_vascular=env_config_racetrack.copy()
env_config_large_vascular["image_string"] = "binary_images/default_segmentation_vascular_large.png"

env_config_large_vascular_by_wall=env_config_racetrack.copy()
env_config_large_vascular_by_wall["image_string"] = "binary_images/default_segmentation_vascular_large.png"
# env_config_large_vascular_by_wall["inv_img_path"] = "binary_images/vascular_dilated.png"


def main(env_configs):
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['medium'])
    # 'large' (deter=2048, units=768) is overkill for a 64x64 binary-channel
    # nav task and overfits on little data; 'medium' (deter=1024, units=640)
    # fits a single 3090 with room to spare and trains noticeably faster.
    # config = config.update(dreamerv3.configs['debug'])
    config = config.update({
        'logdir': LOGDIR,
        'run.train_ratio': -1,
        'run.log_every': 200,  # Seconds
        'run.save_every': 200,  # Seconds
        'batch_size': 16,
        'jax.prealloc': True,
        'encoder.mlp_keys': '$^', #'agent_position|target_position',
        'decoder.mlp_keys': '$^', #'agent_position|target_position',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
        'jax.platform': 'gpu',
        'run.actor_batch': num_envs,
        'run.log_keys_mean': '^log_.*',
        'run.log_keys_max': '^log_.*',
         # Still have to run with 'Explore' for now
    })
    config["wrapper"].update({"length": 50})
    config = embodied.Flags(config).parse()
    logdir = embodied.Path(config.logdir)
    os.makedirs(logdir, exist_ok=True)
    if os.path.exists(logdir / 'config.yaml'):
        print(f'\033[1;31mExperiment already exists at {logdir}.\033[0m')
    else:
        yaml.dump(config, open(logdir / 'config.yaml', 'w'))
        yaml.dump(env_config_racetrack, open(logdir / 'env_config.yaml', 'w'))
    
    step = embodied.Counter()
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandBOutput(logdir.name, config),
        # embodied.logger.MLFlowOutput(logdir.name),
    ])
    envs = envs_gen(dreamer_config=config, configs=env_configs)
    env = embodied.BatchEnv([*envs], parallel=False)
    env.close()
    
    for config_env in env_configs:
        config_ = config_env["config"]
        name = config_env["image_string"]
        name = name.split("/")[-1].split(".")[0]
        with open(config_, 'r') as stream:
            try:
                config_ = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        config_path = f"{logdir}/general_config.yaml"
        with open(config_path, 'w') as stream:
            try:
                yaml.safe_dump(config_, stream)
            except yaml.YAMLError as exc:
                print(exc)
        with open(f"{logdir}/env_config_{name}.yaml", 'w') as stream:
            try:
                yaml.safe_dump(config_, stream)
            except yaml.YAMLError as exc:
                print(exc)
    
    #print(env.obs_space, env.act_space,)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / 'replay',
        min_size=5000)  # prefill: don't start training until 5k transitions
                        # are collected, avoids early overfit on a near-empty buffer
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)
    
    envs_gen_cls_ = envs_gen_cls(dreamer_config=config, configs=env_configs)
    
    embodied.run.parallel(agent, replay, logger, envs_gen_cls_, num_envs, args)
    
    # embodied.run.train(agent, env, replay, logger, args)
    # checkpoint = embodied.Checkpoint(f"{args.logdir}/checkpoint.ckpt")
    # args = args.update({"from_checkpoint": f"{args.logdir}/checkpoint.ckpt"})
    # embodied.run.eval_only(agent, env, logger, args)

def envs_gen(dreamer_config, configs):
    for config in configs:
        env = MicrorobotEnvGameByTheWall(**config)  # Replace this with your Gym env.
        env = MaxAndSkipEnv(env, 4)
        env = RateTargetReachedWrapper(env, 100, dreamer=True, verbose=1)
        env = RewardWrapper(env, logdir=None, dreamer=True)
        env = from_gym.FromGym(env)
        env = dreamerv3.wrap_env(env, dreamer_config)
        yield env

class envs_gen_cls():
    def __init__(self, dreamer_config, configs):
        self.envs = []
        self.count = -1
        self.dreamer_config = dreamer_config
        self.configs = configs
    
    def __call__(self, i=0, *args, **kwds):
        self.count += 1
        print(f"envs_gen_cls: {self.count}")

        os.makedirs(LOGDIR, exist_ok=True)  # spawned worker may not have created it yet
        config = self.configs[i]
        env = MicrorobotEnvGameByTheWall(**config)  # Replace this with your Gym env.
        env = MaxAndSkipEnv(env, 4)
        env = RateTargetReachedWrapper(env, 100, dreamer=True, verbose=1, logdir=f"{LOGDIR}/rate_target_reached_env_{i}.csv")
        env = RewardWrapper(env, logdir=f"{LOGDIR}/reward_{i}.csv", dreamer=True)
        env = from_gym.FromGym(env)
        env = dreamerv3.wrap_env(env, self.dreamer_config)
        return env


if __name__ == '__main__':
    # config = [env_config_racetrack, env_config_large_vascular, env_config_maze_1, env_config_maze_2, env_config_maze_3, env_config_maze_4]
    config = [env_config_simple, env_config_4_squares, env_config_9_squares, env_config_racetrack, env_config_vascular_real, env_config_vascular_bis] 
    # config = [env_config_vascular_real] # , env_config_vascular_bis, env_config_vascular_real, env_config_vascular_real, env_config_vascular_real]
    # config = [env_config_vascular_real for _ in range(num_envs)]
    config = [env_config_large_vascular for _ in range(num_envs)]
    # config = [env_config_large_vascular_by_wall for _ in range(num_envs)]
    main(config)