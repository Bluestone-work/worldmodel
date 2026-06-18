# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements model-based and model-free reinforcement learning for ultrasound-driven microrobots in both simulation and real hardware experiments. The codebase combines:

- simulation environments driven by binary obstacle/channel images
- PPO training/evaluation via Stable-Baselines3
- DreamerV3 training/evaluation via an external `dreamerv3` dependency
- real-world camera-based environments with tracker feedback
- hardware control for Arduino-switched PZTs and a Tektronix function generator
- path planning utilities (RRT*) and post-processing scripts

The best high-level code walkthrough currently in the repo is:

- [doc/call-chain-project-guide.zh.md](doc/call-chain-project-guide.zh.md)

Read that first when you need the project call chains.

## Important dependency notes

### Dreamer dependency is fragile

`requirements.txt` installs Dreamer from:

- `git+https://github.com/Pippo809/dreamerv3.git`

This package may install incompletely into `site-packages` (notably missing `dreamerv3/configs.yaml`), which causes `import dreamerv3` to fail before repository code runs. If Dreamer scripts fail on `configs.yaml` missing, inspect the installed package contents first rather than assuming the local training script is wrong.

### Gym/Gymnasium API mismatch exists in this codebase

This repository mixes old Gym-style APIs and Gymnasium-style APIs. Recent local fixes were made to the PPO path so some wrappers now bridge 4-return and 5-return `step()` styles, but the codebase should still be treated as API-fragile. When changing wrappers or envs, verify whether the current path expects:

- old style: `obs, reward, done, info`
- new style: `obs, reward, terminated, truncated, info`

Do not assume consistency across all scripts and wrappers.

## Common commands

Run all commands from the repository root unless noted otherwise.

### Environment setup

Create and activate a Python environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Conda-style environments are also used in practice; the user currently runs from an environment similar to `us_microbot`.

### PPO simulation training

Main entrypoint:

```bash
python scripts/train_ppo_game.py
```

Notes:
- this script was patched to bootstrap the repository root into `sys.path`
- it now defaults logs to `logdir/` inside the repository instead of the author’s machine-specific `/media/...` path
- experiment directories auto-increment if the base experiment name already exists
- this path is still sensitive to Gymnasium/SB3 wrapper compatibility

### PPO evaluation / playback

```bash
python scripts/play_ppo.py
```

### Dreamer simulation training

From the `scripts/` directory or repository root depending on how you invoke it:

```bash
python scripts/train_dreamer_game.py
```

If this fails on `dreamerv3/configs.yaml`, diagnose the external dependency installation first.

### Dreamer evaluation

```bash
python scripts/play_dreamer.py
```

### Real-world Dreamer control/training

```bash
python scripts/train_dreamer_real.py
```

This accesses real hardware and camera pipelines. Do not run unless the imaging system, Arduino-switched PZTs, and Tektronix function generator are connected and configured.

### Real-world RRT + sweep control

```bash
python scripts/play_RRT_Sweep.py
```

### Test and experiment scripts

The `test_scripts/` directory is not a clean unit-test suite. It is a collection of hardware tests, image-processing experiments, manual-control scripts, and environment probes. Most scripts are run directly, for example:

```bash
python test_scripts/test.py 4_squares_inv_dil.png
python test_scripts/test_funcgen.py
python test_scripts/test_connections_ubuntu.py
```

Do not assume `pytest test_scripts` will work.

### Running a single wrapper / environment check

Useful lightweight probes include:

```bash
python test_scripts/test_game.py
python test_scripts/test_wrappers.py
```

Whether these succeed depends on the current Gym/Gymnasium compatibility state.

## Architecture

### 1. Entry scripts (`scripts/`)

`scripts/` contains the top-level workflows:

- PPO simulation training/eval: `train_ppo_game.py`, `play_ppo.py`
- Dreamer simulation training/eval: `train_dreamer_game.py`, `play_dreamer.py`
- real-world Dreamer training/control: `train_dreamer_real.py`
- real-world/experimental control modes: `play_RRT_Sweep.py`, `train_ppo_real.py`
- YAML experiment/environment configurations: `config*.yaml`

These scripts are the best starting point when you need to understand how a training or evaluation run is assembled.

### 2. Environment layer (`environments/`)

The environment hierarchy is the core of the repository.

#### Shared base

- `environments/microrobot_env.py`

`BaseMicrorobotEnv` defines the common abstractions used across simulation and real-world environments:

- observation layout (`image`, `agent_position`, `target_position`)
- reward function selection from YAML
- collision checking
- target sampling and legal point logic
- shared geometry helpers

#### Simulation environments

The main simulation path is built from:

- `environments/game_env_dreamer_cont.py`
- `environments/game_env_dreamer_rand_freq.py`
- `environments/game_env_8_actions.py`
- `environments/game_env_by_the_wall.py`
- `environments/game_env_no_collision.py`

Important structure:

- `MicrorobotEnvContGame` is the main image-based simulation environment
- `MicrorobotEnvContGameFreq` / `MicrorobotEnvContGameFreqResampled` add amplitude/frequency-related action behavior
- `MicrorobotEnvGame8Act` maps discrete actions to 8 movement directions and is the standard PPO simulation environment
- `MicrorobotEnvGameByTheWall` adds wall-following / inverse-dilation / flow-related behavior and is central to Dreamer simulation runs

Simulation maps come from `binary_images/*.png`.

#### Real-world environments

The real experimental pipeline is centered on:

- `environments/ARSL_env_camera_dreamer.py`
- `environments/env_camera_sweeping.py`
- `environments/env_camera_continous.py`
- `environments/env_camera_sweeping_rrt.py`

`ARSL_env_camera_dreamer.py` is the critical real-world environment implementation. It handles:

- camera acquisition via OpenCV
- ROI setup and resizing/cropping
- segmentation mask loading or generation
- CSRT-based tracking of the microrobot/bubble
- actuator and function generator control
- experiment logging and image saving

The real-world envs are not simple wrappers around simulation envs; they are the hardware-control core of the project.

### 3. Wrappers (`environments/costum_wrappers/`)

The wrappers are important and historically brittle.

Key files:

- `MaxandSkip.py`
- `RateTargetReached.py`
- `NewApi.py`

Their responsibilities include:

- frame skipping / action repeat
- logging rate of target completion and episode rewards to CSV
- bridging old/new step-reset API expectations

Because wrapper behavior has been patched for compatibility, verify wrapper return signatures before refactoring or stacking new wrappers.

### 4. Utility layer (`utils/`)

`utils/` contains the supporting systems that make the environments work.

Most important utilities:

- `utils/actuator.py`: Arduino and function generator control; maps actions to piezo activation, frequency, amplitude, sweep mode, etc.
- `utils/tektronix_func_gen.py`: lower-level Tektronix / VISA interaction
- `utils/tracking_CSRT.py`: OpenCV CSRT tracking wrapper used in real-world envs
- `utils/image_postprocessing.py`: frame crop/resize, cluster extraction, image cleanup, legal point helpers
- `utils/segmentation.py`: SAM-based segmentation wrapper
- `utils/path_planning_v2.py`: RRT* path planning and dilation/inverse-dilation helpers
- `utils/data_saving.py`: experiment data persistence helpers

Conceptually, `utils/` is where perception, hardware, and planning are implemented.

### 5. Assets and outputs

#### Static assets

- `binary_images/`: simulation obstacle maps, vascular masks, inverse-dilated masks, maze variants, etc.
- `results/`: README/paper figures and GIFs

#### Runtime outputs

- `logdir/`: PPO logs and similar local outputs after recent path fixes
- external paths still appear in some scripts and configs (`/home/m4/...`, `/home/mahmoud/...`, `/media/...`); these are author-machine assumptions, not portable defaults
- real-world runs also write image sequences and CSVs such as `experiment_data.csv`

## Practical guidance for modifications

### When touching PPO simulation

Start from:

- `scripts/train_ppo_game.py`
- `environments/game_env_8_actions.py`
- `environments/game_env_dreamer_cont.py`
- `environments/microrobot_env.py`
- `environments/costum_wrappers/MaxandSkip.py`
- `environments/costum_wrappers/RateTargetReached.py`

This path is the one most likely to break on API mismatches.

### When touching Dreamer simulation

Start from:

- `scripts/train_dreamer_game.py`
- `environments/game_env_by_the_wall.py`
- `environments/game_env_dreamer_cont.py`
- `environments/costum_wrappers/RateTargetReached.py`

But first confirm the external `dreamerv3` installation is actually usable.

### When touching real-world code

Start from:

- `scripts/train_dreamer_real.py`
- `environments/ARSL_env_camera_dreamer.py`
- `environments/env_camera_sweeping.py`
- `utils/actuator.py`
- `utils/tracking_CSRT.py`
- `utils/image_postprocessing.py`
- `utils/path_planning_v2.py`

Assume these paths may directly access:

- camera hardware
- Arduino serial ports
- Tektronix instruments via VISA

### When diagnosing slow training

If PPO training appears slow even while logging `Using cuda device`, the bottleneck is often environment rollout generation rather than neural network training. Relevant causes in this codebase include:

- image-based observations generated every step
- Python-heavy collision and reward logic
- `MaxAndSkipEnv(env, 4)` multiplying internal step cost
- single-environment sampling (`max_envs: 1` in the PPO path)
- wrapper overhead

GPU use alone does not imply fast end-to-end training in this repository.

## References

Important repository docs and summaries:

- [README.md](README.md)
- [doc/call-chain-project-guide.zh.md](doc/call-chain-project-guide.zh.md)
