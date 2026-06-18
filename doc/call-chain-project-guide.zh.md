# 项目调用链导读（按执行路径梳理）

> 文档目标：这份文档不重复 [README](../README.md) 的研究背景与结果展示，而是从“代码真正如何跑起来”的角度，按调用链梳理这个项目。适合你在阅读源码、定位入口、理解仿真/真实系统差异时使用。

---

## 1. 项目总览

这个仓库可以分成四条主线：

1. **仿真 PPO 训练/评估**
2. **仿真 Dreamer 训练/评估**
3. **真实世界 Dreamer 在线训练/控制**
4. **感知、硬件、路径规划的横向支撑链**

从代码结构上看，最重要的三层是：

- [scripts/](../scripts/)：训练、评估、运行入口 + YAML 配置
- [environments/](../environments/)：环境定义，负责 observation / action / reward / collision / reset
- [utils/](../utils/)：视觉、跟踪、路径规划、Arduino、函数发生器等支撑组件

配套资源包括：

- [binary_images/](../binary_images/)：二值地图、障碍图、血管通道 mask
- [results/](../results/)：README/论文展示素材
- `Post_processing/`：已有实验结果和后处理示例

---

## 2. 仓库结构速览（从调用角度看）

### 2.1 入口层：`scripts/`

主入口文件：

- [scripts/train_ppo_game.py](../scripts/train_ppo_game.py)：仿真 PPO 训练
- [scripts/play_ppo.py](../scripts/play_ppo.py)：PPO 回放/评估
- [scripts/train_dreamer_game.py](../scripts/train_dreamer_game.py)：仿真 Dreamer 训练
- [scripts/play_dreamer.py](../scripts/play_dreamer.py)：Dreamer 回放/评估
- [scripts/train_dreamer_real.py](../scripts/train_dreamer_real.py)：真实世界 Dreamer 在线训练/控制
- [scripts/train_ppo_real.py](../scripts/train_ppo_real.py)：旧版/较轻量的真实 PPO 入口
- [scripts/play_RRT_Sweep.py](../scripts/play_RRT_Sweep.py)：RRT + sweeping 控制运行脚本

### 2.2 环境层：`environments/`

核心环境文件：

- [environments/microrobot_env.py](../environments/microrobot_env.py)：所有环境共享的基类
- [environments/game_env_8_actions.py](../environments/game_env_8_actions.py)：离散 8 动作仿真环境
- [environments/game_env_dreamer_rand_freq.py](../environments/game_env_dreamer_rand_freq.py)：带 amplitude/frequency 采样逻辑的仿真环境
- [environments/game_env_dreamer_cont.py](../environments/game_env_dreamer_cont.py)：连续动作仿真环境骨架
- [environments/game_env_by_the_wall.py](../environments/game_env_by_the_wall.py)：贴壁/流场偏置版本仿真环境
- [environments/ARSL_env_camera_dreamer.py](../environments/ARSL_env_camera_dreamer.py)：真实相机环境
- [environments/env_camera_sweeping.py](../environments/env_camera_sweeping.py)：真实 sweep 模式环境
- [environments/env_camera_continous.py](../environments/env_camera_continous.py)：真实连续控制 + 路径规划环境
- [environments/env_camera_sweeping_rrt.py](../environments/env_camera_sweeping_rrt.py)：真实 RRT + sweeping 版本

### 2.3 工具层：`utils/`

关键组件：

- [utils/actuator.py](../utils/actuator.py)：Arduino + 函数发生器控制
- [utils/tektronix_func_gen.py](../utils/tektronix_func_gen.py)：Tektronix AFG 底层控制
- [utils/tracking_CSRT.py](../utils/tracking_CSRT.py)：CSRT 跟踪器
- [utils/image_postprocessing.py](../utils/image_postprocessing.py)：图像后处理、裁剪、cluster 提取
- [utils/segmentation.py](../utils/segmentation.py)：SAM 分割封装
- [utils/path_planning_v2.py](../utils/path_planning_v2.py)：RRT* 路径规划
- [utils/data_saving.py](../utils/data_saving.py)：数据保存

---

## 3. 环境继承主干（先建立大图景）

在看训练脚本之前，先看环境继承关系会更容易理解。

### 3.1 仿真环境主干

```text
BaseMicrorobotEnv
  └─ MicrorobotEnvContGame
       ├─ MicrorobotEnvContGameFreq
       ├─ MicrorobotEnvContGameFreqResampled
       │    └─ MicrorobotEnvGame8Act
       └─ MicrorobotEnvContinousGame
```

对应文件：

- [environments/microrobot_env.py](../environments/microrobot_env.py)
- [environments/game_env_dreamer_cont.py](../environments/game_env_dreamer_cont.py)
- [environments/game_env_dreamer_rand_freq.py](../environments/game_env_dreamer_rand_freq.py)
- [environments/game_env_8_actions.py](../environments/game_env_8_actions.py)

此外还有一条贴壁/特殊动力学分支：

```text
BaseMicrorobotEnv
  └─ ...
      └─ MicrorobotEnvGameNoCollision
           └─ MicrorobotEnvGameByTheWall
```

对应文件：

- [environments/game_env_no_collision.py](../environments/game_env_no_collision.py)
- [environments/game_env_by_the_wall.py](../environments/game_env_by_the_wall.py)

### 3.2 真实环境主干

```text
BaseMicrorobotEnv
  └─ MicrorobotEnv   (真实相机 + tracker + actuator)
       ├─ MicrorobotEnvSweeping
       ├─ MicrorobotEnvContinous
       └─ RRTEnvSweeping / 其他变体
```

对应文件：

- [environments/ARSL_env_camera_dreamer.py](../environments/ARSL_env_camera_dreamer.py)
- [environments/env_camera_sweeping.py](../environments/env_camera_sweeping.py)
- [environments/env_camera_continous.py](../environments/env_camera_continous.py)
- [environments/env_camera_sweeping_rrt.py](../environments/env_camera_sweeping_rrt.py)

---

## 4. 调用链一：仿真 PPO 训练/评估

这条链最适合第一次读项目的人，因为它最像标准 RL 管线。

---

### 4.1 训练入口

入口文件：

- [scripts/train_ppo_game.py](../scripts/train_ppo_game.py)

主流程可以浓缩为：

```text
train_ppo_game.py
  -> 构造 env_config
  -> 实例化 MicrorobotEnvGame8Act
  -> 包 wrapper
  -> 创建 stable_baselines3.PPO
  -> 注册 callback
  -> model.learn()
  -> 输出 checkpoint / csv / tensorboard
```

关键代码职责：

1. 设置实验名和日志目录
   - `EXP_NAME`
   - `LOGDIR`

2. 配置环境参数
   - `env_config`
   - 默认引用 [scripts/config_sim_6_envs.yaml](../scripts/config_sim_6_envs.yaml)
   - 默认地图引用如 [binary_images/default_mask_racetrack_segmented.png](../binary_images/default_mask_racetrack_segmented.png)

3. 实例化环境
   - `env = MicrorobotEnvGame8Act(**env_config)`
   - `eval_env = MicrorobotEnvGame8Act(**eval_env_config)`

4. 用 wrapper 包环境
   - `MaxAndSkipEnv`
   - `RateTargetReachedWrapper`
   - `RewardWrapper`
   - `NewApi`
   - `gym.wrappers.StepAPICompatibility`
   - `TimeLimit`
   - `RecordEpisodeStatistics`

5. 创建 PPO 模型
   - `PPO("MultiInputPolicy", env, ...)`

6. 注册 callback
   - `CheckpointCallback`
   - `EvalCallback`
   - `RateTargetReachedCallback`

7. 开始训练
   - `model.learn(...)`

---

### 4.2 PPO 仿真环境具体怎么走

环境类：

- [environments/game_env_8_actions.py](../environments/game_env_8_actions.py)

`MicrorobotEnvGame8Act` 做的事情是：

1. 定义 8 个方向动作映射：
   - 上下左右
   - 左上/右上/左下/右下
2. 在 `step(action)` 中：
   - `act_num = action % NUMBER_PIEZOS`
   - 将离散动作映射为方向向量 `direction`
   - 用 `_amplitude_from_action(action, act_num)` 生成 amplitude
   - 调父类 `super()._post_step(direction, amplitude)`

也就是说，**PPO 输出的是离散动作编号**，环境把它解释成：

- “往哪个方向推”
- “这一步的 amplitude 多大”

---

### 4.3 amplitude 是怎么来的

文件：

- [environments/game_env_dreamer_rand_freq.py](../environments/game_env_dreamer_rand_freq.py)

`MicrorobotEnvGame8Act` 的父类是 `MicrorobotEnvContGameFreqResampled`。

这个类里的 `_amplitude_from_action(...)` 并不是简单固定值，而是：

1. 以前一时刻的 amplitude 为基础
2. 乘一个随机系数 `uniform(0.5, 1.5)`
3. clip 到 `[MIN_AMPLITUDE, MAX_AMPLITUDE]`
4. 少量概率下重新随机初始化 amplitude

所以 PPO 仿真环境不是“动作 = 固定方向 + 固定幅值”，而是：

- 动作主要决定方向
- 幅值带有随机漂移/重采样

这让仿真更贴近真实实验中的不确定性。

---

### 4.4 真正的一步环境动力学在哪里

文件：

- [environments/game_env_dreamer_cont.py](../environments/game_env_dreamer_cont.py)

`MicrorobotEnvContGame._post_step(direction, amplitude)` 是仿真环境核心：

1. 调 `is_valid(...)` 判断移动后是否会撞障碍
2. 如果合法：
   - `self.agent_location = self.move_agent(direction, amplitude)`
3. 如果不合法：
   - 标记 collision
   - 提前 terminate
   - 返回 `reward_collision`
4. 如果合法走完一步后，再判断：
   - 是否 collision
   - 是否到达 target
   - 否则基于距离算 dense reward
5. 返回：
   - observation
   - reward
   - done
   - info

这一步里用到的几何基础、合法性判断、目标采样，都来自基类。

---

### 4.5 基类 `BaseMicrorobotEnv` 在仿真里负责什么

文件：

- [environments/microrobot_env.py](../environments/microrobot_env.py)

这是 PPO 和 Dreamer 仿真共同依赖的底座。它负责：

#### observation space

```text
{
  image,
  agent_position,
  target_position,
}
```

#### action space

- 默认是 `Discrete(TOTAL_ACTIONS)`

#### reward function

支持：
- linear
- quadratic
- log
- inverse
- inverse_squared

#### 碰撞与目标逻辑

关键函数：
- `check_collision(...)`
- `find_legal_point(...)`
- `find_legal_point_target_close(...)`
- `_sample_safe_action(...)`
- `is_valid(...)`

所以如果你想知道：

- reward 怎么设计
- target 怎么采样
- 什么时候算撞墙

优先看这个文件。

---

### 4.6 PPO 使用的 wrapper 链

PPO 训练脚本里包环境的顺序很重要：

1. `MaxAndSkipEnv`
2. `RateTargetReachedWrapper`
3. `RewardWrapper`
4. `NewApi`
5. `StepAPICompatibility`
6. `TimeLimit`
7. `RecordEpisodeStatistics`

涉及文件：

- `environments/costum_wrappers/MaxandSkip.py`
- `environments/costum_wrappers/RateTargetReached.py`
- `environments/costum_wrappers/NewApi.py`

其中最关键的是：

#### `MaxAndSkipEnv`
作用：
- 做 frame skip / action repeat
- 降低训练频率，平滑环境反馈

#### `RateTargetReachedWrapper`
作用：
- 统计每隔一段时间目标到达率
- 支持把结果写到 CSV

#### `RewardWrapper`
作用：
- 记录奖励信息
- 输出 reward 日志

---

### 4.7 PPO 训练的配置、资产与输出

#### 配置来源

- [scripts/config_sim_6_envs.yaml](../scripts/config_sim_6_envs.yaml)
- [scripts/config_sim.yaml](../scripts/config_sim.yaml)
- [scripts/config_PPO.yaml](../scripts/config_PPO.yaml)

重点配置字段：

- `Action_space_settings`
- `Reward Settings`
- `General_environment_settings`
- `Layout_settings`

#### 地图资产来源

典型地图：
- [binary_images/default_mask_racetrack_segmented.png](../binary_images/default_mask_racetrack_segmented.png)
- [binary_images/4_squares.png](../binary_images/4_squares.png)
- [binary_images/simple.png](../binary_images/simple.png)
- [binary_images/default_segmentation_vascular_4_closed.png](../binary_images/default_segmentation_vascular_4_closed.png)

#### 典型输出

训练时会写：

- `config.yaml`
- `env_config.yaml`
- `reward_train.csv`
- `reward_eval.csv`
- `rate_target_reached_train.csv`
- `rate_target_reached_eval.csv`
- checkpoint zip
- TensorBoard 日志

---

### 4.8 PPO 回放/评估调用链

入口：

- [scripts/play_ppo.py](../scripts/play_ppo.py)

浓缩版：

```text
play_ppo.py
  -> 读取训练目录/ckpt
  -> 构造环境
  -> PPO.load(...)
  -> predict(obs)
  -> env.step(action)
```

它基本复用了训练时的环境结构，只是不再 `learn()`，而是反复：

- 载入模型
- 推理动作
- 执行动作
- 记录/显示结果

---

## 5. 调用链二：仿真 Dreamer 训练/评估

这条线比 PPO 更复杂，因为中间多了 Dreamer 的 adapter、replay、logger、parallel runner。

---

### 5.1 训练入口

入口文件：

- [scripts/train_dreamer_game.py](../scripts/train_dreamer_game.py)

浓缩版：

```text
train_dreamer_game.py
  -> 组装 Dreamer config
  -> 定义 env_config_* 地图集合
  -> envs_gen() 生成环境
  -> from_gym.FromGym
  -> dreamerv3.wrap_env
  -> dreamerv3.Agent
  -> replay/logger
  -> embodied.run.parallel()
```

Dreamer 训练脚本做了几件 PPO 没有的事：

1. 从 `dreamerv3.configs['defaults']`、`dreamerv3.configs['large']` 生成基础配置
2. 构造 `embodied.Logger`
3. 构造 `embodied.replay.Uniform`
4. 用 `embodied.run.parallel(...)` 跑 actor/learner
5. 使用 `from_gym.FromGym` 和 `dreamerv3.wrap_env` 适配环境接口

---

### 5.2 Dreamer 仿真默认用哪个环境

训练脚本里最关键的环境生成逻辑在：

- `envs_gen(...)`
- `envs_gen_cls.__call__(...)`

都在 [scripts/train_dreamer_game.py](../scripts/train_dreamer_game.py) 中。

这两个工厂默认构造的是：

- [environments/game_env_by_the_wall.py](../environments/game_env_by_the_wall.py)
  - `MicrorobotEnvGameByTheWall`

这点非常重要：

> **Dreamer 仿真默认不是 `MicrorobotEnvGame8Act`，而是更复杂的 `MicrorobotEnvGameByTheWall`。**

这说明 Dreamer 仿真更偏向：
- 血管式障碍
- 贴壁行为
- 流场或边界诱导

---

### 5.3 `MicrorobotEnvGameByTheWall` 比普通仿真环境多了什么

文件：

- [environments/game_env_by_the_wall.py](../environments/game_env_by_the_wall.py)

这个类继承自 `MicrorobotEnvGameNoCollision`，但做了额外增强：

1. 如果没有提供 `inv_img_path`
   - 从 `image_string` 读入灰度地图
   - 调 `RRTStar.inverse_dilation(...)`
   - 自动生成 `*_inv_dil.png`
2. 载入 `inv_dil_obs`
3. 引入：
   - `reward_center`
   - `flow_direction`
4. 在 `_post_step(...)` 中加入：
   - 靠壁/边界附近的处理
   - flow 推动
   - bubble 半径衰减
   - 终止条件变化

这意味着 Dreamer 学到的环境动力学，比 PPO 那条离散仿真链更复杂。

---

### 5.4 Dreamer 为什么要 `from_gym.FromGym` 和 `dreamerv3.wrap_env`

Dreamer 不直接吃标准 Gym 环境，所以在 `train_dreamer_game.py` 里会做两层适配：

#### 第一层：`from_gym.FromGym`
作用：
- 把 Gym/Gymnasium 风格环境封成 Dreamer 可接受的接口

#### 第二层：`dreamerv3.wrap_env`
作用：
- 做 Dreamer 自己的 observation/action 包装
- 对接 world model 训练所需的格式

所以 Dreamer 的真实输入链不是：

```text
Gym env -> Agent
```

而是：

```text
Gym env -> FromGym -> wrap_env -> Dreamer Agent
```

---

### 5.5 Dreamer 的 logger / replay / parallel runner

这些逻辑都在 [scripts/train_dreamer_game.py](../scripts/train_dreamer_game.py)。

#### `embodied.Logger`
输出：
- 终端日志
- `metrics.jsonl`
- TensorBoard

#### `embodied.replay.Uniform`
输出：
- `replay/` 目录

#### `embodied.run.parallel(...)`
作用：
- 并行 actor / learner / env runner
- 是 Dreamer 训练的核心驱动方式

这和 PPO 很不同：

- PPO 是 `model.learn(...)`
- Dreamer 是 `embodied.run.parallel(...)`

---

### 5.6 多地图训练是怎么组织的

在 [scripts/train_dreamer_game.py](../scripts/train_dreamer_game.py) 里定义了很多 `env_config_*`：

- `env_config_racetrack`
- `env_config_4_squares`
- `env_config_simple`
- `env_config_vascular`
- `env_config_vascular_real`
- `env_config_vascular_bis`
- `env_config_maze_1` ~ `maze_4`
- `env_config_large_vascular`
- `env_config_large_vascular_by_wall`

这些配置本质上是通过切换：

- `image_string`
- `inv_img_path`
- `config`

来切换地图和动力学设定。

对应资产主要来自：

- [binary_images/](../binary_images/)

因此 Dreamer 的多环境预训练主逻辑可以概括为：

```text
多个 env_config_* -> 不同 binary_images 地图 -> 同一 Dreamer 管线训练
```

---

### 5.7 Dreamer 的训练输出

Dreamer 训练常见输出：

- `metrics.jsonl`
- TensorBoard
- `replay/`
- `reward_i.csv`
- `rate_target_reached_env_i.csv`
- `general_config.yaml`
- `env_config_<map>.yaml`

仓库里的后处理目录也能帮助你理解 Dreamer 产物长什么样：

- `Post_processing/random_env_pretrain/`
- `Post_processing/Finetune_vascular_obstacles/`

虽然这些结果大多不在当前工作区浅层展示，但它们体现了日志/配置/CSV 的组织习惯。

---

### 5.8 Dreamer 回放/评估调用链

入口文件：

- [scripts/play_dreamer.py](../scripts/play_dreamer.py)

浓缩版：

```text
play_dreamer.py
  -> 构造 Dreamer config
  -> 构造环境 + wrappers
  -> Agent(...)
  -> embodied.Checkpoint(...)
  -> embodied.run.eval_only(...)
```

与训练相比，差异主要是：

- 不再训练
- 从 `checkpoint.ckpt` 恢复
- 使用 `eval_only(...)`

---

## 6. 调用链三：真实世界训练/控制/推理

这是项目最复杂、也最有研究价值的部分。

这里要先明确一个判断：

> **当前仓库中，真实世界主线应该以 Dreamer real 为主，而不是 PPO real。**

原因：
- [scripts/train_ppo_real.py](../scripts/train_ppo_real.py) 更像旧版/轻量入口
- [scripts/train_dreamer_real.py](../scripts/train_dreamer_real.py) 才把相机、tracker、Arduino、函数发生器、logger、parallel runner 串成了完整闭环

---

### 6.1 旧版 Real PPO 链（作为补充）

入口：

- [scripts/train_ppo_real.py](../scripts/train_ppo_real.py)

它的特点：
- 比较短
- 直接引用 `environments.old.*`
- 更像早期实验入口

所以研究真实系统时，不建议把它当主线。

---

### 6.2 Real Dreamer 主入口

入口文件：

- [scripts/train_dreamer_real.py](../scripts/train_dreamer_real.py)

浓缩版：

```text
train_dreamer_real.py
  -> 构造 Dreamer config/logger/replay
  -> make_env(fake_env=True) 初始化 obs/act space
  -> make_env(fake_env=False) 构造真实环境
  -> embodied.run.parallel(...)
  -> 相机采图 + tracker + actuator 闭环
```

这个脚本里的几个关键点：

1. 定义真实环境配置：
   - `env_config`
   - `env_config_continous`

2. 指定真实日志根目录：
   - `MAIN_PATH`
   - `LOGDIR`

3. `run_parallel(...)`
   - 先用 fake env 建立 Dreamer 所需 observation/action 形状
   - 再启动真实环境 actor

4. `make_env(...)`
   - `fake_env=True` 时，构造假的 sweep env
   - `fake_env=False` 时，构造真实 `MicrorobotEnvSweeping`

---

### 6.3 为什么要 fake env

文件：

- [environments/env_camera_sweeping.py](../environments/env_camera_sweeping.py)

`MicrorobotEnvSweeping(fake=True, ...)` 的作用不是训练替身，而是：

- 读配置文件
- 构造 observation_space
- 构造 action_space
- 不去连接真实硬件

这使得 Dreamer 可以先知道：

- observation 是什么 shape
- action 是什么 shape

然后再把真实环境接上去。

所以它本质上是一个：

> **“接口占位环境”**

---

### 6.4 真实环境冷启动调用链

真实环境核心文件：

- [environments/ARSL_env_camera_dreamer.py](../environments/ARSL_env_camera_dreamer.py)

`MicrorobotEnv.__init__(...)` 是整条真实链的冷启动中心。

它做的事非常多：

1. 调 `super().__init__(config)` 读取基础环境配置
2. 创建实验目录 `_make_folders(...)`
3. 初始化 Arduino
   - `Arduino(...)`
4. 初始化函数发生器
   - `FunctionGenerator_1(...)`
5. 打开相机
   - `cv2.VideoCapture(0)`
6. 处理 ROI
   - 可以用 `default_image_path/default_mask`
   - 也可以人工选择 ROI
7. 采初始帧
8. 裁剪并 resize
   - `resize_and_crop_frame(...)`
9. 生成 segmentation mask
   - 默认 mask 模式，或在线分割模式
10. 生成 cleaned image
   - `plot_cluster_on_image_blue(...)`
11. 初始化 tracker
   - `CSRT_tracker(...)`
12. 如果配置里有路径规划：
   - 初始化 `RRTStar(...)`
13. 采样 target
14. 构造 area -> vpp 的映射 `_vpp_from_area`

这意味着一个真实环境实例化过程，本身就已经做完了：

- 相机准备
- 视觉初始化
- 硬件准备
- 初始 tracking
- 初始 target 设定

---

### 6.5 真实环境的感知反馈链

真实 observation 的产生在：

- `MicrorobotEnv._get_obs()`
- 文件：[environments/ARSL_env_camera_dreamer.py](../environments/ARSL_env_camera_dreamer.py)

调用链：

```text
video_stream.read()
  -> resize_and_crop_frame(...)
  -> plot_cluster_on_image_blue(...)
  -> tracker.track(...)
  -> 失败则 _reinitialize_tracker_autonomously(...)
  -> 叠加 target 点
  -> 保存多种中间图像
  -> 返回 observation
```

返回的 observation 结构：

```text
{
  image,
  agent_position,
  target_position,
}
```

这里的 `image` 不是原始显微镜图像，而是：

- 已裁剪
- 已清洗
- 已高亮蓝色 cluster
- 已叠加 target 的版本

也就是说，**Dreamer 在真实环境里看到的是“任务相关处理后的图像”，不是原始相机帧。**

---

### 6.6 真实 action 是如何打到硬件的

仍在 [environments/ARSL_env_camera_dreamer.py](../environments/ARSL_env_camera_dreamer.py) 里。

`step(action)` 做的事情：

1. 从 tracker 获取 bubble area
2. 根据 area 算电压 `vpp`
3. `function_generator.set_vpp(vpp)`
4. 设置频率
   - 当前代码里大量使用随机频率或轻微随机采样
5. `arduino.set_piezo_from_action(action)`
6. `time.sleep(STEP_DURATION)`
7. 调 `_post_step(...)`

所以真实一步控制链是：

```text
policy action
  -> piezo 选择
  -> 频率/电压设置
  -> 超声作用于真实微气泡
  -> 相机重新观测
  -> tracker 更新状态
  -> reward / done
```

---

### 6.7 真实 reward / terminate / logging 在哪里发生

文件：

- [environments/ARSL_env_camera_dreamer.py](../environments/ARSL_env_camera_dreamer.py)

`_post_step(piezo, vpp, freq)` 负责：

1. 记录一行实验数据到 `experiment_data.csv`
2. 清空本步 done 标志
3. 调 `_get_obs()` 得到动作后的新观测
4. 再根据当前位置判断：
   - collision
   - target reached
   - 否则 distance reward
5. 更新：
   - `cumulative_reward`
   - `last_reward`
   - `last_piezo`
6. 返回：
   - observation
   - reward
   - done
   - info

这说明真实环境与仿真环境的抽象是一致的，只是：

- 仿真环境内部自己 move agent
- 真实环境通过硬件执行后再读回 agent 的真实位置

---

### 6.8 Sweeping 模式真实环境

文件：

- [environments/env_camera_sweeping.py](../environments/env_camera_sweeping.py)

`MicrorobotEnvSweeping` 是真实环境的一个特化版本：

1. 初始化时切换函数发生器到 sweep 模式：
   - `set_sweep_mode()`
   - `set_sweep_limits(...)`
2. 每一步：
   - 仍根据 bubble area 设置 `vpp`
   - 通过 `arduino.set_piezo_from_action(action)` 选择 piezo
   - 当前频率由 sweep generator 决定，并通过 `get_frequency()` 读取
   - 然后走 `_post_step(...)`

所以 sweeping 版本和普通 real 版本的区别是：

- 普通版：每步可显式设频率
- sweep 版：频率在一个区间内扫，策略主要控制 piezo 和/或 amplitude

---

### 6.9 连续控制 + 路径规划版本

文件：

- [environments/env_camera_continous.py](../environments/env_camera_continous.py)
- [environments/env_camera_sweeping_rrt.py](../environments/env_camera_sweeping_rrt.py)

这一组环境是项目里更“系统工程化”的部分。

它们引入了：
- RRT 路径规划
- waypoint 跟踪
- piezo 高层选择
- policy 只调低层频率/振幅

也就是说控制被拆成了两层：

### 高层
- 路径规划 / waypoint
- 决定“现在大致应该往哪边走”

### 低层
- policy 选择频率/振幅
- 硬件实际输出超声驱动

这和“端到端直接输出所有动作”的架构不完全一样。

---

## 7. 调用链四：感知 / 硬件 / 路径规划横向模块

这一节把真实系统里横跨多个环境的公共模块单独拎出来讲。

---

### 7.1 感知链：相机 -> 分割 -> 蓝色 cluster -> tracker

核心文件：

- [environments/ARSL_env_camera_dreamer.py](../environments/ARSL_env_camera_dreamer.py)
- [utils/image_postprocessing.py](../utils/image_postprocessing.py)
- [utils/segmentation.py](../utils/segmentation.py)
- [utils/tracking_CSRT.py](../utils/tracking_CSRT.py)

浓缩版：

```text
cv2.VideoCapture(0)
  -> ROI crop
  -> segmentation mask
  -> plot_cluster_on_image_blue(...)
  -> CSRT_tracker.track(...)
  -> 得到 agent center / bubble area / bbox
  -> 构造成 observation
```

---

### 7.2 分割模块：SAM + 掩膜模式

文件：

- [utils/segmentation.py](../utils/segmentation.py)

`ImageSegmentation` 是对 SAM 的封装：

- `sam_model_registry['vit_h']`
- `SamPredictor`
- 根据点提示做 mask 预测

真实系统里有两种通道 mask 来源：

1. **直接加载默认 mask**
   - 例如 [binary_images/default_segmentation_vascular_large.png](../binary_images/default_segmentation_vascular_large.png)
2. **在线分割**
   - 从初始帧做阈值和 SAM 分割得到

这意味着真实系统既支持：
- “用预先准备好的通道模板”
- 也支持“现场重新初始化分割”

---

### 7.3 图像后处理模块

文件：

- [utils/image_postprocessing.py](../utils/image_postprocessing.py)

关键职责：

- `resize_and_crop_frame(...)`
- `plot_cluster_on_image_blue(...)`
- 通道合法区域处理
- cluster 提取
- target 合法点相关工具

其中 `plot_cluster_on_image_blue(...)` 很关键，因为 tracker 不是直接跟原图，而是跟“蓝色高亮后的 clean 图”。

---

### 7.4 跟踪模块：CSRT tracker

文件：

- [utils/tracking_CSRT.py](../utils/tracking_CSRT.py)

核心职责：

- 初始化 tracker
- 每帧更新 bounding box
- 统计 bubble area
- 提供 agent center
- 在跟踪失败时支持重初始化

所以在真实系统里，agent 的“位置”并不是由环境内部模拟出来，而是由 tracker 视觉估计出来。

---

### 7.5 硬件执行模块：Arduino + 函数发生器

文件：

- [utils/actuator.py](../utils/actuator.py)
- [utils/tektronix_func_gen.py](../utils/tektronix_func_gen.py)

#### Arduino

负责：
- 打开串口
- 根据 action 选择 piezo 通道
- 关闭时发停止信号

关键接口：
- `set_piezo_from_action(...)`
- `set_piezo_by_number(...)`
- `close()`

#### FunctionGenerator_1

负责：
- 设置振幅 `set_vpp(...)`
- 设置频率 `set_frequency(...)`
- 设置 sweep 模式 `set_sweep_mode()`
- 设置 sweep 区间 `set_sweep_limits(...)`
- 开关输出 `turn_on()/turn_off()`

这些接口最终依赖 [utils/costum_fgen.py](../utils/costum_fgen.py) / [utils/tektronix_func_gen.py](../utils/tektronix_func_gen.py) 中的底层 VISA 控制。

> 注意：这些模块会直接访问真实串口和仪器。没有硬件时不要直接运行相关入口脚本。

---

### 7.6 路径规划模块：RRT*

文件：

- [utils/path_planning_v2.py](../utils/path_planning_v2.py)

核心类：
- `RRTStar`

职责：
- 在二值障碍图上规划路径
- 做 obstacle dilation / inverse_dilation
- 生成从 start 到 end 的 waypoint 路径

在项目里的用途：

1. **仿真环境里**
   - 生成 `inv_dil` 地图
   - 支持 by-wall / 安全边界逻辑

2. **真实环境里**
   - 给连续控制环境提供 waypoint/path
   - 支持 RRT + sweeping 版本

---

## 8. 关键配置、资产、日志输出对照表

### 8.1 配置文件

| 类别 | 代表文件 | 用途 |
|---|---|---|
| 仿真 PPO / Dreamer 通用仿真配置 | [scripts/config_sim_6_envs.yaml](../scripts/config_sim_6_envs.yaml) | 动作空间、奖励、碰撞、地图尺寸、多环境设置 |
| 基础仿真配置 | [scripts/config_sim.yaml](../scripts/config_sim.yaml) | 单环境/基础仿真参数 |
| PPO 配置 | [scripts/config_PPO.yaml](../scripts/config_PPO.yaml) | PPO 相关实验设定 |
| 真实系统主配置 | [scripts/config.yaml](../scripts/config.yaml) | Arduino、Tektronix、tracker、SAM、动作参数 |
| sweep 配置 | [scripts/config_sweep.yaml](../scripts/config_sweep.yaml) | sweep 模式实验参数 |
| 其他专用配置 | [scripts/config_flow.yaml](../scripts/config_flow.yaml)、[scripts/config_sweep_rrt.yaml](../scripts/config_sweep_rrt.yaml) | 流场、RRT+sweep 等专项实验 |

重点字段经常包括：

- `Action_space_settings`
- `Reward Settings`
- `General_environment_settings`
- `Layout_settings`
- `Arduino_settings`
- `Tektronix_settings`
- `CSRT_Tracker_settings`
- `sam_config`

---

### 8.2 地图 / mask / 静态资产

| 类别 | 代表文件 | 用途 |
|---|---|---|
| racetrack 地图 | [binary_images/default_mask_racetrack_segmented.png](../binary_images/default_mask_racetrack_segmented.png) | 仿真赛道 |
| square 地图 | [binary_images/4_squares.png](../binary_images/4_squares.png) | 仿真基本场景 |
| square 膨胀图 | [binary_images/4_squares_inv_dil.png](../binary_images/4_squares_inv_dil.png) | 安全边界 / 路径规划 |
| simple 地图 | [binary_images/simple.png](../binary_images/simple.png) | 简化调试场景 |
| vascular 地图 | [binary_images/default_segmentation_vascular.png](../binary_images/default_segmentation_vascular.png) | 血管仿真 |
| large vascular 地图 | [binary_images/default_segmentation_vascular_large.png](../binary_images/default_segmentation_vascular_large.png) | 大尺寸血管图 |
| 真实环境闭合/后处理图 | [binary_images/default_segmentation_vascular_4_closed.png](../binary_images/default_segmentation_vascular_4_closed.png) | 真实系统默认 mask |
| 迷宫地图 | [binary_images/maze_1.png](../binary_images/maze_1.png) 等 | 多环境泛化 |

---

### 8.3 结果展示资产

这些不是训练链的一部分，但能帮助你理解作者最终展示了什么：

- [results/Figure 1a.png](../results/Figure%201a.png)
- [results/Figure 1F.png](../results/Figure%201F.png)
- [results/Figure 1g.png](../results/Figure%201g.png)
- [results/Movie S1.gif](../results/Movie%20S1.gif) ~ [results/Movie S7.gif](../results/Movie%20S7.gif)

---

### 8.4 运行时日志与输出

| 线路 | 典型产物 | 说明 |
|---|---|---|
| PPO 仿真 | `config.yaml`, `env_config.yaml`, `reward_*.csv`, `rate_target_reached_*.csv`, checkpoint, TensorBoard | SB3 风格训练产物 |
| Dreamer 仿真 | `metrics.jsonl`, `replay/`, TensorBoard, `general_config.yaml`, `env_config_<name>.yaml` | Dreamer / embodied 风格产物 |
| 真实 Dreamer | `experiment_data.csv`, 原始帧、tracking 图、blue 图、downsized 图、RRT 图 | 实验级别数据与过程可视化 |

---

## 9. 四条调用链的浓缩版速查

### 9.1 仿真 PPO

```text
scripts/train_ppo_game.py
  -> environments/game_env_8_actions.py: MicrorobotEnvGame8Act
  -> environments/game_env_dreamer_rand_freq.py: amplitude 采样
  -> environments/game_env_dreamer_cont.py: _post_step
  -> environments/microrobot_env.py: reward/collision/target
  -> wrappers
  -> stable_baselines3.PPO.learn()
```

### 9.2 仿真 Dreamer

```text
scripts/train_dreamer_game.py
  -> envs_gen / envs_gen_cls
  -> environments/game_env_by_the_wall.py
  -> wrappers
  -> from_gym.FromGym
  -> dreamerv3.wrap_env
  -> dreamerv3.Agent + replay + logger
  -> embodied.run.parallel()
```

### 9.3 真实 Dreamer / 控制主线

```text
scripts/train_dreamer_real.py
  -> make_env(fake=True) 只建接口
  -> make_env(fake=False) 构造真实 sweeping env
  -> environments/ARSL_env_camera_dreamer.py
  -> utils/actuator.py
  -> utils/image_postprocessing.py
  -> utils/tracking_CSRT.py
  -> camera -> tracker -> reward -> logger
```

### 9.4 感知 / 硬件 / 路径规划

```text
cv2.VideoCapture
  -> ROI crop
  -> segmentation mask
  -> blue cluster extraction
  -> CSRT tracking
  -> agent position
  -> Arduino / Tektronix 执行动作
  -> RRTStar / PathFollower 提供高层引导
```

---

## 10. 推荐阅读顺序

### 10.1 第一次看这个项目

建议顺序：

1. [README.md](../README.md)
2. [scripts/train_ppo_game.py](../scripts/train_ppo_game.py)
3. [environments/game_env_8_actions.py](../environments/game_env_8_actions.py)
4. [environments/game_env_dreamer_cont.py](../environments/game_env_dreamer_cont.py)
5. [environments/microrobot_env.py](../environments/microrobot_env.py)

原因：
- 先用最标准的 PPO 仿真链建立总体认知
- 再回到环境基类理解 reward/collision/target 设计

---

### 10.2 想看 Dreamer 仿真

建议顺序：

1. [scripts/train_dreamer_game.py](../scripts/train_dreamer_game.py)
2. [environments/game_env_by_the_wall.py](../environments/game_env_by_the_wall.py)
3. [environments/game_env_dreamer_cont.py](../environments/game_env_dreamer_cont.py)
4. `environments/costum_wrappers/RateTargetReached.py`
5. [scripts/play_dreamer.py](../scripts/play_dreamer.py)

重点关注：
- `envs_gen()`
- `dreamerv3.wrap_env(...)`
- `embodied.run.parallel(...)`

---

### 10.3 想看真实系统

建议顺序：

1. [scripts/train_dreamer_real.py](../scripts/train_dreamer_real.py)
2. [environments/env_camera_sweeping.py](../environments/env_camera_sweeping.py)
3. [environments/ARSL_env_camera_dreamer.py](../environments/ARSL_env_camera_dreamer.py)
4. [utils/actuator.py](../utils/actuator.py)
5. [utils/tracking_CSRT.py](../utils/tracking_CSRT.py)
6. [utils/image_postprocessing.py](../utils/image_postprocessing.py)
7. [utils/path_planning_v2.py](../utils/path_planning_v2.py)

重点关注：
- `step()` 如何控制 piezo / 频率 / vpp
- `_get_obs()` 如何从图像构造 observation
- `_post_step()` 如何计算真实 reward

---

### 10.4 想看实验室调试方式

建议看：

- [test_scripts/manual_control_linux.py](../test_scripts/manual_control_linux.py)
- [test_scripts/test_funcgen.py](../test_scripts/test_funcgen.py)
- [test_scripts/test_connections_ubuntu.py](../test_scripts/test_connections_ubuntu.py)
- [test_scripts/Shape_Forming.py](../test_scripts/Shape_Forming.py)

这些脚本有助于理解：
- 硬件怎么单独调
- 相机怎么单独测
- 真实实验不是只有 RL 入口，还有很多手动/半自动调试脚本

---

## 11. 备注与边界

1. **真实系统主线请优先看 Dreamer real**  
   [scripts/train_ppo_real.py](../scripts/train_ppo_real.py) 存在，但更像旧版入口；完整的相机 + tracker + actuator 闭环在 [scripts/train_dreamer_real.py](../scripts/train_dreamer_real.py)。

2. **路径中存在作者本机硬编码目录**  
   例如代码里可见 `/home/m4/...`、`/home/mahmoud/...`。研究代码时要区分：
   - 哪些是仓库内相对路径
   - 哪些是作者训练时的本机输出目录

3. **`test_scripts/` 不是标准单元测试目录**  
   更准确地说，它是“实验脚本 / 调试脚本 / 手动控制脚本集合”。

4. **不要把 README 和本文档混为一谈**  
   README 负责讲“研究做了什么”；本文负责讲“代码是怎么跑起来的”。

5. **仿真环境变体很多，但阅读时先抓主线**  
   先看：
   - PPO -> `MicrorobotEnvGame8Act`
   - Dreamer -> `MicrorobotEnvGameByTheWall`
   - Real -> `MicrorobotEnvSweeping` / `MicrorobotEnv`

---

## 12. 一句话总结

如果把整个项目压缩成一句话：

> 这是一个把**二值地图仿真环境、Dreamer/PPO 强化学习、相机视觉反馈、Arduino/PZT 控制、Tektronix 超声驱动、以及可选的 RRT 路径规划**整合在一起的微机器人控制系统；理解它最有效的方法，不是从目录开始，而是从 **`scripts -> environments -> utils -> assets/logs`** 这条调用链一路追下去。
