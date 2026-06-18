# 服务器部署指南 / Server Setup Guide

## 快速开始 / Quick Start

### 1. 克隆代码 / Clone Repository

```bash
git clone https://github.com/Bluestone-work/worldmodel.git
cd worldmodel
```

### 2. 创建Conda环境 / Create Conda Environment

```bash
conda env create -f environment.yml
conda activate us_microbot
```

### 3. 安装Python依赖 / Install Python Dependencies

```bash
pip install -r requirements-server.txt
```

**注意**: 安装过程可能需要10-20分钟，特别是编译JAX和PyTorch的CUDA扩展时。

### 4. 验证安装 / Verify Installation

```bash
# 检查CUDA是否可用
python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())"
python -c "import jax; print('JAX devices:', jax.devices())"

# 检查DreamerV3安装
python -c "import dreamerv3; print('DreamerV3 OK')"
```

**Troubleshooting**: 如果 `import dreamerv3` 失败并提示 `configs.yaml` 缺失，这是Pippo809 fork的已知问题。检查：

```bash
python -c "import dreamerv3; print(dreamerv3.__file__)"
# 输出类似: /path/to/site-packages/dreamerv3/__init__.py
# 检查该目录下是否有 configs.yaml
```

如果缺失，从源码手动复制：
```bash
git clone https://github.com/Pippo809/dreamerv3.git /tmp/dreamerv3
cp /tmp/dreamerv3/dreamerv3/configs.yaml $(python -c "import dreamerv3, os; print(os.path.dirname(dreamerv3.__file__))")
```

### 5. 运行训练 / Run Training

#### DreamerV3 仿真训练（推荐用于服务器）

```bash
# 主训练脚本 - by-the-wall 环境
python scripts/train_dreamer_game.py

# 随机化环境变体
python scripts/train_dreamer_game_randomized.py
```

#### PPO 仿真训练

```bash
python scripts/train_ppo_game.py
```

**日志位置**: 默认写入到 `logdir/` 目录（在代码仓库内）。可以通过环境变量覆盖：

```bash
export DREAMER_GAME_LOGDIR=/path/to/your/logdir
python scripts/train_dreamer_game.py
```

## GPU 要求 / GPU Requirements

### CUDA 版本

- **JAX**: 需要 CUDA 12.x 兼容的NVIDIA驱动
- **PyTorch**: 自动检测CUDA版本（2.0.1支持CUDA 11.7-12.x）

### 显存要求

- **DreamerV3 训练**: 推荐 ≥16GB VRAM (可在配置中调整 `batch_size`)
- **PPO 训练**: 推荐 ≥8GB VRAM

### 检查驱动和CUDA

```bash
nvidia-smi  # 查看驱动版本和CUDA版本
```

如果驱动版本过低（<525.60），需要更新NVIDIA驱动以支持CUDA 12。

## 训练配置 / Training Configuration

### DreamerV3 配置

主要配置文件位于 `scripts/config_*.yaml`：

- `config_sim_by_wall.yaml` - 沿墙导航环境
- `config_sim_no_collision.yaml` - 无碰撞惩罚
- `config_sim_vascular.yaml` - 血管通道环境

在 `train_dreamer_game.py` 中修改配置：

```python
# Line 41
yaml_config = "config_sim_by_wall"  # 修改为你的配置文件名

# Line 40
num_envs = 8  # 并行环境数量（根据CPU核数调整）
```

### 关键超参数

在 `train_dreamer_game.py` 的 `config.update()` 部分（~line 50-80）：

```python
'batch_size': 16,              # 减小以节省显存
'run.train_ratio': 100,        # 训练/采样比例
'run.log_every': 200,          # 日志记录间隔（秒）
'run.save_every': 200,         # 检查点保存间隔（秒）
'jax.platform': 'gpu',         # 强制使用GPU
```

## 监控训练 / Monitor Training

### TensorBoard

```bash
tensorboard --logdir=logdir/ --port=6006
```

通过SSH端口转发访问：
```bash
# 在本地机器上运行
ssh -L 6006:localhost:6006 user@server
```

然后在浏览器打开 `http://localhost:6006`

### 训练指标

关键指标文件：
- `logdir/<experiment>/metrics.jsonl` - 所有训练指标
- `logdir/<experiment>/rate_target_reached*.csv` - 目标到达率
- `logdir/<experiment>/reward*.csv` - 奖励统计

### 检查点

检查点自动保存在：
```
logdir/<experiment>/checkpoint.ckpt
```

## 常见问题 / Troubleshooting

### 1. OOM (显存不足)

**症状**: `CUDA out of memory` 错误

**解决方案**:
```python
# 在 train_dreamer_game.py 中降低 batch_size
'batch_size': 8,  # 或更小

# 减少并行环境数
num_envs = 4  # 或更少
```

### 2. CPU瓶颈

**症状**: GPU利用率低，训练慢

**原因**: 环境采样是CPU密集型的（图像处理、碰撞检测）

**解决方案**:
- 增加 `num_envs` 以充分利用多核CPU
- 检查 `htop` 确认CPU没有被其他进程占用

### 3. 训练不收敛

**检查**:
- 查看 `metrics.jsonl` 中的 `reward` 和 `episode_success` 曲线
- 确认环境配置正确（图像路径、奖励函数）
- 尝试不同的 `yaml_config` 配置文件

### 4. DreamerV3 import 失败

参见上面"验证安装"部分的 troubleshooting。

## 文件说明 / File Reference

### 依赖文件

- `environment.yml` - Conda环境定义（仅Python解释器）
- `requirements-server.txt` - 服务器部署的干净依赖列表（**推荐使用**）
- `requirements-lock.txt` - 开发机器的完整 `pip freeze` 快照（**不要在服务器上使用**，包含ROS2和系统包）
- `requirements.txt` - 原始的简化依赖列表（已过时，使用 `requirements-server.txt` 替代）

### 训练脚本

- `scripts/train_dreamer_game.py` - DreamerV3 主训练脚本（**服务器推荐**）
- `scripts/train_dreamer_game_randomized.py` - 随机化环境变体
- `scripts/train_ppo_game.py` - PPO训练脚本
- `scripts/play_dreamer.py` - DreamerV3 评估/回放
- `scripts/play_ppo.py` - PPO 评估/回放

### 环境配置

- `scripts/config_*.yaml` - 各种环境配置文件
- `binary_images/*.png` - 仿真地图（障碍物、通道等）

## 与原始仓库的差异 / Differences from Upstream

此仓库（`Bluestone-work/worldmodel`）是从上游仓库清理后的服务器训练版本：

**主要变更**:
1. **移除大型演示文件**: `results/*.gif` (~170MB) 不包含在此仓库中，仅保留在[上游仓库](https://github.com/M-Medany/Model-Based-Reinforcement-Learning-for-Ultrasound-Driven-Autonomous-Microrobots)
2. **清理依赖**: 创建了 `requirements-server.txt`，移除了ROS2和硬件相关依赖
3. **路径修复**: 所有训练脚本使用仓库相对路径，移除了硬编码的 `/home/mahmoud/...` 路径
4. **日志目录**: 默认日志目录改为仓库内的 `logdir/`，不再使用外部路径

**保留功能**:
- 所有仿真训练代码（DreamerV3, PPO）
- 环境定义和配置
- 所有二值化地图和配置文件

**不包含** (用于实际硬件实验):
- 实时相机环境 (`ARSL_env_camera_dreamer.py` 等)
- 硬件控制工具 (Arduino, Tektronix 函数发生器)
- SAM分割模型的大型权重文件

## 性能基准 / Performance Benchmarks

基于上游论文的训练时间参考：

- **仿真环境预训练**: ~10天 (340k steps, 真实环境)
- **迁移学习到真实环境**: ~3小时收敛
- **新环境泛化**: 30分钟微调达到90%成功率

服务器训练速度取决于：
- GPU型号和显存
- CPU核数（影响环境采样）
- `num_envs` 并行度设置

## 许可证 / License

继承上游仓库的许可证。详见 [LICENSE](LICENSE)。

## 参考资源 / References

- 上游仓库: [M-Medany/Model-Based-Reinforcement-Learning-for-Ultrasound-Driven-Autonomous-Microrobots](https://github.com/M-Medany/Model-Based-Reinforcement-Learning-for-Ultrasound-Driven-Autonomous-Microrobots)
- DreamerV3: [danijar/dreamerv3](https://github.com/danijar/dreamerv3)
- 项目文档: [doc/call-chain-project-guide.zh.md](doc/call-chain-project-guide.zh.md)
- Claude Code 指导: [CLAUDE.md](CLAUDE.md)
