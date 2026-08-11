# 3D 未知血管中的群体导航与破栓：第一轮研究协议

## 结论先行

当前项目不应该直接把 ROS2/Gazebo TurtleBot3 环境改成血管环境，也不应该把 UAV 的独立三维速度命令直接解释成微机器人控制。第一阶段采用下面的桥接层：

```text
DGR-VDS 的局部图策略
        ↓  3D 局部观测、图邻居、时序记忆
Vascular3DSwarmEnv 的共享执行器
        ↓  一个共享的 [vx, vy, vz, morphology] 命令
3D 分叉血管、流场、接触溶栓代理、碰壁约束
```

这保留了论文真正有价值的难点（未知环境、部分观测、历史上下文、动态扰动），同时不虚构每个微机器人拥有独立推进器。

## 文献与代码对齐

| 来源 | 可迁移组件 | 不能直接迁移的部分 |
|---|---|---|
| An et al., *Nature Machine Intelligence* 8, 955--968 (2026), DOI [10.1038/s42256-026-01252-6](https://doi.org/10.1038/s42256-026-01252-6) | 部分观测、temporal attention、环境/感知/执行器/动力学多级随机化、sim-to-real 评测 | 论文的 Turbo 细节、传感器和磁执行器标定不能从摘要或 DGR 代码推断 |
| `DGR_VDS` commit `c150c21` | GAT、MAPPO、LSTM、局部地图、ORCA/DWA/BC 评测协议 | ROS2/Gazebo TurtleBot3 的二维地面动力学、独立机器人动作 |
| 本项目 `MagneticSwarm25DEnv` / `CanonicalFieldSwarm25DEnv` | 分层拓扑、局部占据图、线圈到场的约束分配、血栓质量代理 | 当前 2.5D 投影不是连续三维血管或 CFD |

## 环境假设与边界

`environments/vascular_3d_swarm_env.py` 是算法 benchmark：血管由采样的三维中心线和半径定义；机器人是过阻尼粒子；流场沿中心线切向；离开管腔的候选位置投影回最近中心线；接触血栓后按距离衰减清除质量。

它明确不声称预测：真实磁偶极耦合、血液 CFD、血栓材料破坏、生物相容性或临床溶栓剂量。后续若要做物理结论，必须替换为实验标定或高保真多物理场模型。

## 第一轮可复现实验

```bash
python scripts/evaluate_vascular_3d_baselines.py \
  --scenario bifurcation --episodes 20 \
  --output autoresearch/vascular-3d-baselines
```

固定报告：成功率、清除率、首次接触步数、剩余质量、碰壁率、路径长度、已知拓扑比例。不能只报告训练 reward。

建议场景顺序：`straight` → `bifurcation` → `anastomosis` → `stenotic`。训练/验证/测试按场景和随机种子拆分；测试集不得被奖励调参使用。

## 算法路线

1. **规则基线**：`flow_only`、随机、局部贪心；验证环境难度和可达性。
2. **结构消融**：MLP-MAPPO、GAT-MAPPO、GAT-LSTM-MAPPO；输入始终为局部观测，额外比较 oracle 全局观测上界。
3. **物理约束消融**：共享动作、粒子独立动作（仅作不现实的上界）、共享动作 + 可微/最小二乘投影。主结果必须排除独立动作作弊。
4. **Turbo 风格随机化**：管径、分叉角、流速、执行器延迟、观测丢帧、定位噪声和血栓位置随机化；逐项报告，而不是只给一个混合随机化结果。
5. **sim-to-real 预备**：先在 `vascular_3d_swarm_env.py` 记录动作、局部体素、质量和接触事件；再用真实超声/磁场标定替换动力学，不提前宣称迁移成功。

## 关于 UAV 桥接的判断

UAV/空中群体算法适合作为“连续三维局部导航 + 图策略 + 时序记忆”的方法来源，尤其可参考 3D clutter、动态障碍和 swarm coordination；它不适合作为物理仿真环境本身。正确桥接是迁移策略和评测协议，不迁移 UAV 的动力学、推进器或独立动作权限。

## 自动科研的验收门槛

- 环境在固定 seed 下 reset 可复现；所有 observation/action 通过 Gymnasium space 校验。
- 所有机器人始终在管腔内；共享执行器维度保持为 4，不出现每粒子独立执行器。
- 至少 5 个 seed、每个测试场景至少 100 个 episode 后再作算法结论。
- 主指标是 success/removal/collision/path，不以单一加权分数替代原始指标。
- 所有实验写入 `autoresearch/`，记录命令、配置、git commit 和依赖版本；不自动 push、发布或部署。

