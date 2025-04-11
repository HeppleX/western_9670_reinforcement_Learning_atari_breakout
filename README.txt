🧠 Background
Atari games—especially the classic Breakout—offer a simple control interface but require complex strategies. These games have become benchmark environments in Reinforcement Learning (RL) research due to their high-dimensional input spaces and sequential decision-making challenges.

🎯 Objectives
This project aims to:

Train an agent to achieve high scores in the Atari Breakout game using RL techniques.

Compare the performance of Online RL (DQN) and Offline RL (CQL) approaches.

Explore ways to improve training efficiency and policy stability.

🔧 Key Technologies & Methods
✅ Online RL: DQN (Deep Q-Network)
Environment: BreakoutNoFrameskip-v4 from OpenAI Gym (ALE)

Input: 84×84 grayscale images with a 4-frame stack

Action Space: NOOP, FIRE, LEFT, RIGHT

Techniques:

Epsilon-greedy exploration

Experience replay buffer

Target network updated every 10K steps

Training Configuration:

Total of 10 million steps

Network updated every 4 steps

Evaluation performed periodically

Metrics include average reward and training loss

✅ Offline RL: CQL (Conservative Q-Learning)
Dataset: Google’s pre-collected Atari Replay Dataset (50% of Breakout transitions)

Key Challenge: Out-of-distribution (OOD) state-action pairs may lead to poor policies or instability

Solution:

CQL adds a conservative penalty to limit overestimation of Q-values:

Training Setup:

Learning rate: 6.25e-5

Penalty coefficient: α = 1.0

Reward clipping between -1.0 and 1.0

Target network updated every 10K steps

Total of 5 million training steps (every 100K steps = 1 epoch)

📊 Results
DQN (Online RL) produced more stable and higher average scores after longer training.

CQL (Offline RL) showed faster training but faced challenges due to dataset bias and reward fluctuation.

The performance gap between online and offline approaches highlights the trade-off between data efficiency and stability.

✅ Conclusions & Future Work
DQN demonstrates strong performance when trained with sufficient environment interactions, though at a high computational cost.

CQL enables learning without live interaction, but managing distribution mismatch is crucial.

Future directions:

Implement Prioritized Experience Replay (PER) to accelerate DQN training.

Fine-tune the CQL penalty factor to reduce reward variance.

Combine offline pretraining with online fine-tuning (e.g., using algorithms like AWAC or AWAS) for better results.


🧠 项目背景
Atari 游戏，尤其是经典的 Breakout（打砖块），以其简单的操作和复杂的策略性，成为强化学习（Reinforcement Learning, RL）研究中的经典测试平台。通过与环境交互进行学习，这些游戏为 RL 算法提供了高噪声、高维状态空间的挑战场景。

🎯 项目目标
本项目旨在：

使用强化学习算法训练智能体在 Breakout 游戏中获得高分；

比较 在线 RL（DQN） 与 离线 RL（CQL） 两种方法在该任务中的性能差异；

探索如何优化训练效率和稳定性。

🔧 主要技术与方法
✅ Online RL：DQN（Deep Q-Network）
环境：Atari BreakoutNoFrameskip-v4（来自 Gym）

输入：84x84 灰度图像的 4 帧堆叠

动作空间：NOOP、FIRE、LEFT、RIGHT

关键技术：

Epsilon-Greedy 策略进行探索

使用经验回放（Replay Buffer）

每 10K 步更新一次目标网络

训练细节：

总训练步数为 10M

每 4 步更新一次网络

训练过程记录 reward 和 loss 变化

✅ Offline RL：CQL（Conservative Q-Learning）
数据来源：Google 开放的 Atari Replay Datasets（使用 50% 的 Breakout 数据）

问题挑战：离线数据存在分布外状态-动作对（Out-of-Distribution），可能导致学习失败

技术解决：

在 DQN 损失基础上添加 conservative 罚项，防止过高 Q 值带来的策略偏差

训练设置：

学习率 6.25e-5，惩罚因子 α=1.0

奖励剪裁 [-1.0, 1.0]

每 100K 步作为一个 epoch，总共 5M 步

📊 实验结果
在线 RL（DQN）表现更稳定，在训练后期能获得较高平均得分；

离线 RL（CQL）受限于数据质量和分布，虽然训练效率高，但存在奖励波动和策略偏移问题；

两者在训练曲线、策略稳定性方面存在显著差异。

✅ 结论与展望
DQN 在充分训练下能够学习出高性能策略，但训练周期长；

CQL 可在无需与环境交互的条件下快速训练策略，但需应对分布偏移问题；

未来计划：

引入优先经验回放（PER）加速 DQN 收敛；

调整 CQL 的惩罚因子缓解奖励不稳定；

探索 离线预训练 + 在线微调 的混合策略（如 AWAC、AWAS）。
