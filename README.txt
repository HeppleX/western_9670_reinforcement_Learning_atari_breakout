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
