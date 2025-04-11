import gym
import torch
import numpy as np
from collections import deque
import d3rlpy


num_episodes=10

# 加载训练好的模型
cql=d3rlpy.load_learnable("model_3000000.d3", device="cpu")

# 创建环境
env = gym.make("BreakoutNoFrameskip-v4",render_mode="rgb_array")
#env = d3rlpy.envs.Atari(env, num_stack=4, is_eval=True)
env = d3rlpy.envs.Atari(env, num_stack=4)

all_scores = []
for episode in range(num_episodes):
    observation,info = env.reset()
    done = True
    total_reward = 0

    while info['lives'] > 0:
        env.render()

        action = cql.predict(np.expand_dims(observation,axis=0))[0]
        if done:
            action = np.uint8(1)

        observation, reward, done, truncated, info = env.step(action)

        total_reward += reward

    all_scores.append(total_reward)
#    print(f"Episode {episode + 1}: Total Score = {total_reward}")

# 计算平均分
average_score = np.mean(all_scores)
min_score = np.min(all_scores)
max_score = np.max(all_scores)
print(f"\nAverage Score over {num_episodes} Episodes: {min_score},{average_score},{max_score}")
env.close()
