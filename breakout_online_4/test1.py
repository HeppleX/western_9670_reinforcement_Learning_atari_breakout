import gym
import torch
import numpy as np
from collections import deque
import d3rlpy


num_episodes=100
# 创建环境
env = gym.make("BreakoutNoFrameskip-v4",render_mode="rgb_array")
env = d3rlpy.envs.Atari(env, num_stack=4)

for i in range(58,59):
    model_name="model_{}00000.d3".format(i)
    # print(model_name)
    # 加载训练好的模型
    cql=d3rlpy.load_learnable(model_name, device="cpu")
    
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

    # 计算平均分
    average_score = np.mean(all_scores)
    lower_bound, upper_bound = np.percentile(all_scores, [2.5, 97.5])
    # print(f"\nAverage Score over {i*100000} Episodes: {lower_bound} {average_score} {upper_bound}")
    print(f"{i*100000} {lower_bound} {average_score} {upper_bound}")
env.close()
