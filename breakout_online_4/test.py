import gym
import torch
import numpy as np
from collections import deque
import d3rlpy


# 加载训练好的模型
cql = d3rlpy.load_learnable("model_5600000.d3", device="cpu")
#cql = d3rlpy.load_learnable("model_7800000.d3", device="cpu")
#cql = d3rlpy.load_learnable("model_10000000.d3", device="cpu")

#print(cql._impl.modules)
#print(cql._impl.observation_shape)
#print(cql._impl.action_size)
#print(cql._config)


# 创建环境
env = gym.make("BreakoutNoFrameskip-v4",render_mode="human")
env = d3rlpy.envs.Atari(env, num_stack=4)

observation,info = env.reset()

done = True
while info['lives'] > 0:
    env.render()

    action = cql.predict(np.expand_dims(observation,axis=0))[0]
    if done:
        action = np.uint8(1)

    observation, reward, done, truncated, info = env.step(action)
env.close()
