import gym
import d3rlpy


env = gym.make('BreakoutNoFrameskip-v4')
env = d3rlpy.envs.Atari(env, num_stack=4)

eval_env = gym.make("BreakoutNoFrameskip-v4")
eval_env = d3rlpy.envs.Atari(eval_env, num_stack=4, is_eval=True)

dqn = d3rlpy.algos.DQNConfig(
    learning_rate=1e-4,
    target_update_interval=10000,
).create(device='cuda:0')

# setup explorer
explorer = d3rlpy.algos.LinearDecayEpsilonGreedy(start_epsilon=1.0,
        end_epsilon=0.1,
        duration=1000000)

# setup replay buffer
buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=500000, env=env)

# start training
dqn.fit_online(
    env,
    buffer,
    explorer,
    eval_env=eval_env,
    n_steps=3000000,
    n_steps_per_epoch=100000,
    update_interval=4,
)
