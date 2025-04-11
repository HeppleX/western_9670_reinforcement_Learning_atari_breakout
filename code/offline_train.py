import d3rlpy
# from torch.utils.data import DataLoader

# prepare dataset (1% dataset)
dataset, env = d3rlpy.datasets.get_atari_transitions(
    'breakout',
    fraction=0.5,
    num_stack=4,
)

# prepare algorithm
cql = d3rlpy.algos.DiscreteCQLConfig(
    observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
    reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
    # compile_graph=True,
    compile_graph=False,
    # batch_size= 128,
).create(device='cuda:0')

# dataloader = DataLoader(dataset, batch_size=128, num_workers=8, shuffle=True)
# start training
cql.fit(
    dataset,
    n_steps=1000000,
    n_steps_per_epoch=100000,
    # dataloader=dataloader,
    evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env, epsilon=0.001)},
)
