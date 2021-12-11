import gym
import json
import datetime as dt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DDPG, SAC, ACKTR, A2C
from stable_baselines.common.callbacks import CheckpointCallback
from nn_engine import PongEnv
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./DDPG40K/',
                                         name_prefix='rl_model')

env = DummyVecEnv([lambda: PongEnv(None)])

model = DDPG('LnMlpPolicy', env, verbose=1, tensorboard_log='logDDPG')
model.learn(total_timesteps=40000, callback=checkpoint_callback)
model.save('DDPG40K')   
obs = env.reset()
for i in range(2000):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  print(rewards)