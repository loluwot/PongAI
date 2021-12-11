import gym
import json
import datetime as dt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DDPG
from nn_engine import PongEnv
env = DummyVecEnv([lambda: PongEnv(None)])
# model = DDPG(LnMlpPolicy, env, verbose=1)
model = PPO2.load('PPO220K')
obs = env.reset()
tot_rewards = 0
for i in range(2000):
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, done, info = env.step(action)
    print(rewards)
    tot_rewards += sum(rewards)
    # print(i)
print(tot_rewards)
# print(sum(tot_rewards))