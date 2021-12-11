from stable_baselines import PPO2, DDPG, SAC
import numpy as np
import adjustable
from utils import *

model = DDPG.load('TRAINED MODEL LOCATION')
print('LOADED')
lastNStates = []
reward = 0
pass_count = 0
cur_direction = -1
THRESH_LINE = 10
NUM_PREV = 4
action = [0.5]
def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
    global lastNStates, reward, pass_count, cur_direction, action
    cur_side = int(paddle_frect.pos[0] > table_size[0]/2)
    if ball_frect.pos[0] == table_size[0]/2 and ball_frect.pos[1] == table_size[1]/2:
        lastNStates = []
        reward = 0
        pass_count = 0
        cur_direction = (cur_side*2 - 1)
    lastNStates.append([paddle_frect.pos[1]/table_size[1], other_paddle_frect.pos[1]/table_size[1]] + [ball_frect.pos[i]/table_size[i] for i in range(2)] + [cur_side])

    obs = lastNStates[-min(len(lastNStates), NUM_PREV):]
    obs += [[0,0,0,0,0] for i in range(max(0, NUM_PREV - len(obs)))]
    obs = np.array(obs)
    # obs = [paddle_frect.pos[1]/table_size[1], other_paddle_frect.pos[1]/table_size[1]] + [ball_frect.pos[i]/table_size[i] for i in range(2)]
    
    paddle_x1 = padx(paddle_frect.pos[0], table_size)
    paddle_x2 = padx(other_paddle_frect.pos[0], table_size)

    
    if reward != 0 or pass_count >= 2:
        print('WHAT', reward, pass_count)
        pass_count = 0
        print(obs)
        action, _states = model.predict(obs)
        reward = 0
        print(action)
    if cur_direction*ball_frect.pos[0] < cur_direction*(paddle_x2 + THRESH_LINE*(cur_side*2 - 1)):
        pass_count += 1
        cur_direction = -cur_direction
    if ball_frect.pos[0] >= table_size[0]:
        reward = 1
    if ball_frect.pos[0] < 0:
        reward = -1
    # print(action)
    # adjustable.ts = action[0]
    return adjustable.pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size, action[0])
    # pass