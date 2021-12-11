#   Modified pong engine that acts as RL environment.
#   Modifications: Andy Cai
#   Original Authors: Michael Guerzhoy and Denis Begun, 2014-2020.
#   http://www.cs.toronto.edu/~guerzhoy/
#   Email: guerzhoy at cs.toronto.edu
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version. You must credit the authors
#   for the original parts of this code.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   Parts of the code are based on T. S. Hayden Dennison's PongClone (2011)
#   http://www.pygame.org/project-PongClone-1740-3032.html



from numpy.lib.function_base import disp
import pygame, sys, time, random, os
from pygame.locals import *
import gym
from gym import spaces
import math

pygame.init()

white = [255, 255, 255]
black = [0, 0, 0]
clock = pygame.time.Clock()

class fRect:
    '''
    pygame's Rect class can only be used to represent whole integer vertices, so we create a rectangle class that can have floating point coordinates
    '''
    def __init__(self, pos, size):
        self.pos = (pos[0], pos[1])
        self.size = (size[0], size[1])
    def move(self, x, y):
        return fRect((self.pos[0]+x, self.pos[1]+y), self.size)

    def move_ip(self, x, y, move_factor = 1):
        self.pos = (self.pos[0] + x*move_factor, self.pos[1] + y*move_factor)

    def get_rect(self):
        return Rect(self.pos, self.size)

    def copy(self):
        return fRect(self.pos, self.size)

    def intersect(self, other_frect):
        # two rectangles intersect iff both x and y projections intersect
        for i in range(2):
            if self.pos[i] < other_frect.pos[i]: # projection of self begins to the left
                if other_frect.pos[i] >= self.pos[i] + self.size[i]:
                    return 0
            elif self.pos[i] > other_frect.pos[i]:
                if self.pos[i] >= other_frect.pos[i] + other_frect.size[i]:
                    return 0
        return 1#self.size > 0 and other_frect.size > 0


class Paddle:
    def __init__(self, pos, size, speed, max_angle,  facing, timeout):
        self.frect = fRect((pos[0]-size[0]/2, pos[1]-size[1]/2), size)
        self.speed = speed
        self.size = size
        self.facing = facing
        self.max_angle = max_angle
        self.timeout = timeout

    def factor_accelerate(self, factor):
        self.speed = factor*self.speed

    #change so that move requires direction input
    def move(self, enemy_frect, ball_frect, table_size):
        direction = self.move_getter(self.frect.copy(), enemy_frect.copy(), ball_frect.copy(), tuple(table_size))
        # direction = timeout(self.move_getter, (self.frect.copy(), enemy_frect.copy(), ball_frect.copy(), tuple(table_size)), {}, self.timeout)
        if direction == "up":
            self.frect.move_ip(0, -self.speed)
        elif direction == "down":
            self.frect.move_ip(0, self.speed)

        to_bottom = (self.frect.pos[1]+self.frect.size[1])-table_size[1]

        if to_bottom > 0:
            self.frect.move_ip(0, -to_bottom)
        to_top = self.frect.pos[1]
        if to_top < 0:
            self.frect.move_ip(0, -to_top)


    def get_face_pts(self):
        return ((self.frect.pos[0] + self.frect.size[0]*self.facing, self.frect.pos[1]),
                (self.frect.pos[0] + self.frect.size[0]*self.facing, self.frect.pos[1] + self.frect.size[1]-1)
                )

    def get_angle(self, y):
        center = self.frect.pos[1]+self.size[1]/2
        rel_dist_from_c = ((y-center)/self.size[1])
        rel_dist_from_c = min(0.5, rel_dist_from_c)
        rel_dist_from_c = max(-0.5, rel_dist_from_c)
        sign = 1-2*self.facing

        return sign*rel_dist_from_c*self.max_angle*math.pi/180





class Ball:
    def __init__(self, table_size, size, paddle_bounce, wall_bounce, dust_error, init_speed_mag):
        rand_ang = (.4+.4*random.random())*math.pi*(1-2*(random.random()>.5))+.5*math.pi
        # rand_ang = 0

        #rand_ang = -110*math.pi/180
        speed = (init_speed_mag*math.cos(rand_ang), init_speed_mag*math.sin(rand_ang))
        # speed = [1.1809523809528173, 5.881451385746569]

        pos = (table_size[0]/2, table_size[1]/2)
        # pos = (394.10595080432444 + size[0]/2, 33.434349153590645 + size[1]/2)
        # pos = (299.5837288565761 + size[0]/2, 262.09863291472834 + size[1]/2)
        #pos = (table_size[0]/2 - 181, table_size[1]/2 - 105)
        self.frect = fRect((pos[0]-size[0]/2, pos[1]-size[1]/2), size)
        self.speed = speed
        self.size = size
        self.paddle_bounce = paddle_bounce
        self.wall_bounce = wall_bounce
        self.dust_error = dust_error
        self.init_speed_mag = init_speed_mag
        self.prev_bounce = None

    def get_center(self):
        return (self.frect.pos[0] + .5*self.frect.size[0], self.frect.pos[1] + .5*self.frect.size[1])


    def get_speed_mag(self):
        return math.sqrt(self.speed[0]**2+self.speed[1]**2)

    def factor_accelerate(self, factor):
        self.speed = (factor*self.speed[0], factor*self.speed[1])



    def move(self, paddles, table_size, move_factor):
        moved = 0
        walls_Rects = [Rect((-100, -100), (table_size[0]+200, 100)),
                       Rect((-100, table_size[1]), (table_size[0]+200, 100))]

        for wall_rect in walls_Rects:
            if self.frect.get_rect().colliderect(wall_rect):
                c = 0
                #print "in wall. speed: ", self.speed
                while self.frect.get_rect().colliderect(wall_rect):
                    self.frect.move_ip(-.1*self.speed[0], -.1*self.speed[1], move_factor)
                    c += 1 # this basically tells us how far the ball has traveled into the wall
                r1 = 1+2*(random.random()-.5)*self.dust_error
                r2 = 1+2*(random.random()-.5)*self.dust_error

                self.speed = (self.wall_bounce*self.speed[0]*r1, -self.wall_bounce*self.speed[1]*r2)
                while c > 0 or self.frect.get_rect().colliderect(wall_rect):
                    self.frect.move_ip(.1*self.speed[0], .1*self.speed[1], move_factor)
                    c -= 1 # move by roughly the same amount as the ball had traveled into the wall
                moved = 1
                #print "out of wall, position, speed: ", self.frect.pos, self.speed

        for paddle in paddles:
            if self.frect.intersect(paddle.frect):
                if (paddle.facing == 1 and self.get_center()[0] < paddle.frect.pos[0] + paddle.frect.size[0]/2) or \
                (paddle.facing == 0 and self.get_center()[0] > paddle.frect.pos[0] + paddle.frect.size[0]/2):
                    continue
                
                c = 0
                
                while self.frect.intersect(paddle.frect) and not self.frect.get_rect().colliderect(walls_Rects[0]) and not self.frect.get_rect().colliderect(walls_Rects[1]):
                    self.frect.move_ip(-.1*self.speed[0], -.1*self.speed[1], move_factor)
                    
                    c += 1
                theta = paddle.get_angle(self.frect.pos[1]+.5*self.frect.size[1])
                

                v = self.speed

                v = [math.cos(theta)*v[0]-math.sin(theta)*v[1],
                             math.sin(theta)*v[0]+math.cos(theta)*v[1]]

                v[0] = -v[0]

                v = [math.cos(-theta)*v[0]-math.sin(-theta)*v[1],
                              math.cos(-theta)*v[1]+math.sin(-theta)*v[0]]


                # Bona fide hack: enforce a lower bound on horizontal speed and disallow back reflection
                if  v[0]*(2*paddle.facing-1) < 1: # ball is not traveling (a) away from paddle (b) at a sufficient speed
                    v[1] = (v[1]/abs(v[1]))*math.sqrt(v[0]**2 + v[1]**2 - 1) # transform y velocity so as to maintain the speed
                    v[0] = (2*paddle.facing-1) # note that minimal horiz speed will be lower than we're used to, where it was 0.95 prior to increase by *1.2

                #a bit hacky, prevent multiple bounces from accelerating
                #the ball too much
                if not paddle is self.prev_bounce:
                    self.speed = (v[0]*self.paddle_bounce, v[1]*self.paddle_bounce)
                else:
                    self.speed = (v[0], v[1])
                self.prev_bounce = paddle
                #print "transformed speed: ", self.speed

                while c > 0 or self.frect.intersect(paddle.frect):
                    #print "move_ip()"
                    self.frect.move_ip(.1*self.speed[0], .1*self.speed[1], move_factor)
                    #print "ball position forward trace: ", self.frect.pos
                    c -= 1
                #print "pos final: (" + str(self.frect.pos[0]) + "," + str(self.frect.pos[1]) + ")"
                #print "speed x y: ", self.speed[0], self.speed[1]

                moved = 1
                #print "out of paddle, speed: ", self.speed

        # if we didn't take care of not driving the ball into a wall by backtracing above it could have happened that
        # we would end up inside the wall here due to the way we do paddle bounces
        # this happens because we backtrace (c++) using incoming velocity, but correct post-factum (c--) using new velocity
        # the velocity would then be transformed by a wall hit, and the ball would end up on the dark side of the wall

        if not moved:
            self.frect.move_ip(self.speed[0], self.speed[1], move_factor)
            #print "moving "
        #print "poition: ", self.frect.pos



def check_point(score, ball, table_size):
    if ball.frect.pos[0]+ball.size[0]/2 < 0:
        score[1] += 1
        # ball = Ball(table_size, ball.size, ball.paddle_bounce, ball.wall_bounce, ball.dust_error, ball.init_speed_mag)
        return (ball, score)
    elif ball.frect.pos[0]+ball.size[0]/2 >= table_size[0]:
        # ball = Ball(table_size, ball.size, ball.paddle_bounce, ball.wall_bounce, ball.dust_error, ball.init_speed_mag)
        score[0] += 1
        return (ball, score)

    return (ball, score)



from utils import *
import better_ai
# import adjustable
from temp_adjustable_class import Adjustable
from itertools import chain
import numpy as np
from functools import partial
from random import randint
RAND = 0
RENDER = False
VERBOSE = True
class PongEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def own_ai(self, paddle_frect, other_paddle_frect, ball_frect, table_size):
        return self.bot1.pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size, self.action)
    def __init__(self, config) -> None:
        super(PongEnv, self).__init__()
        self.table_size = (440, 280)
        self.paddle_size = (10, 70)
        self.ball_size = (15, 15)
        self.paddle_speed = 1
        self.max_angle = 45
        self.paddle_bounce = 1.2
        self.wall_bounce = 1.00
        self.dust_error = 0
        self.init_speed_mag = 2
        self.timeout1 = 0.0001
        self.lastNStates = []
        self.NUM_PREV = 4
        self.observation_space = spaces.Box(low = 0, high = 1, shape = (4, 5), dtype=np.float32)
        self.action_space = spaces.Box(low = 0, high = 1, shape = (1, ), dtype=np.float32)
        self.action = None
        self.bot1 = Adjustable()
        if RENDER:
            self.screen = pygame.display.set_mode(self.table_size)
            self.clock = pygame.time.Clock()
         #-1 is other to your, 1 is your to other
        # turn_wait_rate = 3
        # score_to_win = 20
        self.paddles = [Paddle((20, self.table_size[1]/2), self.paddle_size, self.paddle_speed, self.max_angle,  1, self.timeout1),
                    Paddle((self.table_size[0]-20, self.table_size[1]/2), self.paddle_size, self.paddle_speed, self.max_angle, 0, self.timeout1)]
        self.ball = Ball(self.table_size, self.ball_size, self.paddle_bounce, self.wall_bounce, self.dust_error, self.init_speed_mag)
        
        #ADDED RANDOM SIDE TRAINING
        self.random_side = randint(0,1)*RAND
        # self.random_side = RAND
        self.cur_direction = -1 + 2*self.random_side
        self.paddles[self.random_side].move_getter = self.own_ai#partial(adjustable.pong_ai, ts=self.action)
        # paddle 0 would be the testing mover
        self.paddles[(self.random_side + 1) % 2].move_getter = better_ai.pong_ai

    cur_action = None
    def game_step(self):
        self.paddles[0].move(self.paddles[1].frect, self.ball.frect, self.table_size)
        self.paddles[1].move(self.paddles[0].frect, self.ball.frect, self.table_size)
        inv_move_factor = int((self.ball.speed[0]**2+self.ball.speed[1]**2)**.5)
        if inv_move_factor > 0:
            for i in range(inv_move_factor):
                self.ball.move(self.paddles, self.table_size, 1./inv_move_factor)
        else:
            self.ball.move(self.paddles, self.table_size, 1)
        self.lastNStates.append([self.paddles[self.random_side].frect.pos[1]/self.table_size[1], self.paddles[(self.random_side + 1) % 2].frect.pos[1]/self.table_size[1]] + [self.ball.frect.pos[i]/self.table_size[i] for i in range(2)] + [self.random_side])
        if RENDER:
            self.render()
        # print(self.lastNStates)
    def get_state(self):
        obs = self.lastNStates[-min(len(self.lastNStates), 4):]
        
        obs = obs + [[0,0,0,0,0] for i in range(max(0, self.NUM_PREV - len(obs)))]
        obs = np.array(obs)
        obs = obs.clip(0, 1)
        # obs = chain.from_iterable(self.lastNStates)
        # print(np.array(obs))

        return np.array(obs)

    def step(self, action):
        self.action = action
        # adjustable.ts = action
        THRESH_LINE = 10
        done = False
        pass_count = 0
        #self
        paddle_x1 = padx(self.paddles[self.random_side].frect.pos[0], self.table_size)
        #other
        paddle_x2 = padx(self.paddles[(self.random_side+1) % 2].frect.pos[0], self.table_size)
        reward = 0
        while reward == 0 and pass_count < 2:
            self.game_step()
            if self.cur_direction*self.ball.frect.pos[0] < self.cur_direction*(paddle_x2 + THRESH_LINE*(self.random_side*2 - 1)):
                pass_count += 1
                # print('BOUNCE')
                self.cur_direction = -self.cur_direction
            
            if self.ball.frect.pos[0] >= self.table_size[0]:
                reward = 1*(1 - 2*self.random_side)
            if self.ball.frect.pos[0] < 0:
                reward = -1*(1 - 2*self.random_side)
            # print(pass_count)
        # print('CYCLE ----')
        done = (reward in [1, -1])
        obs = self.get_state()
        # print(reward)
        return obs, reward, done, {}

    def reset(self):
        self.table_size = (440, 280)
        self.paddle_size = (10, 70)
        self.ball_size = (15, 15)
        self.paddle_speed = 1
        self.max_angle = 45
        self.paddle_bounce = 1.2
        self.wall_bounce = 1.00
        self.dust_error = 0
        self.init_speed_mag = 2
        self.timeout1 = 0.0001
        self.lastNStates = []
        self.action = None
        # self.cur_direction = -1
        # turn_wait_rate = 3
        # score_to_win = 20
        self.paddles = [Paddle((20, self.table_size[1]/2), self.paddle_size, self.paddle_speed, self.max_angle,  1, self.timeout1),
                    Paddle((self.table_size[0]-20, self.table_size[1]/2), self.paddle_size, self.paddle_speed, self.max_angle, 0, self.timeout1)]
        self.ball = Ball(self.table_size, self.ball_size, self.paddle_bounce, self.wall_bounce, self.dust_error, self.init_speed_mag)
        self.random_side = randint(0,1)*RAND
        # self.random_side = RAND
        self.cur_direction = -1 + 2*self.random_side
        self.paddles[self.random_side].move_getter = self.own_ai#partial(adjustable.pong_ai, ts=self.action)
        # paddle 0 would be the testing mover
        self.paddles[(self.random_side + 1) % 2].move_getter = better_ai.pong_ai
        self.lastNStates.append([self.paddles[self.random_side].frect.pos[1]/self.table_size[1], self.paddles[(self.random_side + 1) % 2].frect.pos[1]/self.table_size[1]] + [self.ball.frect.pos[i]/self.table_size[i] for i in range(2)] + [self.random_side])
        if VERBOSE:
            print(f'{self.random_side}--------------------------x')
        return self.get_state()
    
    def render(self, mode='human', close=False):
        self.screen.fill(black)

        pygame.draw.rect(self.screen, white, self.paddles[0].frect.get_rect())
        pygame.draw.rect(self.screen, pygame.Color(255,0,0,1), self.paddles[1].frect.get_rect())
        pygame.draw.circle(self.screen, white, (int(self.ball.get_center()[0]), int(self.ball.get_center()[1])),int(self.ball.frect.size[0]/2), 0)
        pygame.draw.line(self.screen, white, [self.screen.get_width()/2, 0], [self.screen.get_width()/2, self.screen.get_height()])
        pygame.display.flip()
        self.clock.tick(80)


