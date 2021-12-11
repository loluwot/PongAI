import time, math, sys
#initial goal
from prediction_engine import predict
from utils import *
best_loc = 140 - 36
goal = best_loc

#global variables---------
default_goal = 140
MAX_VELS = 1
THRESHOLD = 1.2
max_angle = 45*math.pi/180
count = 0
idx = 0

class Adjustable:
    def __init__(self) -> None:
        #changeables--------
        self.vels = []
        self.continue_aiming = True
        self.prediction = None
        self.last_pos = None
        self.ball_vel = None
        self.ticks_prediction = 0
        self.keep_predicting = False
        self.last_collided = None
        self.tot_bounces = None
        self.predicted_vel = None
        self.prediction_time = None
        self.tot_prediction_time = None
        self.region = None
        self.target = None
        self.temp_counter = 10
        self.goal = default_goal
        pass


    
    # ts = None

    #debug metrics--------
    accuracy = 0
    accuracy_count = 0
    error_count = 0

    def pong_ai(self, paddle_frect, other_paddle_frect, ball_frect, table_size, ts):
        try:
            center_self = center(paddle_frect)
            center_other = center(other_paddle_frect)
            ball_center = center(ball_frect)
            init_time = time.time()
            if ball_center[0] == table_size[0]/2 and ball_center[1] == table_size[1]/2:
                self.ball_vel = None
                self.prediction = None
                self.last_pos = None
                self.ticks_prediction = 10
                self.keep_predicting = True
                self.last_collided = None
                self.predicted_vel = None
                self.prediction_time = None
                self.tot_bounces = None
                self.count = 0
                self.tot_prediction_time = None
                self.region = 0
                self.idx = 0
                self.target = None
                self.temp_counter = 10

            

            if self.last_pos:
                # print('got pos')
                vel = [ball_frect.pos[i] - self.last_pos[i] for i in range(2)]
                self.ball_vel = vel[::]
                self.vels.append(self.ball_vel)
                self.vels = self.vels[max(0, len(self.vels) - 2):]

            self.last_pos = ball_frect.pos
            # print(self.last_pos)
            paddle_x1 = padx(paddle_frect.pos[0], table_size)
            paddle_x2 = padx(other_paddle_frect.pos[0], table_size)
            # print('ACTUAL VELOCITY:', ball_vel, predicted_vel)
            facing = paddle_frect.pos[0] < table_size[0]/2
            facing_sign = (2*facing - 1)
            if abs(ball_frect.pos[0] - paddle_x1) < (0 if self.ball_vel is None else mag(self.ball_vel)*THRESHOLD) and self.last_collided != 0:
                # print('COLLISION')
                # accuracy = max(abs(prediction - ball_center[1]), accuracy)
                self.keep_predicting = False
                self.ticks_prediction = math.inf
                self.goal = default_goal
                self.last_collided = 0
                self.continue_aiming = False
                self.temp_counter = 10

            if abs(ball_frect.pos[0] - paddle_x2) < (0 if self.ball_vel is None else mag(self.ball_vel)*THRESHOLD) and self.last_collided != 1:
                self.ticks_prediction = 2
                self.keep_predicting = True
                self.continue_aiming = True
                self.last_collided = 1
                self.prediction = None

            if self.ticks_prediction > 0:
                # print(self.ticks_prediction, self.vels)
                self.ticks_prediction -= 1

            elif len(self.vels) == 2 and self.vels[-1] == self.vels[0] and sign(self.vels[0][0]) == -facing_sign:
                # print('TS', ts)
                # last_pred = prediction
                # last_pred_time = prediction_time
                self.prediction, self.prediction_time, self.n_bounces, self.bounce_time = predict(ball_frect.pos, self.ball_vel, table_size, padx(paddle_frect.pos[0], table_size))


                prediction_sign = 1 - 2*(self.n_bounces % 2)
                predicted_vel = [-self.ball_vel[0], prediction_sign*self.ball_vel[1]]
                #targeting section
                max_dist = (self.prediction_time)/paddle_frect.size[1]
                leeway = min((abs(self.prediction - center_self[1]) - paddle_frect.size[1]/2 - 15/2)/paddle_frect.size[1], 1)
                leeway = (max_dist - leeway)
                dv = sign(self.prediction - center_self[1])
                bounds = None
                if leeway >= 0:
                    if leeway <= 1:
                        bound = dv*0.5
                        bounds = [bound*max_angle*facing_sign, (bound - dv*leeway)*max_angle*facing_sign]
                    else:
                        bound = (self.prediction - center_self[1])/paddle_frect.size[1]
                        bounds = [max(-0.5, bound - max_dist)*max_angle*facing_sign, min(0.5, bound + max_dist)*max_angle*facing_sign]
                else:
                    # print('HOPELESS?')
                    self.goal = self.prediction
                    self.ticks_prediction = 2
                    return
                bounds.sort()
                
                # region = 0
                if self.continue_aiming:
                    modulus = table_size[1] - 15
                    # idx = center_other[1] > table_size[1]/2
                    current = [-self.ball_vel[0], prediction_sign*self.ball_vel[1]]
                    dx = padx(other_paddle_frect.pos[0], table_size) - padx(paddle_frect.pos[0], table_size)
                    for bounce in [1, -1]:
                        for iii in [0, 1, -1]:
                            self.target = [dx, ((0 if ts is None else ts*table_size[1])*bounce + iii*modulus*2) - self.prediction]
                            self.target_region = ang(current, self.target)/2*facing_sign
                            if bounds[0] <= self.target_region <= bounds[1]:
                                self.region = (self.target_region/max_angle)*paddle_frect.size[1]
                                break
                        else:
                            continue
                        break
                    if self.region != 0:
                        # print('AIMING UP' if idx == 1 else 'AIMING DOWN')
                        pass
                self.goal = self.prediction - self.region
                self.ticks_prediction = math.inf if len(self.bounce_time) < 1 else self.bounce_time[0]
                self.continue_aiming = False
                # keep_predicting = False
                # count += 1

            # print('TOOK:', time.time() - init_time)
            if center_self[1] < self.goal:
                return 'down'
            elif center_self[1] > self.goal:
                return 'up'
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            # print('ERROR1')
            self.error_count += 1
            print('exception', e)
            # input()
            return
        # pass