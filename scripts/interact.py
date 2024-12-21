import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from go2_gym.envs import *
from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.go2.go2_config import config_go2
from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm
import pygame

class TextPrint:
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 30)

    def print(self, screen, textString):
        textList = textString.split("\n")
        for string in textList:
            textBitmap = self.font.render(string, True, (255, 255, 255))
            screen.blit(textBitmap, [self.x, self.y])
            self.y += self.line_height
        self.y += self.new_line_height


    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 30
        self.new_line_height = 40

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False):
    dirs = glob.glob(f"./runs/{label}/*")
    logdir = sorted(dirs)[0]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 10
    Cfg.terrain.num_cols = 10
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    # default control_typw is "actuator_net", you can also switch it to "P" to enable joint PD control
    Cfg.control.control_type = "actuator_net" 
    Cfg.asset.flip_visual_attachments = True


    from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper
    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go2_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy


def interact_go2(headless=False):
    from ml_logger import logger

    from pathlib import Path
    from go2_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    pygame.display.set_caption('Unitree GO-2 Controller')
    pygame.init()
    screen_size = 500
    screen = pygame.display.set_mode((screen_size, screen_size))
    
    # Clock for setting game frame rate
    clock = pygame.time.Clock()
    textPrint = TextPrint()
    if (pygame.joystick.get_count() != 1):
        raise RuntimeError("Please connect exactly one controller")
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
        
    # label = "gait-conditioned-agility/pretrain-v0/train"
    # label = "gait-conditioned-agility/pretrain-go2/train"
    label = "gait-conditioned-agility/2024-12-10/train"

    # Start Sim
    env, policy = load_env(label, headless=headless)

    # Define gait modes
    gaits = {
        "pronking": [0, 0, 0],
        "trotting": [0.5, 0, 0],
        "bounding": [0, 0.5, 0],
        "pacing": [0, 0, 0.5],
    }

    # Default commands
    x_vel_cmd = 0.0
    y_vel_cmd = 0.0
    yaw_vel_cmd = 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["pronking"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    # Initialize environment observation
    obs = env.reset()
    xbox = True

    modes1 = ["A", "B", "C"] # A: Body Height | B: Lateral Velocity | C: Stance Width
    modes2 = ["D", "E", "F"] # D: Footswing Height | E: Step frequency | F: Body Pitch
    descrip1 = ["Body Height", "Lateral Velocity", "Stance Width"]
    descrip2 = ["Footswing Height", "Step frequency", "Body Pitch"]
    Mode1 = 1
    Mode2 = 0

    while xbox:
        # Drawing setup
        screen.fill((0, 102, 204))
        textPrint.reset()

        name = joystick.get_name()
        textPrint.print(screen, "Welcome! remember to make this the active \nwindow when you wish to use the remote")
        textPrint.print(screen, "Controller detected: {}".format(name) )
        textPrint.print(screen, "X velocity: {}".format(x_vel_cmd) )
        textPrint.print(screen, "Y velocity: {}".format(y_vel_cmd) )
        textPrint.print(screen, "Yaw command: {}".format(yaw_vel_cmd) )
        textPrint.print(screen, "Mode 1: {} - {}".format(modes1[Mode1], descrip1[Mode1]) )
        textPrint.print(screen, "Mode 2: {} - {}".format(modes2[Mode2], descrip2[Mode2]) )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Handle quit event
                xbox = True

            if event.type == pygame.JOYAXISMOTION:
                if event.axis == 1:  # Left vertical stick (forward/backward control)
                    x_vel_cmd = -event.value
                    print(f"x_velocity command: {x_vel_cmd}")
                
                elif event.axis == 0: # Left Horizontal stick
                    if Mode1 == 0:
                        body_height_cmd = -event.value / 20. #[-0.05, 0.05]
                        print(f"Body height command: {body_height_cmd}")
                    
                    elif Mode1 == 1:
                        y_vel_cmd = -event.value
                        print(f"y_velocity command: {y_vel_cmd}")
                    
                    elif Mode1 == 2:
                        stance_width_cmd = -event.value / 10. + 0.25
                        print(f"Stance_width command: {stance_width_cmd}")

                elif event.axis == 4: # Right Horizontal stick (Yaw Velocity)
                    yaw_vel_cmd = -event.value
                    print(f"yaw command: {yaw_vel_cmd}")

                elif event.axis == 3: # Right vertical stick
                    if Mode2 == 0:
                        footswing_height_cmd = -event.value / 10. + 0.08
                        print(f"Foot Swing Height command: {footswing_height_cmd}")
                    
                    elif Mode2 == 1:
                        step_frequency_cmd = -event.value + 3.0
                        print(f"Step Frequency command: {step_frequency_cmd}")
                    
                    elif Mode2 == 2:
                        pitch_cmd = -event.value
                        print(f"Pitch command: {pitch_cmd}")


            # Handle button press for gait selection
            if event.type == pygame.JOYBUTTONDOWN:
                """
                A = 0 | bounding: [0, 0.5, 0]
                B = 1 | trotting: [0.5, 0, 0]
                X = 2 | pacing:   [0, 0, 0.5]
                Y = 3 | pronking: [0, 0, 0]
                # Left stick click = 4, Mode A/B/C
                # Right stick click = 5, Mode D/E/F
                """
                if event.button == 0:  # A button
                    gait = torch.tensor(gaits["bounding"])
                    print("Selected gait: Bounding")
                elif event.button == 1:  # B button
                    gait = torch.tensor(gaits["trotting"])
                    print("Selected gait: Trotting")
                elif event.button == 2:  # X button
                    gait = torch.tensor(gaits["pacing"])
                    print("Selected gait: Pacing")
                elif event.button == 3:  # Y button
                    gait = torch.tensor(gaits["pronking"])
                    print("Selected gait: Pronking")
                elif event.button == 4: # Left stick click
                    Mode1 = (Mode1 + 1) % len(modes1)
                    current_mode = modes1[Mode1]
                    print(f"Switched to Mode: {current_mode}")
                elif event.button == 5: # Left stick click
                    Mode2 = (Mode2 + 1) % len(modes2)
                    current_mode = modes2[Mode2]
                    print(f"Switched to Mode: {current_mode}")
        
        pygame.display.flip()

        # Simulation step
        with torch.no_grad():
            actions = policy(obs)

        # Update environment commands
        env.commands[:, 0] = x_vel_cmd  # Forward/backward velocity
        env.commands[:, 1] = y_vel_cmd  # Lateral velocity
        env.commands[:, 2] = yaw_vel_cmd  # Yaw rate
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd

        obs, rew, done, info = env.step(actions)
        
    pygame.quit()

if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    interact_go2(headless=False)
