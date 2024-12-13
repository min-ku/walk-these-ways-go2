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
    Cfg.terrain.num_rows = 50
    Cfg.terrain.num_cols = 50
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 1
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


def play_go2(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go2_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    # label = "gait-conditioned-agility/pretrain-v0/train"
    # label = "gait-conditioned-agility/pretrain-go2/train"
    label = "gait-conditioned-agility/2024-12-11/train"


    env, policy = load_env(label, headless=headless)

    num_eval_steps = 1000 #250
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    # x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.5, 0.0, 0.0
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = .5, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0 #3.0
    # gait = torch.tensor(gaits["pronking"])
    # gait = torch.tensor(gaits["trotting"])
    # gait = torch.tensor(gaits["bounding"])
    gait = torch.tensor(gaits["pacing"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))
    ###### -----------ldt---------------
    joint_torques = np.zeros((num_eval_steps, 12))

    obs = env.reset()
    reward_buffer = np.zeros((num_eval_steps, len(env.reward_functions), env.num_envs))

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        joint_positions[i] = env.dof_pos[0, :].cpu()

        for j in range(19): # 19
            name = env.reward_names[j]
            rew = env.reward_functions[j]() * env.reward_scales[name]
            reward_buffer[i][j] = rew.detach().cpu().numpy()
            #print(rew)
        ###### -----------ldt---------------
        # joint_torques[i] = env.torques.detach().cpu().numpy()

    # plot target and measured forward velocity
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    axs[1].set_title("Joint Positions")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")

    axs[2].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_torques, linestyle="-", label="Measured")
    axs[2].set_title("Joint Torques")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Joint Torques (Nm)")

    plt.tight_layout()
    # plt.show()
    plt.savefig("velocity_plot.png")

    """
    tracking_lin_vel : xy velocity tracking 0.08
    tracking_ang_vel : yaw velocity tracking

    lin_vel_z        : z velocity
    ang_vel_xy       : roll-pitch velocity
    torques          : joint torques
    dof_vel          : joint velocities
    dof_acc          : joint accelerations
    collision        : thigh/calf collision
    action_rate      : Penalizes changes in actions sum([a_{t-1} - a_t]^2)
    tracking_contacts_shaped_force  : swing phase tracking (force)
    tracking_contacts_shaped_vel    : stance phase tracking (velocity)
    jump                : body height tracking
    dof_pos_limits      : joint limit violation
    feet_slip           : foot slip
    feet_clearance_cmd_linear : footswing height tracking
    action_smoothness_1 : action smoothing (joint)
    action_smoothness_2 : action smoothing, second order (joint)
    raibert_heuristic   : raibert heuristic footswing tracking
    orientation_control : body pitch tracking
    
    Scale:  0.07999999821186066
            0.07999999821186066
            0.19999999552965164
           -0.19999999552965164
           -0.0007999999821186066
           -0.5999999865889549
           -0.0019999999552965165
           -0.0019999999552965165
           -0.19999999552965164
           -0.09999999776482582
            0.019999999552965164
            0.009999999776482582
           -0.0003999999910593033
           -1.9999999552965164e-05
           -1.9999999552965164e-06
           -1.9999999552965164e-06
           -4.999999888241291e-09
           -0.09999999776482582
           -0.00019999999552965166

    """
    reward_names = [
    "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z", "ang_vel_xy", "torques",
    "dof_vel", "dof_acc", "collision", "action_rate",
    "tracking_contacts_shaped_force", "tracking_contacts_shaped_vel", "jump",
    "dof_pos_limits", "feet_slip", "feet_clearance_cmd_linear",
    "action_smoothness_1", "action_smoothness_2", "raibert_heuristic", "orientation_control"
    ]

    total_rewards_per_section = np.sum(reward_buffer, axis=(0, 2))

    for idx, total_rewards in enumerate(total_rewards_per_section):
        print(f"Total rewards for section {reward_names[idx]}: {total_rewards}")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(reward_names, total_rewards_per_section, color='skyblue')

    plt.bar(reward_names, total_rewards_per_section, color='skyblue')
    plt.title("Total Rewards for Each Reward Section", fontsize=16)
    plt.xlabel("Reward Function", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)

    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)


    plt.tight_layout()
    plt.savefig("Reward_plot.png")

    

if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go2(headless=False)
