import multiprocessing
import queue
import sys
import os
import flygym
import flygym.state as flygym_state
import flygym.preprogrammed as flygym_preprogrammed
from flygym.preprogrammed import all_leg_dofs
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pygame
from tqdm import trange

import display

def setup_fly():
    actuated_joints = flygym.preprogrammed.all_leg_dofs # type: ignore
    #poses: stretch, tripod, zero
    fly = flygym.Fly(init_pose="tripod", 
        actuated_joints=actuated_joints, 
        floor_collisions="all",
        control="position")
    cam = flygym.ZStabilizedCamera(
        attachment_point=fly.model.worldbody, camera_name="camera_left",
        targeted_fly_names=fly.name,
        play_speed=0.05,
        window_size=(1280, 720),
    )
    sim = flygym.SingleFlySimulation(
        fly=fly,
        cameras=[cam],
    )
    obs, info = sim.reset()
    return sim, obs, info
    

if __name__ == "__main__":
    run_time = .1
    timestep = 1e-4

    actuated_joints = flygym_preprogrammed.all_leg_dofs
    print(actuated_joints)

    data_path = flygym.get_data_path("flygym", "data")
    with open(data_path / "behavior" / "210902_pr_fly1.pkl", "rb") as f:
        data = pickle.load(f)

    sim, obs, info = setup_fly()

    sim.fly.init_pose
    starting_pose: flygym_state.kinematic_pose.KinematicPose = flygym_preprogrammed.get_preprogrammed_pose("tripod")
    print(starting_pose.joint_pos)
    starting_positions = np.zeros(len(actuated_joints))
    for i, joint in enumerate(actuated_joints):
        starting_positions[i] = starting_pose[joint]

    target_num_steps = int(run_time / timestep)
    data_block = np.zeros((len(actuated_joints), target_num_steps))
    input_t = np.arange(len(data["joint_LFCoxa"])) * data["meta"]["timestep"]
    output_t = np.arange(target_num_steps) * timestep
    for i, joint in enumerate(actuated_joints):
        data_block[i, :] = np.interp(output_t, input_t, data[joint])

    # fig.savefig(output_dir / "kin_replay_joint_dof_time_series.png")

    if __name__ == "__main__":
        multiprocessing.freeze_support()
        mp_context = multiprocessing.get_context("spawn")
        frame_queue = mp_context.Queue()
        frame_ready = mp_context.Condition()
        event_queue = mp_context.Queue()
        WIDTH, HEIGHT = (1280, 720)
        process = mp_context.Process(target=display.pygame_thread, args=(frame_queue, frame_ready, event_queue, (WIDTH, HEIGHT)))
        process.start()

        # for i in range(target_num_steps):
        i = 0
        #joints = data_block[:, 0]
        joints = starting_positions
        current_actuator = 0
        while True:
            try: 
                while event := event_queue.get_nowait():
                    if event[0] == pygame.KEYDOWN:
                        if event[1]["key"] == pygame.K_UP:
                            joints[current_actuator] += .4
                            print(actuated_joints[current_actuator], joints[current_actuator])
                        if event[1]["key"] == pygame.K_DOWN:
                            joints[current_actuator] -= .4
                            print(actuated_joints[current_actuator], joints[current_actuator])
                        if event[1]["key"] == pygame.K_LEFT:
                            current_actuator -= 1
                            print(actuated_joints[current_actuator])
                        if event[1]["key"] == pygame.K_RIGHT:
                            current_actuator += 1
                            print(actuated_joints[current_actuator])
            except queue.Empty:
                pass

            i += 1
            i %= 1000
            # here, we simply use the recorded joint angles as the target joint angles
            # joint_pos = data_block[:, i]
            action = {"joints": joints}
            # action = {}
            obs, reward, terminated, truncated, info = sim.step(action)
            frame = sim.render()[0]
            if frame is not None:
                frame_queue.put(frame.tobytes())

        # output_dir = Path("outputs/gym_basics/")
        # output_dir.mkdir(exist_ok=True, parents=True)
        # cam.save_video(output_dir / "kinematic_replay.mp4")
