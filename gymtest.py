import multiprocessing
import queue
import sys
import os
import flygym
import flygym.state as flygym_state
import flygym.preprogrammed as flygym_preprogrammed
from flygym.preprogrammed import all_leg_dofs
import pickle
from dm_control import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pygame
from tqdm import trange

import display
import neuron_groups

def setup_fly():
    actuated_joints = flygym.preprogrammed.all_leg_dofs # type: ignore
    #poses: stretch, tripod, zero
    fly = flygym.Fly(init_pose="tripod", 
        actuated_joints=actuated_joints, 
        floor_collisions="legs",
        control="position",
        #control="motor",
        )
    cam = flygym.ZStabilizedCamera(
        attachment_point=fly.model.worldbody, camera_name="camera_left",
        targeted_fly_names=fly.name,
        play_speed=0.3,
        window_size=(1280, 720),
    )
    sim = flygym.SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=0.0001
    )
    sim.physics.copy
    sim.physics._warnings_cause_exception = False
    obs, info = sim.reset()
    return sim, obs, info
    

def start_mjc_thread(dataset, mjc_spikes_queue, frame_queue: queue.Queue | None, obs_queue: queue.Queue):
    leg_neuron_groups = {}
    if dataset == "banc":
        leg_neuron_groups = neuron_groups.banc_leg_neuron_groups
    if dataset == "mbanc":
        leg_neuron_groups = neuron_groups.mbanc_leg_neuron_groups
    muscle_to_gym_muscle = {}

    muscle_type_to_gym_type = {
        "trochanter_flexor": ("Femur", -1),
        "trochanter_extensor": ("Femur", +1),
        "tibia_flexor": ("Tibia", 2),
        "tibia_extensor": ("Tibia", -2),
        "tarsus_depressor": ("Tarsus1", 4),
        "tarsus_levetator": ("Tarsus1", -4),
        "long_tendon": (None, 1),
    }

    for neuron in leg_neuron_groups.keys():
        side = leg = None
        if neuron[0] == 'r':
            side = "R"
        if neuron[0] == 'l':
            side = "L"
        if neuron[1] == 'f':
            leg = 'F'
        if neuron[1] == 'm':
            leg = 'M'
        if neuron[1] == 'h':
            leg = 'H'

        if side is None or leg is None:
            print("error building map thing")
            print(neuron, side, leg)
            exit()

        for muscle in muscle_type_to_gym_type.keys():
            if muscle in neuron: # example: joint_LHFemu
                gym_name, sign = muscle_type_to_gym_type[muscle]
                if gym_name != None:
                    muscle_to_gym_muscle[neuron] = ("joint_" + side + leg + gym_name, sign)

    print(muscle_to_gym_muscle)

    sim, obs, info = setup_fly()
    obs_queue.put(obs) #this is nothing for now
    print("asdf 1")
    # with sim.physics.suppress_physics_errors() as thing:
        # print(thing)
        # print(type(thing))
    print("asdf 2")
    actuated_joints = flygym_preprogrammed.all_leg_dofs
    starting_pose: flygym_state.kinematic_pose.KinematicPose = flygym_preprogrammed.get_preprogrammed_pose("tripod")
    print(starting_pose.joint_pos)
    starting_positions = np.zeros(len(actuated_joints))
    joints = {}
    for i, joint in enumerate(actuated_joints):
        starting_positions[i] = starting_pose[joint]
        joints[joint] = i

    action_size = len(flygym.preprogrammed.all_leg_dofs) # type: ignore

    joint_state = starting_positions.copy()

    step_num = 0
    steps_per_second = 10_000
    while True:
        spikes = mjc_spikes_queue.get()
        for spike in spikes:
            # spikes_processed += 1

            for group, group_neurons in leg_neuron_groups.items():
                if spike in group_neurons and group in muscle_to_gym_muscle:
                    joint, sign = muscle_to_gym_muscle[group]
                    joint_state[joints[joint]] += sign * .1
                    print("spiked!", group, joint)
            # print("spiked", spike)

        action = {"joints": joint_state}
        joint_state[:] = (starting_positions * .001 + joint_state * .999)
        obs, reward, terminated, truncated, info = sim.step(action)
        obs_queue.put(obs) #this is nothing for now

        print('in sim')

        if frame_queue is not None:
            frame = sim.render()[0]
            if frame is not None:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put((frame.shape[1], frame.shape[0], frame.tobytes(), step_num / steps_per_second))
                # frame_queue.width = frame.shape[1]
                # frame_queue.height = frame.shape[0]
                # frame_queue.data = frame.tobytes()

        step_num += 1


if __name__ == "__main__":
    run_time = .1
    timestep = 1e-4

    actuated_joints = flygym_preprogrammed.all_leg_dofs
    print(actuated_joints)

    data_path = flygym.get_data_path("flygym", "data")
    with open(data_path / "behavior" / "210902_pr_fly1.pkl", "rb") as f:
        data = pickle.load(f)

    sim, obs, info = setup_fly()

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

        scene_option = mujoco.wrapper.core.MjvOption()
        print(scene_option)
        scene_option.geomgroup[:] = 1  # Hide external meshes.
        print(scene_option)

        # for i in range(target_num_steps):
        i = 0
        #joints = data_block[:, 0]
        joints = starting_positions
        current_actuator = 0
        step = 0
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
            step += 1
            # here, we simply use the recorded joint angles as the target joint angles
            # joint_pos = data_block[:, i]
            action = {"joints": joints}
            # action = {}
            obs, reward, terminated, truncated, info = sim.step(action)
            frame = sim.render()[0]
            if frame is not None:
                frame_queue.put((step / 10000, frame.tobytes()))

        # output_dir = Path("outputs/gym_basics/")
        # output_dir.mkdir(exist_ok=True, parents=True)
        # cam.save_video(output_dir / "kinematic_replay.mp4")
