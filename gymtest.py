from collections import defaultdict
import multiprocessing
import queue
import sys
import os
import threading
import typing
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

def setup_fly(timestep):
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
        targeted_fly_names=[fly.name],
        play_speed=.2,
        window_size=(1280, 720),
    )
    sim = flygym.SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep
    )
    sim.physics._warnings_cause_exception = False
    obs, info = sim.reset()
    return sim, obs, info

class MjcSim:
    def __init__(self, dataset_name, timestep = .0001):
        self.dataset_name = dataset_name
        self.mjc_spikes_queue = queue.Queue()
        self.obs_queue = queue.Queue()
        self.frame_queue: typing.Any = queue.Queue()
        self.control_queue = queue.Queue()
        self.joint_records = []
        self.timestep = timestep

        self._start()

    def _start(self):
        self.mjc_thread = threading.Thread(target=self.start_mjc_thread_, args=())
        self.mjc_thread.start()
    
    def put_spikes(self, spikes):
        self.mjc_spikes_queue.put(spikes, False)

    def reset(self):
        self.joint_records = []
        self.control_queue.put("stop")
        self.mjc_spikes_queue.put([])
        self.mjc_thread.join()
        self._start()

    def start_mjc_thread_(self):
        if self.dataset_name == "banc":
            leg_neuron_groups = neuron_groups.banc_leg_neuron_groups
        elif self.dataset_name == "mbanc" or self.dataset_name == "mbanc-no-optic":
            leg_neuron_groups = neuron_groups.mbanc_leg_neuron_groups
        else:
            raise Exception("invalid dataset " + self.dataset_name)

        muscle_to_gym_muscle_index = get_muscle_to_gym_muscle_index(leg_neuron_groups)
        neuron_to_muscle_index = get_neuron_to_muscle_index(leg_neuron_groups, muscle_to_gym_muscle_index)

        actuated_joints = flygym_preprogrammed.all_leg_dofs

        sim, obs, info = setup_fly(self.timestep)
        self.obs_queue.put(obs) #this is nothing for now
        # print("asdf 1")
        # with sim.physics.suppress_physics_errors() as thing:

            # print(thing)
            # print(type(thing))
        # print("asdf 2")
        #starting_pose: flygym_state.kinematic_pose.KinematicPose = flygym_preprogrammed.get_preprogrammed_pose("tripod")
        #starting_positions = np.zeros(len(actuated_joints))
        starting_positions = get_starting_pose2()

        action_size = len(flygym.preprogrammed.all_leg_dofs) # type: ignore

        joint_state = starting_positions.copy()

        walk_data, walk_data_timestep = get_walk_data()

        step_num = 0
        steps_per_second = 10_000
        i = 0
        while True:
            if self.control_queue.qsize() > 0:
                try:
                    control = self.control_queue.get_nowait()
                    if control == "stop":
                        print("mjc stopping")
                        break
                except queue.Empty:
                    pass
            spikes = self.mjc_spikes_queue.get()
            for spike in spikes:
                # spikes_processed += 1

                if spike in neuron_to_muscle_index:
                    index, sign = neuron_to_muscle_index[spike]
                    joint_state[index] += sign * .09

                    # print("spiked!", spike, actuated_joints[index], joint_state[index])
                # for group, group_neurons in leg_neuron_groups.items():
                    # if spike in group_neurons and group in muscle_to_gym_muscle_index:
                        # joint_index, sign = muscle_to_gym_muscle_index[group]
                        # print(joint, joints, joints[joint], joint_state)
                        # joint_state[joint_index] += sign * .05

            action = {"joints": joint_state}
            self.joint_records.append(joint_state.copy())
            # action = {"joints": interp_walk_data(walk_data, walk_data_timestep, i / steps_per_second)}
            # self.joint_records.append(interp_walk_data(walk_data, walk_data_timestep, i / steps_per_second))
            i += 1

            angle_decay = .005
            joint_state[:] = (starting_positions * angle_decay + joint_state * (1 - angle_decay))
            obs, reward, terminated, truncated, info = sim.step(action)
            self.obs_queue.put(obs) #this is nothing for now

            if step_num % 10 == 0:
                if self.frame_queue is not None:
                    frame = sim.render()[0]
                    if frame is not None:
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.frame_queue.put((frame.shape[1], frame.shape[0], frame.tobytes(), step_num / steps_per_second))
                        # self.frame_queue.width = frame.shape[1]
                        # self.frame_queue.height = frame.shape[0]
                        # self.frame_queue.data = frame.tobytes()

            step_num += 1


def get_neuron_to_muscle_index(leg_neuron_groups, muscle_to_gym_muscle_index):
    result = {}
    for group in leg_neuron_groups:
        neurons = leg_neuron_groups[group]
        if group in muscle_to_gym_muscle_index:
            for neuron in neurons:
                result[neuron] = muscle_to_gym_muscle_index[group]

    return result

def get_muscle_index_to_neuron(rev_dict):
    result = defaultdict(list)
    for item in rev_dict:
        result[rev_dict[item][0]].append((item, rev_dict[item][1]))
    return result

def get_muscle_to_gym_muscle_index(leg_neuron_groups):
    muscle_to_gym_muscle = {}

    muscle_type_to_gym_type = {
        "trochanter_flexor": ("Femur", -1),
        "trochanter_extensor": ("Femur", +1),
        "tibia_flexor": ("Tibia", 2),
        "tibia_extensor": ("Tibia", -2),
        "tarsus_depressor": ("Tarsus1", 5),
        "tarsus_levetator": ("Tarsus1", -5),
        "long_tendon": (None, 1),
    }

    actuated_joints = flygym_preprogrammed.all_leg_dofs
    joint_indices = {}
    for i, joint in enumerate(actuated_joints):
        joint_indices[joint] = i

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
                    muscle_to_gym_muscle[neuron] = (joint_indices["joint_" + side + leg + gym_name], sign)

    return muscle_to_gym_muscle

def get_walk_data():
    actuated_joints = flygym_preprogrammed.all_leg_dofs
    data_path = flygym.get_data_path("flygym", "data")
    file = "210902_pr_fly1.pkl"
    # file = "single_steps_untethered.pkl"
    with open(data_path / "behavior" / file, "rb") as f:
        data = pickle.load(f)

    data_block = np.zeros((len(actuated_joints), len(data["joint_LFCoxa"])))
    for i, joint in enumerate(actuated_joints):
        data_block[i, :] = data[joint]

    # return data_block, data["meta"]["timestep"]
    return data_block, .00025

def interp_walk_data(walk_data, orig_timestep, time):
    actuated_joints = flygym_preprogrammed.all_leg_dofs
    times = np.arange(walk_data.shape[1]) * orig_timestep
    joint_positions = np.zeros(len(actuated_joints))
    for i in range(joint_positions.shape[0]):
        joint_positions[i] = np.interp(time, times, walk_data[i])

    return joint_positions

def get_starting_pose1():
    actuated_joints = flygym_preprogrammed.all_leg_dofs
    starting_pose: flygym_state.kinematic_pose.KinematicPose = flygym_preprogrammed.get_preprogrammed_pose("tripod")
    starting_positions = np.zeros(len(actuated_joints))
    for i, joint in enumerate(actuated_joints):
        starting_positions[i] = starting_pose[joint]
    return starting_positions

def get_starting_pose2():
    walk_data, _ = get_walk_data()
    pose = np.median(walk_data, axis=1)
    # pose[26] = .9
    pose[5] = .6
    return pose
    
if __name__ == "__main__":
    run_time = 1.1
    timestep = .0001

    actuated_joints = flygym_preprogrammed.all_leg_dofs

    sim, obs, info = setup_fly(timestep=.0001)

    starting_positions = get_starting_pose1()

    walk_data, walk_data_timestep = get_walk_data()

    alt_starting_positions = get_starting_pose2()

    walk_data[:, 0] = alt_starting_positions

    target_num_steps = int(run_time / timestep)
    data_block = np.zeros((len(actuated_joints), target_num_steps))
    # input_t = np.arange(len(walk_data["joint_LFCoxa"])) * walk_data["meta"]["timestep"]
    input_t = np.arange(len(walk_data[0])) * walk_data_timestep
    output_t = np.arange(target_num_steps) * timestep
    print("input", input_t)
    print("output shape", output_t.shape)
    print("data timestep", walk_data_timestep)
    for i, joint in enumerate(actuated_joints):
        data_block[i, :] = np.interp(output_t, input_t, walk_data[i])

    walk_data_interp = data_block

    """
    import matplotlib.pyplot as plt
    plt.plot(walk_data[3], color="green")
    plt.plot(walk_data_interp[3])
    plt.show()
    """

    print("shapes", walk_data_interp.shape, alt_starting_positions.shape)
    print("starts", starting_positions, alt_starting_positions)

    print("shape", walk_data_interp.shape)

    # fig.savefig(output_dir / "kin_replay_joint_dof_time_series.png")

    multiprocessing.freeze_support()
    mp_context = multiprocessing.get_context("spawn")
    frame_queue = mp_context.Queue()
    event_queue = mp_context.Queue()
    WIDTH, HEIGHT = (1280, 720)
    process = mp_context.Process(target=display.pygame_thread, args=(frame_queue, event_queue, (WIDTH, HEIGHT)))
    process.start()

    scene_option = mujoco.wrapper.core.MjvOption()
    print(scene_option)
    scene_option.geomgroup[:] = 1  # Hide external meshes.
    print(scene_option)

    # for i in range(target_num_steps):
    i = 0
    joints = walk_data_interp[:, 0]
    joints = starting_positions
    current_actuator = 0
    step = 0
    mode = 0
    paused = False
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
                        if not paused:
                            current_actuator -= 1
                            print(actuated_joints[current_actuator])
                        else:
                            i -= 10
                            i %= walk_data_interp.shape[1]
                    if event[1]["key"] == pygame.K_RIGHT:
                        if not paused:
                            current_actuator += 1
                            print(actuated_joints[current_actuator])
                        else:
                            i += 10
                            i %= walk_data_interp.shape[1]
                    if event[1]["key"] == pygame.K_SPACE:
                        paused = not paused
                    if event[1]["key"] == pygame.K_j:
                        mode = 1 - mode
        except queue.Empty:
            pass

        step += 1
        # here, we simply use the recorded joint angles as the target joint angles
        if not paused and step > 2000:
            i += 1
            i %= walk_data_interp.shape[1]
        joint_pos = walk_data_interp[:, i]
        joint_pos2 = interp_walk_data(walk_data, walk_data_timestep, i * timestep)
        if (joint_pos != joint_pos2).any():
            print("UNEQUAL", joint_pos, joint_pos2)
            input()
        if mode == 0:
            action = {"joints": joint_pos2}
        elif mode == 1:
            action = {"joints": joints}
        # action = {}
        obs, reward, terminated, truncated, info = sim.step(action)
        frame = sim.render()[0]
        if frame is not None:
            frame_queue.put((step / 10000, frame.tobytes()))

    # output_dir = Path("outputs/gym_basics/")
    # output_dir.mkdir(exist_ok=True, parents=True)
    # cam.save_video(output_dir / "kinematic_replay.mp4")
