from collections import defaultdict
import functools
import multiprocessing
from queue import Empty, Queue
from sys import argv
import time
import typing
import pygame
import drawutils
import pandas as pd
import random
from scipy.fft import fft
import neuron_groups
import data
import test
from functools import partial
import pygame_loop

class PygameProcess:
    def __init__(self, dataset_name, color_map):
        multiprocessing.freeze_support()
        mp_context = multiprocessing.get_context("spawn")
        self.pygame_spike_queue = mp_context.Queue()
        self.control_queue = mp_context.Queue()
        self.frame_queue = mp_context.Queue()
        self.dataset_name = dataset_name
        # frame_queue.put((700, 700, np.empty(shape=(700*700*3,), dtype=np.int8).tobytes(), 0))
        render_process = mp_context.Process(target=start_pygame, args=[self.pygame_spike_queue, self.control_queue, self.frame_queue, self.dataset_name, color_map])
        render_process.start()
        self.spike_queue = Queue

    def add_spikes(self, spikes):
        self.pygame_spike_queue.put(spikes)

    def reset(self):
        self.pygame_spike_queue.put(None)

def start_pygame(spike_queue: Queue, control_queue: Queue, frame_queue: Queue, dataset: str, color_map):
    control_queue.put(("pause", True))
    print("start pygame thread")

    WIDTH, HEIGHT = (1500, 1200)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    if dataset == "banc":
        neuron_coords_path = "../flywire/banc_coordinates.csv"
        synapse_coords_filename = "../flywire/banc_connections_princeton.csv"
        SYNAPSES_FILENAME = "./banc_connectivity.parquet"
    elif dataset == "fafb":
        neuron_coords_path = "../flywire/fafb_coordinates.csv"
        synapse_coords_filename = "../flywire/fafb_v783_princeton_synapse_table.csv"
        SYNAPSES_FILENAME = "./2023_03_23_connectivity_630_final.parquet"
    elif dataset == "mbanc":
        # neuron_coords_path = "../flywire/fafb_coordinates.csv"
        # synapse_coords_filename = "../flywire/fafb_v783_princeton_synapse_table.csv"
        # SYNAPSES_FILENAME = "./2023_03_23_connectivity_630_final.parquet"
        pass

    _, synapse_map, _ = data.get_synapse_map(dataset)

    spike_drawer = drawutils.SpikeDrawer(WIDTH, HEIGHT)
    spike_drawer.time_size = 1
    spike_drawer.set_unit_height(1)
    # neuron_drawer = drawutils.NeuronDrawer(neuron_coords_path, WIDTH, HEIGHT)
    spike_drawer.color_map.update(color_map)
    for n in color_map.keys():
        spike_drawer.neurons.append(n)

    fft_drawer = drawutils.FFTDrawer(WIDTH, HEIGHT)
    fft_drawer.update_freq = float("inf")

    c = 0
    for neuron in synapse_map.keys():
        c += 1
        weight = synapse_map[neuron][0][1]
        # spike_drawer.neurons.append(neuron)
        if weight > 0:
            spike_drawer.color_map[neuron] = (255, random.randrange(0, 255), 0)
        else:
            spike_drawer.color_map[neuron] = (0, random.randrange(0, 255), 255)
        
    #TODO all this color stuff shouldn't be in this file
    target_neurons = [
        720575941669833905,
        720575941560438643,
        720575941504247575,
        720575941527002649,
        720575941554038555,
        720575941552245822,

        720575941469024064,720575941487873488,720575941545638505,720575941625121674,720575941437338412,720575941519468654
    ]
    for i, n in enumerate(target_neurons):
        spike_drawer.neurons.append(n)
        if i % 2 == 0:
            spike_drawer.color_map[n] = (255, 255, 255)
        else:
            spike_drawer.color_map[n] = (200, 200, 200)
    red_neurons = [720575941496703104,720575941503572133,720575941589995319,720575941441065919,720575941544954556,720575941687554799,
    ]
    for i, n in enumerate(red_neurons):
        spike_drawer.neurons.append(n)
        if i % 2 == 0:
            spike_drawer.color_map[n] = (255, 100, 100)
        else:
            spike_drawer.color_map[n] = (200, 100, 100)

    yellow_neurons = [720575941569601650,720575941515421443,720575941555553363,720575941414465684,720575941461249747,720575941649246741]
    for i, n in enumerate(yellow_neurons):
        spike_drawer.neurons.append(n)
        if i % 2 == 0:
            spike_drawer.color_map[n] = (255, 255, 100)
        else:
            spike_drawer.color_map[n] = (200, 200, 100)

    purple_neurons = [720575941626500746]
    for i, n in enumerate(purple_neurons):
        spike_drawer.neurons.append(n)
        if i % 2 == 0:
            spike_drawer.color_map[n] = (255, 100, 255)
        else:
            spike_drawer.color_map[n] = (200, 100, 200)

    '''
    right_front_leg_motor = neuron_groups.mbanc_rf_leg_neurons
    for i, n in enumerate(right_front_leg_motor):
        spike_drawer.neurons.append(n)
        if i % 2 == 0:
            spike_drawer.color_map[n] = (200, 255, 100)
        else:
            spike_drawer.color_map[n] = (150, 200, 80)
    left_front_leg_motor = neuron_groups.mbanc_lf_leg_neurons
    for i, n in enumerate(left_front_leg_motor):
        spike_drawer.neurons.append(n)
        if i % 2 == 0:
            spike_drawer.color_map[n] = (50, 255, 150)
        else:
            spike_drawer.color_map[n] = (20, 200, 100)
    all_leg_motor = neuron_groups.mbanc_leg_neurons
    for i, n in enumerate(all_leg_motor):
        if n not in spike_drawer.neurons:
            spike_drawer.neurons.append(n)
            if i % 2 == 0:
                spike_drawer.color_map[n] = (150, 255, 150)
            else:
                spike_drawer.color_map[n] = (100, 200, 100)
    '''

    for leg in neuron_groups.legs:
        neurons = neuron_groups.banc_by_leg[leg]
        leg_index = (neuron_groups.legs.index(leg) * 5) % 7
        color_1 = (200 - leg_index * 25, 100 + leg_index * 25, 255)
        color_2 = (150 - leg_index * 25, 80 + leg_index * 25, 200)
        color_3 = (200 - leg_index * 25, 255, 100 + leg_index * 25)
        color_4 = (150 - leg_index * 25, 200, 80 + leg_index * 25)
        for i, (key, neuron) in enumerate(neurons):
            # print(key, neuron)
            spike_drawer.neurons.append(neuron)
            if "extensor" in key or "levetator" in key:
                if i % 2 == 0:
                    spike_drawer.color_map[neuron] = color_1
                else:
                    spike_drawer.color_map[neuron] = color_2
            else:
                if i % 2 == 0:
                    spike_drawer.color_map[neuron] = color_3
                else:
                    spike_drawer.color_map[neuron] = color_4
        
    spike_drawer.color_map[neuron_groups.mbanc_leg_neuron_groups["rf_trochanter_extensor"][0]] = (180, 180, 180)

    #for r in red_neurons:
    # for t in target_neurons:
        # print(t, "", synapse_map[t])

    running = True
    paused = False
    font = pygame.font.Font(size=30)
    synapse_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    neuron_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    screen.fill("black")
    clock = pygame.time.Clock()

    spikes = []
    spike_index = 0
    current_time = 0
    last_time = current_time
    time_step = 1/60

    mjc_time = 0

    start_time = time.monotonic()
    print("start pygame loop")

    def make_callback(queuename):
        def callback(value: float):
            control_queue.put((queuename, value))
        return callback
    test_slider = drawutils.Slider((200, 0), name="rate1", on_changed=make_callback("rate1"), initial_value=100)

    frame_surface = pygame.Surface((WIDTH, HEIGHT))


    ui_elements = [test_slider]

    control_queue.put(("pause", False))
    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.dict["key"] == pygame.K_1:
                    control_queue.put(("group1", True))
                if event.dict["key"] == pygame.K_SPACE:
                    paused = not paused
                    control_queue.put(("pause", paused))
                if event.dict["key"] == pygame.K_MINUS:
                    spike_drawer.time_size /= 2
                    spike_drawer.reset_surfaces()
                if event.dict["key"] == pygame.K_EQUALS:
                    spike_drawer.time_size *= 2
                    spike_drawer.reset_surfaces()
                if event.dict["key"] == pygame.K_f:
                    if pygame.key.get_mods() & pygame.KMOD_LSHIFT == 0:
                        fft_drawer.update_cache()
                    else:
                        fft_drawer.reset_fft()
                        print("reset fft")
            if event.type == pygame.KEYUP:
                if event.dict["key"] == pygame.K_1:
                    if pygame.key.get_mods() & pygame.KMOD_LSHIFT == 0:
                        control_queue.put(("group1", False))
                    else:
                        print("WAS SHIFTING")
            if event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.key.get_pressed()[pygame.K_m]:
                    current_time = pygame.mouse.get_pos()[0] / WIDTH
                    last_time = current_time
                for slider in ui_elements:
                    slider.click(True)
            elif event.type == pygame.MOUSEBUTTONUP:
                for slider in ui_elements:
                    slider.click(False)
            elif event.type == pygame.MOUSEMOTION:
                for slider in ui_elements:
                    slider.drag(event.dict["rel"])

        try:
            while True:
                queue_get = spike_queue.get_nowait()
                if queue_get is None:
                    print("fucking reset")
                    spike_drawer.data.clear()
                    spike_drawer.reset_surfaces()
                    continue
                current_time, new_spikes = queue_get
                spike_drawer.add_points(new_spikes)
                fft_drawer.add_points(new_spikes)
        except Empty:
            pass
        # queue_len = spike_queue.qsize()
        # if queue_len > 1:
            # print("RUNNING BEHIND BY", queue_len, "FRAMES")

        frame = None
        try:
            while frame := frame_queue.get_nowait():
                pass
        except Empty:
            if frame is not None:
                width, height, frame_bytes, mjc_time = frame

                image = pygame.image.frombytes(frame_bytes, (width, height), "RGB")
                frame_surface.blit(image)

        screen.fill((0, 0, 0))
        screen.blit(font.render(f"{clock.get_fps():.1f} fps", True, "blue"))
        screen.blit(font.render(f"brian time: {current_time:.5f}", True, "blue"), (0, 20))
        screen.blit(font.render(f"mjc time: {mjc_time:.5f}", True, "blue"), (0, 40))
        screen.blit(font.render(f"real time: {time.monotonic() - start_time:.2f}", True, "blue"), (0, 60))
        # screen.blit(neuron_surface)
        screen.blit(synapse_surface)
        synapse_surface.fill((0, 0, 0, 0))

        screen.blit(frame_surface, (0, 100))

        for slider in ui_elements:
            slider.draw(screen)

        spike_drawer.draw(screen, (0, 100), current_time)
        fft_drawer.draw(screen)

        pygame.display.flip()
        clock.tick(60)

        # if not paused:
            # last_time = time
            # time += time_step
        
def start_pygame_profile(spike_queue: Queue, input_queue: Queue, frame_queue: Queue, banc: bool, neurons_to_activate):
    import cProfile
    cProfile.runctx("start_pygame(spike_queue, input_queue, frame_queue, banc, neurons_to_activate)", globals=globals(), locals=locals(), filename="pygame_loop.prof")
