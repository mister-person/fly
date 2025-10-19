from collections import defaultdict
from sys import argv
import pandas as pd
import pygame
import numpy as np
import rust_utils
from rust_utils import load_synapses

from drawutils import NeuronDrawer, SpikeDrawer

TIME_MULTIPLIER = 1000

WIDTH, HEIGHT = (1500, 1200)

SPIKE_FILE = "results/test1.parquet"
#SPIKE_FILE = "results/no_in_sorted.parquet"
#SPIKE_FILE = "results/no_activations.parquet"

if len(argv) > 1:
    SPIKE_FILE = argv[1]

print("reading from", SPIKE_FILE)

if "banc" in argv:
    neuron_coords_path = "../flywire/banc_coordinates.csv"
    neuron_coords = pd.read_csv(neuron_coords_path).to_numpy()
    synapse_coords_filename = "../flywire/banc_connections_princeton.csv"
    SYNAPSES_FILENAME = "./data/banc_connectivity.parquet"
else:
    neuron_coords_path = "../flywire/fafb_coordinates.csv"
    neuron_coords = pd.read_csv(neuron_coords_path).to_numpy()
    synapse_coords_filename = "../flywire/fafb_v783_princeton_synapse_table.csv"
    # SYNAPSES_FILENAME = "./Drosophila_brain_model/2023_03_23_connectivity_630_final.parquet"
    SYNAPSES_FILENAME = "./Drosophila_brain_model/Connectivity_783.parquet"
synapse_coords_file = open(synapse_coords_filename, "rb")

num_synapses = 80215790
# num_synapses = 8000000
# synapse_coords = np.empty((num_synapses, 4), dtype=np.integer)

synapse_coords_file.readline()
index = 0
#while (synapse := synapse_coords_file.readline()) != "":

# a = load_synapses(synapse_coords_filename)
# synapse_coords = np.frombuffer(a, dtype=np.int64).reshape((-1, 4))

drawer = NeuronDrawer(neuron_coords_path, WIDTH, HEIGHT)

synapses = pd.read_parquet(SYNAPSES_FILENAME)
synapses = synapses.to_numpy()
synapse_map = defaultdict(list)
for synapse in synapses[1:]:
    synapse_map[synapse[0]].append((synapse[1], synapse[6]))

spikes = pd.read_parquet(SPIKE_FILE)
print(spikes)
print("about to sort")
# spikes.sort_values("t")
# print(spikes[0][0])
spikes = spikes.to_numpy()
if "-nos" not in argv:
    spikes = spikes[spikes[:, 0].argsort()]
print("done sorting")

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

screen.fill("black")
clock = pygame.time.Clock()
running = True
frame = 0
all_coords = list(drawer.coord_map.values())

spike_index = 0

neu_sugar = [
    720575940624963786,
    720575940630233916,
    720575940637568838,
    720575940638202345,
    720575940617000768,
    720575940630797113,
    720575940632889389,
    720575940621754367,
    720575940621502051,
    720575940640649691,
    720575940639332736,
    720575940616885538,
    720575940639198653,
    720575940620900446,
    720575940617937543,
    720575940632425919,
    720575940633143833,
    720575940612670570,
    720575940628853239,
    720575940629176663,
    720575940611875570,
]

target_neurons = [
    720575940660219265,
    720575940618238523,
]

main_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

'''
for i in range(len(all_coords)):
    neuron = all_coords[i]
    for pos in neuron:
        #pygame.draw.circle(screen, "blue", to_screen_coords(pos), 1)
        main_surface.set_at(drawer.to_screen_coords(pos), (0, 0, 100))
'''
for i in range(len(neu_sugar)):
    drawer.color_map[neu_sugar[i]] = (255, 255, 0, 255)
for i in range(len(target_neurons)):
    drawer.color_map[target_neurons[i]] = (0, 255, 0, 255)

drawer.surface = main_surface
drawer.draw()
'''
for i in range(len(neu_sugar)):
    neuron = drawer.coord_map[neu_sugar[i]]
    for pos in neuron:
        #pygame.draw.circle(screen, "blue", to_screen_coords(pos), 1)
        main_surface.set_at(drawer.to_screen_coords(pos), "yellow")
for i in range(len(target_neurons)):
    neuron = drawer.coord_map[target_neurons[i]]
    for pos in neuron:
        pygame.draw.circle(main_surface, "green", drawer.to_screen_coords(pos), 10)
        #main_surface.set_at(drawer.to_screen_coords(pos), "green")
        #'''

pygame.key.set_repeat(200, 5)
spike_count = 0
font = pygame.font.Font(size=50)
synapse_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

spiked_neurons = set()

spike_drawer = SpikeDrawer(800, 600)
for n in neu_sugar:
    spike_drawer.neurons.append(n)
    spike_drawer.color_map[n] = (255, 255, 0)
for n in target_neurons:
    spike_drawer.neurons.append(n)
    spike_drawer.color_map[n] = (255, 255, 255)

spike_drawer.add_points(spikes[:, [0, 2]])

last_time = 0.0
time = 0.0
paused = False
time_step = 1/60/TIME_MULTIPLIER
while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.dict["key"] == pygame.K_PERIOD:
                # synapse_surface.fill((0, 0, 0, 0))
                # frame += 1
                # spike_count = 0
                if paused:
                    last_time = time
                    time += time_step
            if event.dict["key"] == pygame.K_COMMA:
                if paused:
                    time -= time_step
                    last_time = time - time_step
            if event.dict["key"] == pygame.K_MINUS:
                time_step /= 2

                spike_drawer.time_size /= 2
                spike_drawer.reset_surfaces()
            if event.dict["key"] == pygame.K_EQUALS:
                time_step *= 2
                spike_drawer.time_size *= 2
                spike_drawer.reset_surfaces()
            if event.dict["key"] == pygame.K_SPACE:
                paused = not paused
        if event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.key.get_pressed()[pygame.K_m]:
                time = pygame.mouse.get_pos()[0] / WIDTH
                last_time = time

    spike_count = 0
    unique_spikes = set()
    if spikes[spike_index][0] > time:
        spike_index = 0
    for i, spike in enumerate(spikes[spike_index:]):
        if spike[1] != 0:
            continue
        # if spike[0] * 60 * TIME_MULTIPLIER < frame:
        # print(last_time, spike[0], time)
        if spike[0] < time:
            if spike[0] < last_time:
                continue
            # print(int(spike[0] * 10000), spike[1:])
            
            if spike[2] not in unique_spikes:
                spike_count += 1
                # print("unique:", spike[2])
            unique_spikes.add(spike[2])
            if spike[2] not in spiked_neurons:
                print("unique:", spike[2], "count:", len(spiked_neurons))
            spiked_neurons.add(spike[2])
            n_poses = drawer.coord_map[spike[2]]
            for pos in n_poses:
                if spike[2] in target_neurons:
                    new_color = "white"
                else:
                    color = main_surface.get_at(drawer.to_screen_coords(pos[:3]))
                    new_color = (min(color.r + 85, 255), color.g, color.b)
                main_surface.set_at(drawer.to_screen_coords(pos[:3]), new_color)
                # print(pos, spike_index)

            other_neurons = synapse_map[spike[2]]
            for other_neuron in other_neurons:
                color = (max(0, min(255, other_neuron[1] * 5 + 128)), max(0, min(255, other_neuron[1] * 5 + 128)), 128)
                # print(n_poses)
                # print(other_neuron)
                # print(coord_map[other_neuron])
                if len(n_poses) == 0:
                    print(n_poses, spike)
                if len(n_poses) > 0 and len(drawer.coord_map[other_neuron[0]]) > 0:
                    pygame.draw.line(synapse_surface, color, drawer.to_screen_coords(n_poses[0]), drawer.to_screen_coords(drawer.coord_map[other_neuron[0]][0]))
        else:
            if not paused:
                spike_index += i
            # print(spike_index, spikes[spike_index][0], frame)
            break

    screen.fill((0, 0, 0))
    screen.blit(font.render(f"{clock.get_fps():.1f} fps", True, "blue"))
    screen.blit(font.render("spike count: " + str(spike_count), True, "blue"), dest=(0, 25))
    screen.blit(font.render(f"time {time:.4f} step {time_step * 60:.4f}/sec", True, "blue"), dest=(0, 50))
    screen.blit(main_surface)
    screen.blit(synapse_surface)
    synapse_surface.fill((0, 0, 0, 0))

    spike_drawer.draw(screen, (12, 600), time)

    pygame.display.flip()
    clock.tick(60)

    frame += 1
    if not paused:
        last_time = time
        time += time_step
    

