from collections import defaultdict
import pandas as pd
import pygame
import numpy as np
import rust_utils
from rust_utils import load_synapses

neuron_coords_path = "../flywire/banc_coordinates.csv"
# neuron_coords_path = "../flywire/fafb_coordinates.csv"
neuron_coords = pd.read_csv(neuron_coords_path).to_numpy()
synapse_filename = "../flywire/fafb_v783_princeton_synapse_table.csv"
synapse_file = open(synapse_filename, "rb")
print("done loading synapses")

num_synapses = 80215790
# num_synapses = 8000000
# synapses = np.empty((num_synapses, 4), dtype=np.integer)

synapse_file.readline()
index = 0
#while (synapse := synapse_file.readline()) != "":

a = load_synapses(synapse_filename)
synapses = np.frombuffer(a, dtype=np.int64).reshape((-1, 4))
'''
file_buffer = bytes()
file_index = 0
for index in range(num_synapses):
    next_nl = file_buffer.find(b"\n", file_index)
    if next_nl == -1:
        READ_SIZE = 100000
        new_data = synapse_file.read(READ_SIZE)
        file_buffer = file_buffer[file_index:] + new_data
        file_index = 0
        next_nl = file_buffer.find(b"\n", file_index)
        if index % 1_000_000  < 1000:
            print(index, "/", num_synapses)
    synapse = file_buffer[file_index:next_nl]
    arr = synapse.split(b',')
    synapses[index] = [arr[3], arr[4], arr[5], arr[10]]
    file_index = next_nl + 1

split = [b""]
index = 0
last_print = 0
while index < num_synapses:
    READ_SIZE = 100000
    new_data = synapse_file.read(READ_SIZE)
    file_buffer = split[-1]+ new_data
    split = file_buffer.split(b"\n")
    for i in range(len(split) - 1):
        synapse = split[i]
        arr = synapse.split(b',')
        synapses[index] = [arr[3], arr[4], arr[5], arr[10]]
        index += 1
    if index > last_print + 1_000_000:
        print(last_print, "/", num_synapses)
        last_print += 1_000_000
#'''
print("done converting to numpy")

max_x = 0
min_x = 100000000
max_y = 0
min_y = 100000000
max_z = 0
min_z = 100000000
coord_map = defaultdict(list)
for n in neuron_coords:
    [x, y, z] = n[1][1:-1].strip().replace("  ", " ").replace("  ", " ").split(" ")
    x, y, z = (int(x), int(y), int(z))
    coord_map[n[0]].append((x, y, z))

    if x > max_x:
        max_x = x
    if x < min_x:
        min_x = x
    if y > max_y:
        max_y = y
    if y < min_y:
        min_y = y
    if z > max_z:
        max_z = z
    if z < min_z:
        min_z = z

print(max_x, min_x, max_y, min_y, max_z, min_z)
print(len(coord_map))

WIDTH, HEIGHT = (1500, 1200)
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

def to_screen_coords(pos):
    (x, y, _) = pos
    width = WIDTH
    height = HEIGHT
    scale = max((max_x - min_x) / width, (max_y - min_y) / height)
    out_x = (x - min_x) / scale
    out_y = (y - min_y) / scale
    return (int(out_x), int(out_y))

screen.fill("black")
clock = pygame.time.Clock()
running = True
frame = 0
all_coords = list(coord_map.values())

print(synapses[:30])

drawing_synapses = False
while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    n_per_frame = 100000
    if not drawing_synapses:
        for i in range(n_per_frame):
            coord_index = frame * n_per_frame + i
            if coord_index >= len(all_coords):
                drawing_synapses = True
                frame = 0
                break
            neuron = all_coords[coord_index]
            for pos in neuron:
                #pygame.draw.circle(screen, "blue", to_screen_coords(pos), 1)
                screen.set_at(to_screen_coords(pos), "blue")
    else:
        for i in range(n_per_frame):
            coord_index = frame * n_per_frame + i
            if coord_index % 1_000_000 == 0:
                print(coord_index, "/", num_synapses)
            if coord_index >= len(synapses):
                break
            synapse = synapses[coord_index]
            try:
                color = screen.get_at(to_screen_coords(synapse[:3]))
                new_color = (min(color.r + 1, 255), min(color.g + 1, 255), color.b)
                screen.set_at(to_screen_coords(synapse[:3]), new_color)
            except IndexError:
                pass
        

    pygame.display.flip()

    frame += 1
    clock.tick(60)

