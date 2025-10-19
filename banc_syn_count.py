from collections import defaultdict
import numpy as np
import pandas as pd
import pygame
import random

import drawutils

path_comp = "banc_completeness.csv"
path_con = "banc_connectivity.parquet"
banc_comp_np = pd.read_csv(path_comp, index_col=0).to_numpy()
banc_con_np = pd.read_parquet(path_con).to_numpy()

path_comp = "Completeness_783.csv"
path_con = "Connectivity_783.parquet"
fafb_comp_np = pd.read_csv(path_comp, index_col=0).to_numpy()
fafb_con_np = pd.read_parquet(path_con).to_numpy()

fafb_class = pd.read_csv("../flywire/fafb_classification.csv").to_numpy()
print(fafb_class)
fafb_class = fafb_class.astype("str")

print()

fafb_classes = np.unique(fafb_class[:, 2])
print("fafb flows:")
print(np.unique(fafb_class[:, 1]))
print("fafb classes:")
print(fafb_classes)
#print("fafb sub classes:")
#print(np.unique(fafb_class[:, 3]))

print()

banc_class = pd.read_csv("../flywire/banc_neurons.csv").to_numpy()
banc_class = banc_class.astype("str")

print("banc flows:")
print(np.unique(banc_class[:, 9]))
banc_classes = np.unique(banc_class[:, 10])
print("banc classes:")
print(banc_classes)
#print("banc sub classes:")
#print(np.unique(banc_class[:, 11]))
# for i in range(banc_class.shape[1]):
    # print(np.unique(banc_class[:, i]))

fafb_class_colors = {}
banc_class_colors = {}
def randomize_colors():
    for c in fafb_classes:
        r = random.randrange(0, 255)
        g = random.randrange(0, 255)
        b = random.randrange(0, 255)
        s = r + g + b
        fafb_class_colors[c] = (r * 255 // s, g * 255 // s, b * 255 // s, 150)
        print(fafb_class_colors[c])
    for c in banc_classes:
        r = random.randrange(0, 255)
        g = random.randrange(0, 255)
        b = random.randrange(0, 255)
        s = r + g + b
        banc_class_colors[c] = (r * 255 // s, g * 255 // s, b * 255 // s, 150)
        print(banc_class_colors[c])
randomize_colors()

fafb_class_dict = {}
for n_id in fafb_class:
    fafb_class_dict[int(n_id[0])] = n_id

banc_class_dict = {}
for n_id in banc_class:
    banc_class_dict[int(n_id[0])] = n_id

print(list(fafb_class_dict.items())[:10])

pygame.init()
WIDTH, HEIGHT = (1500, 1200)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.Font(size=50)
drawer_fafb = drawutils.NeuronDrawer("../flywire/fafb_coordinates.csv", WIDTH, HEIGHT)
drawer_banc = drawutils.NeuronDrawer("../flywire/banc_coordinates.csv", WIDTH, HEIGHT)
'''
for n_id in drawer.coord_map.keys():
    c = fafb_class_dict[n_id]
    drawer.color_map[n_id] = class_colors[c[2]]
'''
fafb_class_counts = defaultdict(int)
banc_class_counts = defaultdict(int)
for n_id in drawer_fafb.coord_map.keys():
    c = fafb_class_dict[n_id]
    drawer_fafb.set_color(n_id, fafb_class_colors[c[2]])
    fafb_class_counts[c[2]] += 1
for n_id in drawer_banc.coord_map.keys():
    c = banc_class_dict[n_id]
    drawer_banc.set_color(n_id, banc_class_colors[c[10]])
    banc_class_counts[c[10]] += 1

print("done")

banc = False

while True:
    drawer = drawer_banc if banc else drawer_fafb
    class_dict = banc_class_dict if banc else fafb_class_dict
    class_colors = banc_class_colors if banc else fafb_class_colors
    class_column = 10 if banc else 2
    class_counts = banc_class_counts if banc else fafb_class_counts
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.dict["key"] == pygame.K_PERIOD:
                randomize_colors()
                for n_id in drawer.coord_map.keys():
                    # c = class_dict[n_id]
                    # drawer.set_color(n_id, class_colors[c[2]])
                    c = class_dict[n_id]
                    drawer.set_color(n_id, class_colors[c[class_column]])
            if event.dict["key"] == pygame.K_COMMA:
                banc = not banc
        if event.type == pygame.MOUSEBUTTONDOWN and event.dict["button"] == pygame.BUTTON_LEFT:
            if pygame.mouse.get_pos()[0] < 400:
                y = pygame.mouse.get_pos()[1]
                selected = y//25
                for i, c in enumerate(class_colors.keys()):
                    if i == selected:
                        class_colors[c] = (255, 0, 0, 200)
                    else:
                        class_colors[c] = (0, 0, 255, 100)
                for n_id in drawer.coord_map.keys():
                    # c = fafb_class_dict[n_id]
                    # drawer.set_color(n_id, class_colors[c[2]])
                    c = class_dict[n_id]
                    drawer.set_color(n_id, class_colors[c[class_column]])
    
    screen.fill((0, 0, 0))

    coords = (400, 0) if banc else (0, 400)
    screen.blit(drawer.draw(), coords)
    for i, c in enumerate(class_colors.keys()):
        r, g, b = (class_colors[c][0], class_colors[c][1], class_colors[c][2])
        m = 255 - max(r, g, b)
        screen.blit(font.render(c + " " + str(class_counts[c]), True, (r + m, g + m, b + m)), (0, 25*i))


    screen.blit(font.render(f"{clock.get_fps():.0f} fps", True, "blue"), (0, HEIGHT-50))
    pygame.display.flip()

    clock.tick(60)
