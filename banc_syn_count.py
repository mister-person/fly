from collections import defaultdict
import numpy as np
import pandas as pd
import pygame
import random

import drawutils

datasets = ["banc", "fafb", "mbanc"]
current_dataset = "banc"

"""
path_comp = "data/banc_completeness.csv"
path_con = "data/banc_connectivity.parquet"
banc_comp_np = pd.read_csv(path_comp, index_col=0).to_numpy()
banc_con_np = pd.read_parquet(path_con).to_numpy()

path_comp = "Drosophila_brain_model/Completeness_783.csv"
path_con = "Drosophila_brain_model/Connectivity_783.parquet"
fafb_comp_np = pd.read_csv(path_comp, index_col=0).to_numpy()
fafb_con_np = pd.read_parquet(path_con).to_numpy()
"""

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
def randomize_colors(classes, class_colors):
    for c in classes:
        r = random.randrange(0, 255)
        g = random.randrange(0, 255)
        b = random.randrange(0, 255)
        s = r + g + b
        class_colors[c] = (r * 255 // s, g * 255 // s, b * 255 // s, 150)
        print(class_colors[c])

randomize_colors(fafb_classes, fafb_class_colors)
randomize_colors(banc_classes, banc_class_colors)

fafb_class_dict = {}
for n_data in fafb_class:
    fafb_class_dict[int(n_data[0])] = n_data

banc_class_dict = {}
for n_data in banc_class:
    banc_class_dict[int(n_data[0])] = n_data

print(list(fafb_class_dict.items())[:10])

pygame.init()
WIDTH, HEIGHT = (1500, 1200)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.Font(size=50)
drawer_fafb = drawutils.NeuronDrawer("../flywire/fafb_coordinates.csv", WIDTH, HEIGHT)
drawer_banc = drawutils.NeuronDrawer("../flywire/banc_coordinates.csv", WIDTH, HEIGHT)
# drawer_banc = drawutils.NeuronDrawer("../flywire/body-neurotransmitters-male-cns-v0.9.feather", WIDTH, HEIGHT)
'''
for n_id in drawer.coord_map.keys():
    c = fafb_class_dict[n_id]
    drawer.color_map[n_id] = class_colors[c[2]]
'''
fafb_class_counts = defaultdict(int)
banc_class_counts = defaultdict(int)
for n_data in drawer_fafb.coord_map.keys():
    c = fafb_class_dict[n_data]
    drawer_fafb.set_color(n_data, fafb_class_colors[c[2]])
    fafb_class_counts[c[2]] += 1
for n_data in drawer_banc.coord_map.keys():
    print(n_data)
    c = banc_class_dict[n_data]
    drawer_banc.set_color(n_data, banc_class_colors[c[10]])
    banc_class_counts[c[10]] += 1

print("done")

configs = {
    "banc": {
        "class_arr": pd.read_csv("../flywire/banc_neurons.csv").to_numpy().astype("str"),
        "coord_arr": pd.read_csv("../flywire/banc_coordinates.csv").to_numpy(),
        "class_col": 10,
    },
    "fafb": {
        "class_arr": pd.read_csv("../flywire/fafb_classification.csv").to_numpy().astype("str"),
        "coord_arr": pd.read_csv("../flywire/fafb_coordinates.csv").to_numpy(),
        "class_col": 2,
    },
    "mbanc": {
        "class_arr": pd.read_feather("../flywire/body-annotations-male-cns-v0.9-minconf-0.5.feather")[["bodyId", "superclass"]].astype({"superclass": str}).dropna().to_numpy(),
        "coord_arr": pd.read_feather("../flywire/body-annotations-male-cns-v0.9-minconf-0.5.feather")[["bodyId", "somaLocation"]].dropna().astype({"somaLocation": str}).to_numpy(),
        "class_col": 1,
    },
}

class_colorss = {}
class_dicts = {}
drawers = {}
class_counts = {}
columns = {}

for dataset_name, cfg in configs.items():
    class_col = cfg["class_col"]
    print("starting", dataset_name)

    class_arr = cfg["class_arr"]

    classes = np.unique(class_arr[:, class_col])
    print(dataset_name, "classes:")
    print(classes)
    print()

    class_dict = {}
    for n_data in class_arr:
        class_dict[int(n_data[0])] = n_data

    # randomize colors
    class_colorss[dataset_name] = {}
    for c in classes:
        r, g, b = (random.randrange(0, 255) for _ in range(3))
        s = r + g + b
        color = (r * 255 // s, g * 255 // s, b * 255 // s, 150)
        class_colorss[dataset_name][c] = color
        print(color)

    drawers[dataset_name] = drawutils.NeuronDrawer(cfg["coord_arr"], WIDTH, HEIGHT)
    class_counts[dataset_name] = defaultdict(int)
    for n_id in drawers[dataset_name].coord_map.keys():
        try:
            n_class = class_dict[n_id][class_col]
            drawers[dataset_name].set_color(n_id, class_colorss[dataset_name][n_class])
            class_counts[dataset_name][n_class] += 1
        except:
            if n_id < 10050:
                print("error key n_id", n_id)

    class_dicts[dataset_name] = class_dict
    columns[dataset_name] = class_col

'''
drawers = {"banc": drawer_banc, "fafb": drawer_fafb}
class_dicts = {"banc": banc_class_dict, "fafb": fafb_class_dict}
class_colors = {"banc": banc_class_colors, "fafb": fafb_class_colors}
columns = {"banc": 10, "fafb": 2}
class_counts = {"banc": banc_class_counts, "fafb": fafb_class_counts}
'''

while True:
    drawer: drawutils.NeuronDrawer = drawers[current_dataset]
    class_dict = class_dicts[current_dataset]
    class_colors = class_colorss[current_dataset]
    class_column = columns[current_dataset]
    class_count = class_counts[current_dataset]
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.dict["key"] == pygame.K_PERIOD:
                randomize_colors(class_count.keys(), class_colors)
                for n_data in drawer.coord_map.keys():
                    # c = class_dict[n_id]
                    # drawer.set_color(n_id, class_colors[c[2]])
                    c = class_dict[n_data]
                    drawer.set_color(n_data, class_colors[c[class_column]])
            if event.dict["key"] == pygame.K_COMMA:
                current_dataset = datasets[(datasets.index(current_dataset) + 1) % len(datasets)]
        if event.type == pygame.MOUSEBUTTONDOWN and event.dict["button"] == pygame.BUTTON_LEFT:
            if pygame.mouse.get_pos()[0] < 400:
                y = pygame.mouse.get_pos()[1]
                selected = y//25
                for i, c in enumerate(class_colors.keys()):
                    if i == selected:
                        class_colors[c] = (255, 0, 0, 200)
                    else:
                        class_colors[c] = (0, 0, 255, 100)
                for n_data in drawer.coord_map.keys():
                    # c = fafb_class_dict[n_id]
                    # drawer.set_color(n_id, class_colors[c[2]])
                    c = class_dict[n_data]
                    drawer.set_color(n_data, class_colors[c[class_column]])
    
    screen.fill((0, 0, 0))

    coords = {"banc": (400, 0), "fafb": (0, 400), "mbanc": (200, 0)}[current_dataset]
    screen.blit(drawer.draw(), coords)
    for i, c in enumerate(class_colors.keys()):
        r, g, b = (class_colors[c][0], class_colors[c][1], class_colors[c][2])
        m = 255 - max(r, g, b)
        screen.blit(font.render(c + " " + str(class_count[c]), True, (r + m, g + m, b + m)), (0, 25*i))


    screen.blit(font.render(f"{clock.get_fps():.0f} fps", True, "blue"), (0, HEIGHT-50))
    pygame.display.flip()

    clock.tick(60)
