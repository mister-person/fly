from collections import defaultdict
import time
from types import NoneType
from typing import Dict, Iterable, Tuple
import pandas as pd
import numpy as np
import pygame
import random
import pyopencl
import pyopencl.array as pyopencl_array
from pyvkfft.fft import fftn
import pyvkfft
import os

if 'PYOPENCL_CTX' in os.environ:
    ctx = pyopencl.create_some_context()
else:
    ctx = None
    # Find the first OpenCL GPU available and use it, unless
    for p in pyopencl.get_platforms():
        for d in p.get_devices():
            if d.type & pyopencl.device_type.GPU == 0:
                continue
            print("Selected device: ", d.name)
            ctx = pyopencl.Context(devices=(d,))
            break
        if ctx is not None:
            break
    if ctx == None:
        exit()
cq = pyopencl.CommandQueue(ctx)

class NeuronDrawer():
    def __init__(self, neuron_coords_path, width: int, height: int):
        neuron_coords = pd.read_csv(neuron_coords_path).to_numpy()
        print(neuron_coords[:30])

        self.dirty = True
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)

        self.max_x = self.max_y = self.max_z = 0
        self.min_x = self.min_y = self.min_z = 100000000
        self.coord_map = defaultdict(list)
        self.color_map: dict[int, tuple[int, int, int, int]] = {}
        for n in neuron_coords:
            [x, y, z] = n[1][1:-1].strip().replace("  ", " ").replace("  ", " ").split(" ")
            x, y, z = (int(x), int(y), int(z))
            self.coord_map[n[0]].append((x, y, z))
            self.color_map[n[0]] = (0, 0, 255, 100)

            if x > self.max_x:
                self.max_x = x
            if x < self.min_x:
                self.min_x = x
            if y > self.max_y:
                self.max_y = y
            if y < self.min_y:
                self.min_y = y
            if z > self.max_z:
                self.max_z = z
            if z < self.min_z:
                self.min_z = z

    def to_screen_coords(self, pos):
        (x, y, _) = pos
        scale = max((self.max_x - self.min_x) / self.width, (self.max_y - self.min_y) / self.height)
        out_x = (x - self.min_x) / scale
        out_y = (y - self.min_y) / scale
        return (clamp(int(out_x), ub=self.width - 1), clamp(int(out_y), ub=self.height - 1))

    def set_color(self, n_id, color):
        self.color_map[n_id] = color
        self.dirty = True

    def draw(self):
        if self.surface is None:
            self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        elif not self.dirty:
            return self.surface

        self.surface.fill((0, 0, 0, 0))
        for id, positions in self.coord_map.items():
            for pos in positions:
                #pygame.draw.circle(screen, "blue", to_screen_coords(pos), 1)
                screen_pos = self.to_screen_coords(pos)
                existing_color = self.surface.get_at(screen_pos)
                self.surface.set_at(screen_pos, addc(existing_color, self.color_map[id]))

        self.dirty = False
        return self.surface

class SpikeDrawer():
    def __init__(self, width, height):
        self.surface1 = pygame.Surface((width, height), pygame.SRCALPHA)
        self.surface2 = pygame.Surface((width, height), pygame.SRCALPHA)
        self.data = []
        self.end_time = .01
        self.time_size = .01
        self.neurons = []
        self.width = width
        self.height = height

        self.color_map: dict[int, tuple[int, int, int]] = defaultdict(lambda: (255, 255, 255))

        self.unit_height = 2

        self.max_neurons = self.height // self.unit_height + 1

        self.reset_surfaces()

    @property
    def start_time(self):
        return self.end_time - self.time_size * 2

    @property
    def surface_split(self):
        return self.end_time - self.time_size 

    def add_points(self, spikes: Iterable[tuple[float, int]]):
        self.data.extend(spikes)
        for spike in spikes:
            self.draw_spike(spike)

    def draw_spike(self, spike, add_spike = True):
        if spike[0] < self.start_time or spike[0] > self.end_time:
            return
        if spike[1] not in self.neurons:
            if add_spike and len(self.neurons) < self.max_neurons:
                self.neurons.append(spike[1])
                if spike[1] not in self.color_map:
                    self.color_map[spike[1]] = random_color()
            else:
                return
        if spike[0] < self.surface_split:
            surface = self.surface1
            time = spike[0] - self.start_time
            x = time / (self.surface_split - self.start_time) * self.width
        else:
            surface = self.surface2
            time = spike[0] - self.surface_split
            x = time / (self.end_time - self.surface_split) * self.width

        index = self.neurons.index(spike[1])
        h = self.unit_height
        if h < 4:
            bottom = h - 1
        else:
            bottom = h - 2
        pygame.draw.line(surface, self.color_map[spike[1]], (x, index * h), (x, index * h + bottom))

    def reset_surfaces(self):
        self.surface1.fill((0, 255, 0, 5))
        self.surface2.fill((0, 0, 255, 5))
        for i, n in enumerate(self.neurons):
            color = self.color_map[n]
            color = (color[0], color[1], color[2], 30)
            rect = (0, i*self.unit_height, self.width, self.unit_height)
            pygame.draw.rect(self.surface1, color, rect)
            pygame.draw.rect(self.surface2, color, rect)
        for spike in self.data:
            self.draw_spike(spike)

    def swap_surfaces(self):
        temp = self.surface2
        self.surface2 = self.surface1
        self.surface1 = temp
        self.end_time += self.time_size
        
        self.reset_surfaces()

    def draw(self, surface: pygame.Surface, pos, now):
        while now > self.end_time:
            self.swap_surfaces()
        while now < self.surface_split:
            self.end_time -= self.time_size
            self.reset_surfaces()
        xoffset1 = ((now - self.surface_split) / (self.end_time - self.surface_split)) * self.width + self.width
        xoffset2 = ((now - self.surface_split) / (self.end_time - self.surface_split)) * self.width
        area1 = (xoffset1-self.width, 0, self.width, self.height)
        area2 = (0, 0, xoffset2, self.height)
        # area2 = (0, 0, self.width, self.height)
        surface.blit(self.surface1, (pos[0], pos[1]), area1)
        surface.blit(self.surface2, (pos[0] + self.width - xoffset2, pos[1]), area2)

        # pygame.draw.line(surface, "red", (pos[0], pos[1]), (pos[0]+self.width, pos[1]), 3)
        # pygame.draw.line(surface, "red", (pos[0]+self.width, pos[1]), (pos[0]+self.width, pos[1]+self.height), 3)
        # pygame.draw.line(surface, "red", (pos[0]+self.width, pos[1]+self.height), (pos[0], pos[1]+self.height), 3)
        # pygame.draw.line(surface, "red", (pos[0], pos[1]+self.height), (pos[0], pos[1]), 3)
        # surface.blit(self.surface2, (pos[0] + self.surface2.width, pos[1]))

class Slider:
    
    def __init__(self, pos: tuple[int, int], name: str = "", size = (100, 30), initial_value = 0, value_range = (-10, 10), on_changed = lambda _: None):
        # self.pos = pos
        # self.size = size
        self.rect: pygame.Rect = pygame.Rect(pos[0], pos[1], size[0], size[1])
        self.value = initial_value
        self.name = name
        self.range = value_range
        self.font = pygame.font.Font(size=size[1] - 2)
        self.mouse_down = False
        self.on_changed = on_changed

        self.scale = .1

    def draw(self, surface: pygame.Surface):
        x, y = self.rect.topleft
        xe, ye = (x + self.rect.width, y + self.rect.height)
        pygame.draw.line(surface, "white", (x, y), (x, ye))
        pygame.draw.line(surface, "white", (x, ye), (xe, ye))
        pygame.draw.line(surface, "white", (xe, ye), (xe, y))
        pygame.draw.line(surface, "white", (xe, y), (x, y))
        surface.blit(self.font.render(f"{self.name}: {self.value:.01f}", True, "white"), (x + 1, y + 1))

    def drag(self, rel):
        if self.mouse_down:
            self.value += rel[0] * self.scale
            self.on_changed(self.value)

    def click(self, down: bool):
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos) and down:
            self.mouse_down = True
        elif not down:
            self.mouse_down = False

class FFTDrawer():
    def __init__(self, width, height):
        self.data = []
        self.width = width
        self.height = height

        self.fft_start = 0
        self.fft_end = 0
        self.neurons = {}
        self.dense_cache = np.array([])

        self.last_time = 0
        self.update_freq = .5
        self.cache = np.array([])
        self.font = pygame.font.Font(size=20)

    def add_points(self, spikes: Iterable[tuple[float, int]]):
        # self.data.extend(spikes)
        for spike in spikes:
            if spike[1] not in self.neurons:
                self.neurons[spike[1]] = len(self.neurons)

            self.data.append(spike)
            if self.data[-1][0] > self.data[self.fft_end][0]:
                self.fft_end = len(self.data) - 1

    def reset_fft(self):
        self.fft_start = len(self.data)
        
    def get_fft(self, neuron = None):
        times = [d[0] for d in self.data if neuron == None or d[1] == neuron]
        dt = .0001
        if len(times) > self.fft_start:
            start_time = self.data[self.fft_start][0]
            end_time = self.data[self.fft_end][0]
            dense = np.zeros(int((end_time - start_time) / dt) + 1)
            for time in times[self.fft_start:]:
                dense[int((time - start_time) / dt)] = 1
            time_fft = np.fft.fft(dense)
            return np.abs(time_fft)
        return np.array([])

    def get_all_fft(self):
        if len(self.data) > self.fft_start:
            start_time = self.data[self.fft_start][0]
            end_time = self.data[self.fft_end][0]
            dt = .0001

            nptimes = np.array([[self.neurons[d[1]], int((d[0] - start_time) / dt)] for d in self.data[self.fft_start:]])
            neuron_d = len(self.neurons)
            time_d = int((end_time - start_time) / dt) + 1
            dense = np.zeros((neuron_d, time_d))

            dense[nptimes[:, 0], nptimes[:, 1]] = 1
            # before = time.monotonic()
            time_fft = np.fft.fft(dense).sum(0)
            # print("fft took", time.monotonic() - before)

            ''' opencl fft stuff
            before = time.monotonic()
            ocl_dense = pyopencl_array.to_device(cq, dense.astype(np.csingle))
            print("array to cl took", time.monotonic() - before)
            before = time.monotonic()
            ans: pyopencl_array.Array = fftn(ocl_dense, ndim=1) # type: ignore
            # print("shape 1", ans.shape)
            # ans = pyopencl_array.sum(ans)
            # print("shape 2", ans.shape)
            print("shape 2", ans.dtype)
            np_ans = ans.get().sum(0)
            print("cl fft took", time.monotonic() - before)
            '''

            return np.abs(time_fft)
        return np.array([])

    def update_cache(self):
        # fft = self.get_all_fft()
        fft = self.get_fft()
        self.last_time = time.monotonic()
        self.cache = fft
        return fft

    def peaks(self, n):
        top10 = self.cache[:n].argsort()[-10:]
        used = set()
        top10_merged = []
        for x in top10:
            if x in used:
                continue
            if x+1 in top10:
                weight = self.cache[x] / (self.cache[x] + self.cache[x+1])
                top10_merged.append((x + weight, (self.cache[x] + self.cache[x+1])/2))
                used.add(x+1)
            elif x-1 in top10:
                weight = self.cache[x] / (self.cache[x] + self.cache[x-1])
                top10_merged.append((x - weight, (self.cache[x] + self.cache[x-1])/2))
                used.add(x-1)
            else:
                top10_merged.append((x, self.cache[x]))
        return top10_merged

    def draw(self, surface: pygame.Surface):
        if time.monotonic() - self.last_time < self.update_freq:
            fft = self.cache
        else:
            fft = self.update_cache()

        l = len(fft)
        dt = .0001
        hz_min = 9
        hz_max = 16
        color_rate_min = l / (1/hz_min / dt)
        color_rate_max = l / (1/hz_max / dt)
        draw_hz_max = 500
        draw_hz_max = l / (1/draw_hz_max / dt)

        peaks = self.peaks(int(draw_hz_max / 4))
        for i, peak in peaks:
            x = i/draw_hz_max
            y = peak/l
            screen_x = x * self.width
            screen_y = self.height - y * 10000
            freq = i/(dt*l)
            surface.blit(self.font.render(f"{freq:.2f}", True, "yellow"), (screen_x, screen_y))

        last_point = None
        for i, item in enumerate(fft):
            if i > draw_hz_max:
                break
            x = i/draw_hz_max
            y = item/l
            screen_x = x * self.width
            screen_y = self.height - y * 10000
            color = "white"
            if i > color_rate_min and i < color_rate_max:
                color = (255, 128, 0)
            if last_point != None:
                # surface.set_at((screen_x, screen_y), color)
                # print("drawing line", last_point, (screen_x, screen_y))
                pygame.draw.line(surface, color, last_point, (screen_x, screen_y))
            last_point = (screen_x, screen_y)


def clamp(n, lb=0, ub=255):
    return max(lb, min(ub, n))

def addc(color1, color2):
    if color1[3] == 255:
        return color1
    if color2[3] == 255:
        return color2
    r = clamp(color1[0] + color2[0])
    g = clamp(color1[1] + color2[1])
    b = clamp(color1[2] + color2[2])
    a = clamp(color1[3] + color2[3], ub=254)
    return (r, g, b, a)

def random_color():
    r = random.randrange(0, 255)
    g = random.randrange(0, 255)
    b = random.randrange(0, 255)
    s = max(r, g, b)
    return (r + 255 - s, g + 255 - s, b + 255 - s)

