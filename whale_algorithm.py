# -*- encoding: utf-8 -*-
# @Author: Kitty
# @ModuleName: whale_algorithm.py
# @Time: 2023-06-14 19:59

import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import sys

np.random.seed(42)
# 将检测到的居民装入字典中
detected_residents = {}
# 统计充电次数
charge_num_count = 0
# 居民对象，有名称、坐标、可以移动
class Resident:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
        self.detected = False

    def move(self, max_step):
        self.x += np.random.uniform(-max_step, max_step)
        self.y += np.random.uniform(-max_step, max_step)
        # self.x += 1
        # self.y += 1
        self.x = np.clip(self.x, 0, area_size[0])
        self.y = np.clip(self.y, 0, area_size[1])


# 边缘设备对象，有位置，检测半径，电池容量
class EdgeDevice:
    def __init__(self, x, y, radius, max_battery, warning_battery):
        self.x = x
        self.y = y
        self.radius = radius
        self.max_battery = max_battery
        self.battery = max_battery
        self.warning_battery = warning_battery

    def detect_resident(self, resident):
        global charge_num_count
        distance = np.sqrt((resident.x - self.x) ** 2 + (resident.y - self.y) ** 2)

        if self.battery <= self.warning_battery:
            self.battery = max_battery
            charge_num_count += 1

        if distance <= self.radius and not resident.detected:
            self.battery -= 1
            resident.detected = True
            detected_residents[resident.name] = True


class Whale:
    def __init__(self, positions):
        self.positions = positions
        self.fitness = 0


def generate_initial_whales(num_whales, area_size):
    whales = []
    for _ in range(num_whales):
        positions = [(np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1])) for _ in
                     range(num_devices)]
        whales.append(Whale(positions))
    return whales


def evaluate_whale_fitness(whale, residents, edge_devices):
    for i, device in enumerate(edge_devices):
        device.x, device.y = whale.positions[i]

    for device in edge_devices:
        for resident in residents:
            device.detect_resident(resident)

    fitness = sum([1 for resident in residents if resident.detected])
    return fitness


# 生成居民对象
def generate_residents(num_residents, area_size):
    # 使用正态分布随机生成居民的初始位置
    df = pd.read_csv('dataset.csv')
    # 提取两列数据并转换为NumPy数组
    x = df['x1'].values
    y = df['y1'].values
    residents = [Resident("resident_" + str(i), x[i], y[i]) for i in range(num_residents)]
    for resident in residents:
        detected_residents[resident.name] = resident.detected

    return residents


def optimize_edge_device_positions_with_whale_algorithm(num_residents, area_size, num_devices, radius, max_battery,
                                                        warning_battery,
                                                        max_iterations):
    residents = generate_residents(num_residents, area_size)
    edge_devices = [EdgeDevice(0, 0, radius, max_battery, warning_battery) for _ in range(num_devices)]

    whales = generate_initial_whales(num_devices, area_size)

    best_whale = None
    best_fitness = 0

    while False in detected_residents.values():
        for resident in residents:
            if not resident.detected:
                resident.move(max_resident_step)

        for i, whale in enumerate(whales):
            positions = whale.positions
            for j, position in enumerate(positions):
                neighbor_index = np.random.choice([k for k in range(len(whales)) if k != i])
                neighbor_whale = whales[neighbor_index]
                neighbor_positions = neighbor_whale.positions
                new_position = (
                    (position[0] + neighbor_positions[j][0]) / 2, (position[1] + neighbor_positions[j][1]) / 2)
                fitness = evaluate_whale_fitness(Whale(positions[:j] + [new_position] + positions[j + 1:]), residents,
                                                 edge_devices)
                if fitness > whale.fitness:
                    positions[j] = new_position
                    whale.fitness = fitness

        for whale in whales:
            fitness = evaluate_whale_fitness(whale, residents, edge_devices)
            if fitness > best_fitness:
                best_whale = whale
                best_fitness = fitness

    return residents, edge_devices, best_whale


def visualize_whales(residents, edge_devices, best_whale, area_size):
    fig, ax = plt.subplots()
    ax.set_xlim([0, area_size[0]])
    ax.set_ylim([0, area_size[1]])

    for device in edge_devices:
        circle = plt.Circle((device.x, device.y), device.radius, color='r', fill=False)
        ax.add_artist(circle)
        ax.annotate(f'Battery: {device.battery}', (device.x, device.y + device.radius), ha='center')

    for resident in residents:
        if resident.detected:
            ax.plot(resident.x, resident.y, 'bo')
        else:
            ax.plot(resident.x, resident.y, 'ko')

    for position in best_whale.positions:
        ax.plot(position[0], position[1], 'g^', markersize=10)

    ax.annotate(f'Best Fitness: {best_whale.fitness}',
                (best_whale.positions[0][0], best_whale.positions[0][1] - 10), ha='center')
    print(f"最好路径的适应度{best_whale.fitness}")
    plt.show()
    program_name = sys.argv[0]
    image_name = program_name.replace('.py', '.png')
    plt.savefig(image_name)


if __name__ == "__main__":
    num_residents = 1000
    area_size = (100, 100)
    num_devices = 5
    radius = 10
    max_battery = 100
    warning_battery = 20
    max_iterations = 30
    max_resident_step = 8
    # 调用鲸鱼算法优化
    start_time = time.time()
    residents, edge_devices, best_whale = optimize_edge_device_positions_with_whale_algorithm(num_residents, area_size,
                                                                                              num_devices, radius,
                                                                                              max_battery,
                                                                                              warning_battery,
                                                                                              max_iterations)
    end_time = time.time()
    print(f"充电次数：{charge_num_count}")
    print(f"运行时间：{end_time - start_time}")
    # 可视化鲸鱼算法结果
    visualize_whales(residents, edge_devices, best_whale, area_size)
