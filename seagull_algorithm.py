# -*- coding: UTF-8 -*-
# @Author: Kitty
# @File：seagull_algorithm.py
# @Date：2023-08-01 21:07


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


np.random.seed(42)
df = pd.read_csv("dataset.csv")

# 将检测到的居民装入字典中
detected_residents = {}
# 统计充电次数
charge_num_count = 0


class Resident:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
        self.detected = False

    def move(self, max_step):   # 对象每次移动的最大步长
        self.x += np.random.uniform(-max_step, max_step)
        self.y += np.random.uniform(-max_step, max_step)  # 对象的x坐标（self.x）和y坐标（self.y）分别增加一个在范围[-max_step, max_step]内的随机值。
        self.x = np.clip(self.x, 0, area_size[0])
        self.y = np.clip(self.y, 0, area_size[1])  # 将对象的位置限制在给定的区域内

class EdgeDevice:
    def __init__(self, x, y, radius, max_battery, warning_battery):
        self.x = x
        self.y = y
        self.radius = radius
        self.max_battery = max_battery
        self.battery = max_battery
        self.warning_battery = warning_battery
        self.energy_consume = 2996.64

    def detect_resident(self, resident):
        global charge_num_count
        distance = np.sqrt((resident.x - self.x) ** 2 + (resident.y - self.y) ** 2)

        if self.battery <= self.warning_battery:
            self.battery = max_battery
            charge_num_count += 1

        if distance <= self.radius and not resident.detected:
            self.battery -= 1
            resident.detected = True
            # print(resident.name)  # 打印已检测居民name
            detected_residents[resident.name] = True   #用来跟踪已被检测到的居民和充电次数的操作
            self.energy_consume += 0.0092



class Seagull:
    def __init__(self, positions):
        self.positions = positions
        self.fitness = 0


def generate_residents(num_residents):
    # 提取两列数据并转换为NumPy数组
    x = df['x3'].values
    y = df['y3'].values
    residents = [Resident("resident_" + str(i), x[i], y[i]) for i in range(num_residents)]
    for resident in residents:
        detected_residents[resident.name] = resident.detected

    return residents


def generate_initial_solutions(num_devices, area_size):
    solutions = []
    for _ in range(num_devices):
        positions = [(np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1]))] * num_devices
        solutions.append(Seagull(positions))
    return solutions


def evaluate_fitness(solution, residents, edge_devices):
    for i, device in enumerate(edge_devices):
        device.x, device.y = solution.positions[i]

    for device in edge_devices:
        for resident in residents:
            device.detect_resident(resident)

    fitness = sum([1 for resident in residents if resident.detected])
    return fitness


def update_seagull_positions(solution, residents, edge_devices, max_seagull_step):
    positions = solution.positions

    for i, position in enumerate(positions):
        resident_covered = False
        for _ in range(5):
            new_position = (position[0] + np.random.uniform(-max_seagull_step, max_seagull_step),
                            position[1] + np.random.uniform(-max_seagull_step, max_seagull_step))

            old_fitness = evaluate_fitness(solution, residents, edge_devices)
            positions[i] = new_position
            distance = np.sqrt((new_position[0] - position[0]) ** 2 + (new_position[1] - position[1]) ** 2)
            energy = round(107.44 * (distance / 9) / 3600 - 124.86 * (distance / 9) / 3600, 4)
            edge_devices[i].energy_consume += energy
            new_fitness = evaluate_fitness(solution, residents, edge_devices)

            if new_fitness > old_fitness:
                resident_covered = True
                break

        if not resident_covered:
            positions[i] = position

    solution.positions = positions


def seagull_algorithm(solutions, residents, edge_devices,max_iterations, max_seagull_step):
    best_solution = None
    best_fitness = 0
    i = 0
    while False in detected_residents.values():
        i += 1
        for _ in range(max_iterations):
            for resident in residents:
                if not resident.detected:
                    resident.move(max_resident_step)
            for solution in solutions:
                fitness = evaluate_fitness(solution, residents, edge_devices)
                if fitness >= best_fitness:

                    best_solution = solution
                    best_fitness = fitness

            for solution in solutions:
                update_seagull_positions(solution, residents, edge_devices, max_seagull_step)

            # 更新最佳解的适应度值
            best_solution.fitness = best_fitness

    sum_energy = 0
    for i in range(num_devices):
        sum_energy += edge_devices[i].energy_consume
    print("sum_energy")
    print(sum_energy)
    print("迭代次数：")
    print(i)
    return best_solution



def optimize_edge_device_positions(num_residents, area_size, num_devices, radius, max_battery, warning_battery,
                                   max_iterations, max_seagull_step):
    # 生成居民对象，放入列表中
    residents = generate_residents(num_residents)
    # 生成边缘设备对象，放入列表中
    edge_devices = [EdgeDevice(0, 0, radius, max_battery, warning_battery) for _ in range(num_devices)]

    # 海鸥算法所需的参数
    solutions = generate_initial_solutions(num_devices, area_size)

    best_solution = seagull_algorithm(solutions, residents, edge_devices, max_iterations, max_seagull_step)

    return residents, edge_devices, best_solution


def visualize(residents, edge_devices, best_solution, area_size):
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

    for position in best_solution.positions:
        ax.plot(position[0], position[1], 'g^', markersize=10)

    ax.annotate(f'Best Fitness: {best_solution.fitness}',
                (best_solution.positions[0][0], best_solution.positions[0][1] - 10), ha='center')

    plt.show()
    print("最佳路径适应度:")
    print(best_solution.fitness)


# 参数设置
num_residents = 1000
area_size = (100, 100)
num_devices = 5
radius = 10
max_battery = 100
warning_battery = 20
max_resident_step = 8

max_seagull_step = 5

# 优化
start_time = time.time()
residents, edge_devices, best_solution = optimize_edge_device_positions(num_residents, area_size, num_devices, radius,
                                                                        max_battery, warning_battery,
                                                                        max_resident_step, max_seagull_step)

end_time = time.time()
print("充电次数:")
print(charge_num_count)
print("运行时间:")
print(end_time - start_time)
# 可视化
visualize(residents, edge_devices, best_solution, area_size)


