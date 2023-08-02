import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

df = pd.read_csv('dataset.csv')

# 设置相同的随机种子
np.random.seed(42)

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

    def move(self, max_step):
        self.x += np.random.uniform(-max_step, max_step)
        self.y += np.random.uniform(-max_step, max_step)
        self.x = np.clip(self.x, 0, area_size[0])
        self.y = np.clip(self.y, 0, area_size[1])


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
            detected_residents[resident.name] = True
            self.energy_consume += 0.0092


class ArtificialBee:
    def __init__(self, positions):
        self.positions = positions
        self.fitness = 0


# 随机生成居民（位置、名字）
def generate_residents(num_residents, area_size):
    # 使用正态分布随机生成居民的初始位置
    x = df['x3'].values
    y = df['y3'].values

    residents = [Resident("resident_" + str(i), x[i], y[i]) for i in range(num_residents)]
    for resident in residents:
        detected_residents[resident.name] = resident.detected

    return residents


# 生成初始化解
def generate_initial_solutions(num_devices, area_size):
    solutions = []
    for _ in range(num_devices):
        positions = [(np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1]))] * num_devices
        solutions.append(ArtificialBee(positions))
    return solutions


# 评估
def evaluate_fitness(solution, residents, edge_devices):
    for i, device in enumerate(edge_devices):
        device.x, device.y = solution.positions[i]

    for device in edge_devices:
        for resident in residents:
            device.detect_resident(resident)

    fitness = sum([1 for resident in residents if resident.detected])
    return fitness


#
def scout_bee_phase(solutions, residents, edge_devices, area_size):
    for solution in solutions:
        if solution.fitness == 0:
            positions = [(np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1]))]
            solution.positions = positions


def employed_bee_phase(solutions, residents, edge_devices):
    for i, solution in enumerate(solutions):
        positions = solution.positions
        for j, position in enumerate(positions):
            neighbor_index = np.random.choice([k for k in range(len(solutions)) if k != i])
            neighbor_solution = solutions[neighbor_index]
            neighbor_positions = neighbor_solution.positions
            new_position = ((position[0] + neighbor_positions[j][0]) / 2, (position[1] + neighbor_positions[j][1]) / 2)

            fitness = evaluate_fitness(ArtificialBee(positions[:j] + [new_position] + positions[j + 1:]), residents,
                                       edge_devices)
            if fitness > solution.fitness:
                positions[j] = new_position

                distance = np.sqrt((new_position[0] - position[0]) ** 2 + (new_position[1] - position[1]) ** 2)
                energy = round(107.44 * (distance / 9) / 3600 - 124.86 * (distance / 9) / 3600, 4)
                edge_devices[i].energy_consume += energy

                solution.fitness = fitness


def onlooker_bee_phase(solutions, residents, edge_devices):
    total_fitness = sum([solution.fitness for solution in solutions])
    probabilities = [solution.fitness / total_fitness for solution in solutions]

    for i, solution in enumerate(solutions):
        if np.random.rand() < probabilities[i]:
            positions = solution.positions
            for j, position in enumerate(positions):
                neighbor_index = np.random.choice([k for k in range(len(solutions)) if k != i])
                neighbor_solution = solutions[neighbor_index]
                neighbor_positions = neighbor_solution.positions
                new_position = (
                    (position[0] + neighbor_positions[j][0]) / 2, (position[1] + neighbor_positions[j][1]) / 2)
                fitness = evaluate_fitness(ArtificialBee(positions[:j] + [new_position] + positions[j + 1:]), residents,
                                           edge_devices)
                if fitness > solution.fitness:
                    positions[j] = new_position

                    distance = np.sqrt((new_position[0] - position[0]) ** 2 + (new_position[1] - position[1]) ** 2)
                    energy = round(107.44 * (distance / 9) / 3600 - 124.86 * (distance / 9) / 3600, 4)
                    edge_devices[i].energy_consume += energy

                    solution.fitness = fitness


# 优化边缘设备位置
def optimize_edge_device_positions(num_residents, area_size, num_devices, radius, max_battery, warning_battery,
                                   max_iterations):
    # 生成居民对象，放入列表中
    residents = generate_residents(num_residents, area_size)
    # 生成边缘设备对象，放入列表中
    edge_devices = [EdgeDevice(0, 0, radius, max_battery, warning_battery) for _ in range(num_devices)]

    # 蜂群算法所需的参数
    solutions = generate_initial_solutions(num_devices, area_size)  # 设备数量、矩形大小 （初始解）
    best_solution = None
    best_fitness = 0

    # for _ in range(max_iterations):
    while False in detected_residents.values():

        for resident in residents:
            if not resident.detected:
                resident.move(max_resident_step)

        # 以下3个函数都是执行蜂群算法的流程
        # 该函数表示雇佣蜂的行为。雇佣蜂通过搜索其局部邻域来利用已知解。
        # 对于每个解，该函数遍历其位置，并通过与随机选择的相邻解的对应位置进行平均，计算出一个新位置。评估新位置的适应度，如果适应度优于当前适应度，则更新位置。
        employed_bee_phase(solutions, residents, edge_devices)
        # 该函数表示侦察蜂的行为。侦察蜂根据解的适应度值以概率选择解，并执行类似于雇佣蜂的探索。选择概率与每个解的适应度成比例。该函数遍历解和其位置，以概率选择一个相邻解，计算一个新位置，评估其适应度，并在适应度改善时更新位置。
        onlooker_bee_phase(solutions, residents, edge_devices)
        # 该函数负责初始化侦察蜂的位置。侦察蜂通过随机探索搜索空间中的新解来进行探索。在这个函数中，如果一个解的适应度值为0，则生成一个在给定区域大小内的随机位置，并赋给该解。
        scout_bee_phase(solutions, residents, edge_devices, area_size)

        for solution in solutions:
            fitness = evaluate_fitness(solution, residents, edge_devices)
            if fitness > best_fitness:
                best_solution = solution
                best_fitness = fitness

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
max_iterations = 30
max_resident_step = 8

# 优化
start_time = time.time()
residents, edge_devices, best_solution = optimize_edge_device_positions(num_residents, area_size, num_devices, radius,
                                                                        max_battery, warning_battery, max_iterations)

# 可视化
visualize(residents, edge_devices, best_solution, area_size)

sum_energy = 0
for i in range(num_devices):
    sum_energy += edge_devices[i].energy_consume
end_time = time.time()

print("sum_energy")
print(sum_energy)
print("充电次数:")
print(charge_num_count)
print("运行时间:")
print(end_time - start_time)


