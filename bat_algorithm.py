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
            self.battery = self.max_battery
            charge_num_count += 1

        if distance <= self.radius and not resident.detected:
            self.battery -= 1
            resident.detected = True
            # print(resident.name)  # 打印已检测居民name
            detected_residents[resident.name] = True
            self.energy_consume += 0.0092


class Bat:
    def __init__(self, position):
        self.position = position
        self.velocity = np.zeros(2)
        self.frequency = 0
        self.pulse_rate = 0
        self.fitness = 0


# 随机生成居民（位置、名字）
def generate_residents(num_residents, area_size):
    # 使用正态分布随机生成居民的初始位置
    x = df['x1'].values
    y = df['y1'].values

    residents = [Resident("resident_" + str(i), x[i], y[i]) for i in range(num_residents)]
    for resident in residents:
        detected_residents[resident.name] = resident.detected

    return residents


# 生成初始化解
def generate_initial_solutions(num_devices, area_size):
    solutions = []
    for _ in range(num_devices):
        position = np.array([np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1])])
        solutions.append(Bat(position))
    return solutions


# 评估
def evaluate_fitness(solution, residents, edge_devices):
    for i, device in enumerate(edge_devices):
        device.x, device.y = solution.position

    for device in edge_devices:
        for resident in residents:
            device.detect_resident(resident)

    fitness = sum([1 for resident in residents if resident.detected])
    return fitness


# 更新蝙蝠的位置
def update_bat_position(bat, best_solution):
    bat.velocity += (bat.position - best_solution.position) * bat.frequency
    new_position = bat.position + bat.velocity
    new_position = np.clip(new_position, [0, 0], area_size)
    bat.position = new_position
    return bat.position


# 蝙蝠搜索行为
def bat_search(solutions, residents, edge_devices, area_size):
    alpha = 0.9
    gamma = 0.9
    f_min = 0
    f_max = max(residents, key=lambda resident: np.sqrt(resident.x ** 2 + resident.y ** 2))
    f_max = np.sqrt(f_max.x ** 2 + f_max.y ** 2)

    best_solution = max(solutions, key=lambda solution: solution.fitness)
    for i, bat in enumerate(solutions):
        bat.frequency = f_min + (f_max - f_min) * np.random.rand()
        bat.pulse_rate = np.random.rand()
        position = bat.position

        new_position = update_bat_position(bat, best_solution)

        if np.random.rand() > bat.pulse_rate:
            bat.position += alpha * (np.random.rand(2) - 0.5)
            bat.position = np.clip(bat.position, [0, 0], area_size)

        fitness = evaluate_fitness(bat, residents, edge_devices)
        if fitness > bat.fitness:
            bat.fitness = fitness
            distance = np.sqrt((new_position[0] - position[0]) ** 2 + (new_position[1] - position[1]) ** 2)
            energy = round(107.44 * (distance / 9) / 3600 - 124.86 * (distance / 9) / 3600, 4)
            edge_devices[i].energy_consume += energy

        if np.random.rand() < bat.pulse_rate and bat.fitness > best_solution.fitness:
            best_solution = bat

        bat.frequency *= gamma


# 优化边缘设备位置
def optimize_edge_device_positions(num_residents, area_size, num_devices, radius, max_battery, warning_battery,):
    # 生成居民对象，放入列表中
    residents = generate_residents(num_residents, area_size)
    # 生成边缘设备对象，放入列表中
    edge_devices = [EdgeDevice(0, 0, radius, max_battery, warning_battery) for _ in range(num_devices)]

    # 蝙蝠算法所需的参数
    solutions = generate_initial_solutions(num_devices, area_size)  # 设备数量、矩形大小 （初始解）
    best_solution = None
    best_fitness = 0
    i = 0
    while False in detected_residents.values():
        i += 1
        for resident in residents:
            if not resident.detected:
                resident.move(max_resident_step)

        bat_search(solutions, residents, edge_devices, area_size)

        for solution in solutions:
            fitness = evaluate_fitness(solution, residents, edge_devices)
            if fitness >= best_fitness:
                best_solution = solution
                best_fitness = fitness
    print(f"迭代次数：{i}")
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

    ax.plot(best_solution.position[0], best_solution.position[1], 'g^', markersize=10)

    ax.annotate(f'Best Fitness: {best_solution.fitness}',
                (best_solution.position[0], best_solution.position[1] - 10), ha='center')

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

# 优化
start_time = time.time()
residents, edge_devices, best_solution = optimize_edge_device_positions(num_residents, area_size, num_devices, radius,
                                                                        max_battery, warning_battery)

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
