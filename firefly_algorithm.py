import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
# 将检测到的居民装入字典中
detected_residents = {}
# 统计充电次数
charge_num_count = 0
df2 = pd.read_csv('dataset.csv')

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

    def detect_resident(self, resident):
        global charge_num_count
        distance = np.sqrt((resident.x - self.x) ** 2 + (resident.y - self.y) ** 2)

        if self.battery <= self.warning_battery:
            self.battery = self.max_battery
            charge_num_count += 1

        if distance <= self.radius and not resident.detected:
            self.battery -= 1
            resident.detected = True
            print(resident.name) #打印已检测居民name
            detected_residents[resident.name] = True



class Firefly:
    def __init__(self, positions):
        self.positions = positions
        self.fitness = 0


def generate_residents(num_residents, area_size):
    # 使用正态分布随机生成居民的初始位置
    #x = np.random.normal(area_size[0] / 2, area_size[0] / 6, num_residents)
    #y = np.random.normal(area_size[0] / 2, area_size[0] / 6, num_residents)
    # 提取两列数据并转换为NumPy数组
    x = np.array(df2['x3'].values)

    y = np.array(df2['y3'].values)


    residents = [Resident("resident_" + str(i), x[i], y[i]) for i in range(num_residents)]

    for resident in residents:
        detected_residents[resident.name] = resident.detected

    return residents


def generate_initial_solutions(num_devices, area_size):
    solutions = []
    for _ in range(num_devices):
        positions = [(np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1]))] * num_devices
        solutions.append(Firefly(positions))
    return solutions


def evaluate_fitness(solution, residents, edge_devices):
    for i, device in enumerate(edge_devices):
        device.x, device.y = solution.positions[i]

    for device in edge_devices:
        for resident in residents:
            device.detect_resident(resident)

    fitness = sum([1 for resident in residents if resident.detected])
    return fitness


def scout_firefly_phase(solutions, residents, edge_devices, area_size):
    for solution in solutions:
        if solution.fitness == 0:
            positions = [(np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1]))]
            solution.positions = positions


def firefly_bee_phase(solutions, residents, edge_devices):
    for i, solution in enumerate(solutions):
        positions = solution.positions
        for j, position in enumerate(positions):
            for k in range(len(solutions)):
                if k != i:
                    neighbor_solution = solutions[k]
                    neighbor_positions = neighbor_solution.positions

                    # 计算两个萤火虫之间的距离
                    distance = np.sqrt((position[0] - neighbor_positions[j][0]) ** 2 + (position[1] - neighbor_positions[j][1]) ** 2)

                    # 根据距离调整亮度值
                    brightness = solution.fitness / (distance ** 2)

                    # 更新位置时考虑亮度值和距离的影响
                    new_position = (
                        position[0] + brightness * (neighbor_positions[j][0] - position[0]) + np.random.uniform(-0.5, 0.5),
                        position[1] + brightness * (neighbor_positions[j][1] - position[1]) + np.random.uniform(-0.5, 0.5)
                    )
                    #亮度值表示萤火虫的亮度或吸引力，较高的亮度值意味着较强的吸引力，较远的距离会降低亮度值的影响。此外，我们还添加了一个随机扰动项，以增加搜索的多样性。

                    # 检查新位置是否超出边界，若超出则将其限制在边界内
                    new_position = (
                        np.clip(new_position[0], 0, area_size[0]),
                        np.clip(new_position[1], 0, area_size[1])
                    )

                    # 计算使用新位置后的适应度
                    fitness = evaluate_fitness(Firefly(positions[:j] + [new_position] + positions[j + 1:]), residents, edge_devices)

                    # 判断是否接受新位置
                    if fitness > solution.fitness:
                        positions[j] = new_position
                        solution.fitness = fitness

def optimize_edge_device_positions(num_residents, area_size, num_devices, radius, max_battery, warning_battery):
    # 生成居民对象，放入列表中
    residents = generate_residents(num_residents, area_size)
    # 生成边缘设备对象，放入列表中
    edge_devices = [EdgeDevice(0, 0, radius, max_battery, warning_battery) for _ in range(num_devices)]



    # 萤火虫算法所需的参数
    solutions = generate_initial_solutions(num_devices, area_size)
    best_solution = None
    best_fitness = 0
    i = 0
    while False in detected_residents.values():
        i += 1
        for resident in residents:
            if not resident.detected:
                resident.move(max_resident_step)

            # 萤火虫算法的流程
        firefly_bee_phase(solutions, residents, edge_devices)
        scout_firefly_phase(solutions, residents, edge_devices, area_size)

        for solution in solutions:
            fitness = evaluate_fitness(solution, residents, edge_devices)
            if fitness >= best_fitness:
                best_solution = solution
                best_fitness = fitness

    print("迭代次数：")
    print(i)

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


    print("最好路径的适应度：")
    print(best_solution.fitness)
    plt.show()

if __name__ == '__main__':
    # 参数设置
    np.random.seed(42)
    num_residents = 1000
    area_size = (100, 100)
    num_devices = 5
    radius = 10
    max_battery = 100
    warning_battery = 20
    max_resident_step = 8

    # 优化
    start_time = time.time()
    residents, edge_devices, best_solution = optimize_edge_device_positions(num_residents, area_size, num_devices,
                                                                           radius, max_battery, warning_battery,
                                                                           )

    end_time = time.time()
    print("充电次数：")
    print(charge_num_count)
    print("运行时间")
    print(end_time - start_time)
    # 可视化
    visualize(residents, edge_devices, best_solution, area_size)
