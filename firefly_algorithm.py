import numpy as np

import matplotlib.pyplot as plt
import time
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

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
            # print(resident.name) #打印已检测居民name
            detected_residents[resident.name] = True
            self.energy_consume += 0.0092


class Firefly:
    def __init__(self, positions):
        self.positions = positions
        self.fitness = 0


def generate_residents(num_residents, area_size):
    # 使用正态分布随机生成居民的初始位置
    # x = np.random.normal(area_size[0] / 2, area_size[0] / 6, num_residents)
    # y = np.random.normal(area_size[0] / 2, area_size[0] / 6, num_residents)
    # 提取两列数据并转换为NumPy数组
    # 将检测到的居民装入字典中

    x = np.array(df2['x3'].values)

    y = np.array(df2['y3'].values)

    residents = [Resident("resident_" + str(i), x[i], y[i]) for i in range(num_residents)]

    for resident in residents:
        detected_residents[resident.name] = resident.detected

    return residents


def generate_initial_solutions(num_devices, area_size):
    solutions = []
    for _ in range(num_devices):
        positions = (np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1]))

        solutions.append(Firefly(positions))

    return solutions


def find_the_best(solution, residents, edge_device):
    x = solution.positions[0]
    y = solution.positions[1]
    num = 0
    for resident in residents:
        distance = np.sqrt((resident.x - x) ** 2 + (resident.y - y) ** 2)

        if distance <= edge_device.radius and not resident.detected:
            num += 1

    return num


def evaluate_fitness_1(solution, residents, edge_device):
    device = edge_device
    if isinstance(solution.positions, list):
        solution.positions = solution.positions[0]
    device.x = solution.positions[0]
    device.y = solution.positions[1]

    for resident in residents:
        device.detect_resident(resident)

    fitness = sum([1 for resident in residents if resident.detected])
    return fitness


def scout_firefly_phase(solutions, residents, edge_devices, area_size):
    for solution in solutions:
        if solution.fitness == 0:
            positions = [(np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1]))]
            solution.positions = positions
    return solutions


def firefly_bee_phase(solutions, residents, edge_devices, step_size):
    for i, solution in enumerate(solutions):
        position = solution.positions
        num = 0
        num_new_position = 0
        # print(position)
        for k in range(len(solutions)):
            if k != i:
                neighbor_solution = solutions[k]
                neighbor_position = neighbor_solution.positions

                # 计算两个萤火虫之间的距离
                distance = np.sqrt(
                    (position[0] - neighbor_position[0]) ** 2 + (position[1] - neighbor_position[1]) ** 2)

                # 根据距离调整亮度值和步长
                if distance == 0:
                    brightness = 0
                else:
                    brightness = solution.fitness / (distance ** 2)
                step = step_size / (distance + 1e-6)  # 防止除零错误

                # 更新位置时考虑亮度值、距离和步长的影响
                new_position = (
                    position[0] + brightness * (neighbor_position[0] - position[0]) * step + np.random.uniform(-0.5,
                                                                                                               0.5) * step,
                    position[1] + brightness * (neighbor_position[1] - position[1]) * step + np.random.uniform(-0.5,
                                                                                                               0.5) * step
                )

                # 检查新位置是否超出边界，若超出则将其限制在边界内
                new_position = (
                    np.clip(new_position[0], 0, area_size[0]),
                    np.clip(new_position[1], 0, area_size[1])
                )

                temp_num = find_the_best(Firefly(new_position), residents, edge_devices[i])
                # 判断是否接受新位置
                if temp_num >= num:
                    num = temp_num
                    num_new_position = new_position
        # 更新位置
        distance = np.sqrt(
            (num_new_position[0] - solution.positions[0]) ** 2 + (num_new_position[1] - solution.positions[1]) ** 2)
        energy = round(107.44 * (distance / 9) / 3600 - 124.86 * (distance / 9) / 3600, 4)

        edge_devices[i].energy_consume += energy

        solution.positions = num_new_position
        solution.fitness = evaluate_fitness_1(Firefly(num_new_position), residents, edge_devices[i])

    return solutions


def step(solutions, step_size, edge_devices, best_solution, best_fitness, residents):
    for resident in residents:
        if not resident.detected:
            resident.move(max_resident_step)

    # 萤火虫算法的流程
    solutions = firefly_bee_phase(solutions, residents, edge_devices, step_size)
    solutions = scout_firefly_phase(solutions, residents, edge_devices, area_size)

    for i, solution in enumerate(solutions):
        fitness = evaluate_fitness_1(solution, residents, edge_devices[i])
        if fitness >= best_fitness:
            best_solution = solution
            best_fitness = fitness

    return residents, edge_devices, best_solution, best_fitness, solutions


def visualize(residents, edge_devices, best_solution, area_size, solutions):
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

    for solution in solutions:
        ax.plot(solution.positions[0], solution.positions[1], 'g^', markersize=10)

    ax.annotate(f'Best Fitness: {best_solution.fitness}',
                (best_solution.positions[0], best_solution.positions[1] - 10), ha='center')

    print("最好路径的适应度：")
    print(best_solution.fitness)
    plt.show()


# 定义强化学习的训练函数
def train(num_episodes):
    charge_num_count = 0
    for episode in range(num_episodes):
        if episode % 5 == 0:
            sum_reward = 0

        # 初始化环境
        residents = generate_residents(num_residents, area_size)
        # 生成边缘设备对象，放入列表中
        edge_devices = [EdgeDevice(0, 0, radius, max_battery, warning_battery) for _ in range(num_devices)]

        done = False
        # 萤火虫算法所需的参数
        solutions = generate_initial_solutions(num_devices, area_size)
        best_solution = None
        best_fitness = 0
        i = 0
        state = []

        for i, solution in enumerate(solutions):
            position = solution.positions
            state.append(position)  # 初始状态为边缘设备位置
        state = np.array(state)
        state = np.reshape(state, (1, 5, 2))
        print("state")
        print(state)

        reward = 0

        print("-------------------------------------------------")
        print("第几个episode:", episode)
        print("-------------------------------------------------")

        while not done:
            # 选择动作
            if np.random.rand() < epsilon:
                step_size = np.random.choice(step_size_range)  # 随机选择动作的索引

            else:
                # 根据当前状态选择最优动作

                step_size = model.predict(state)
                step_size = int(round(np.max(step_size) * 100))

            print("step_size")
            print(step_size)
            temp_1 = best_fitness
            temp_2 = charge_num_count
            _, _, best_solution, best_fitness, solutions = step(solutions, step_size, edge_devices, best_solution,
                                                                best_fitness, residents)

            i += 1

            # 计算奖励
            # 每多迭代一次就-1
            reward = reward - 1
            if best_fitness > temp_1:
                reward = reward + 1
                # 充电次数增加-1
            if charge_num_count > temp_2:
                reward = reward - 1

            sum_reward += reward

            # 执行动作并观察新状态
            new_state = []

            for i, solution in enumerate(solutions):
                position = solution.positions
                new_state.append(position)  # 初始状态为边缘设备位置

            new_state = np.reshape(new_state, (1, 5, 2))
            # 将经验存储到回放缓冲区

            buffer.append((state, step_size, reward, new_state, done))
            if len(buffer) > buffer_size:
                buffer.pop(0)

            # 从回放缓冲区中随机采样一批数据进行训练
            if len(buffer) >= batch_size:  # 通过检查回放缓冲区的大小是否大于等于批量大小
                batch_indices = np.random.choice(len(buffer), size=batch_size, replace=False)
                batch_data = [buffer[i] for i in batch_indices]

                states_batch = np.concatenate([data[0] for data in batch_data], axis=0)

                action_indices_batch = np.array([data[1] for data in batch_data])
                rewards_batch = np.array([data[2] for data in batch_data])
                new_states_batch = np.concatenate([data[3] for data in batch_data], axis=0)
                dones_batch = np.array([data[4] for data in batch_data])

                max = int(round(np.max(target_model.predict(new_states_batch) * 100)))

                # 使用目标网络计算目标Q值,目标Q值是由奖励(rewards_batch)和下一个状态(new_states_batch)的最大Q值乘以折扣因子(gamma)得到的。
                target_values = rewards_batch + gamma * max * (1 - dones_batch)

                # 训练深度强化学习模型
                q_values = model.predict(states_batch)
                q_values = np.squeeze(q_values)
                q_values = np.max(q_values, axis=1)
                q_values = np.array(q_values) * 100

                td_error = q_values - target_values
                loss = tf.reduce_mean(tf.square(td_error))

                train_loss = model.train_on_batch(states_batch, td_error)

                if episode % 4 == 0:
                    target_model.set_weights(model.get_weights())

            state = new_state

            # 检查是否达到终止状态
            done = all(resident.detected for resident in residents)
        sum_energy = 0
        for i in range(num_devices):
            sum_energy += edge_devices[i].energy_consume
        print("sum_energy")
        print(sum_energy)
        if (episode + 1) % 5 == 0:
            ave_reward = sum_reward / 5
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
            sum_reward = 0

        print("迭代次数：")
        print(i)


def optimize_edge_device_positions1(num_residents, area_size, num_devices, radius, max_battery, warning_battery):
    print("------现在进入实际的运行---------")
    # 生成居民对象，放入列表中
    residents = generate_residents(num_residents, area_size)
    # 生成边缘设备对象，放入列表中
    edge_devices = [EdgeDevice(0, 0, radius, max_battery, warning_battery) for _ in range(num_devices)]
    # 萤火虫算法所需的参数
    solutions = generate_initial_solutions(num_devices, area_size)
    state = []
    for i, solution in enumerate(solutions):
        position = solution.positions
        state.append(position)  # 初始状态为边缘设备位置
    state = np.array(state)
    state = np.reshape(state, (1, 5, 2))

    step_size = model.predict(state)
    step_size = int(round(np.max(step_size) * 100))
    print("step_size")
    print(step_size)

    best_solution = None
    best_fitness = 0
    i = 0
    while False in detected_residents.values():
        i = i + 1
        for resident in residents:
            if not resident.detected:
                resident.move(max_resident_step)

            # 萤火虫算法的流程
        solutions = firefly_bee_phase(solutions, residents, edge_devices, step_size)
        solutions = scout_firefly_phase(solutions, residents, edge_devices, area_size)

        for i, solution in enumerate(solutions):

            fitness = evaluate_fitness_1(solution, residents, edge_devices[i])
            if fitness >= best_fitness:
                best_solution = solution
                best_fitness = fitness
    sum_energy = 0
    for i in range(num_devices):
        sum_energy += edge_devices[i].energy_consume
    print("sum_energy")
    print(sum_energy)
    print("迭代次数：")
    print(i)

    return residents, edge_devices, best_solution, solutions


def dqn_loss(y_true, y_pred):
    td_error = y_true - y_pred

    loss = tf.reduce_mean(tf.square(td_error))
    total_loss = loss

    return total_loss


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
    # step_size=80

    # 定义强化学习的环境和动作空间

    step_size_range = list(range(10, 101))
    print(step_size_range)

    # 定义强化学习的超参数
    alpha = 0.001  # 学习率
    gamma = 0.9  # 折扣因子
    epsilon = 0.2  # 探索率
    num_episodes = 5

    # 创建经验回放缓冲区
    buffer_size = 200
    batch_size = 32
    buffer = []

    # 创建深度强化学习模型
    model = Sequential()
    model.add(Dense(32, input_shape=(5, 2), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss=dqn_loss, optimizer=Adam(learning_rate=alpha))
    # 创建目标Q值网络，与当前状态的Q值网络结构相同
    target_model = Sequential()
    target_model.add(Dense(32, input_shape=(5, 2), activation='relu'))
    target_model.add(Dense(32, activation='relu'))
    target_model.add(Dense(32, activation='relu'))
    target_model.add(Dense(1, activation='linear'))
    target_model.add(Dense(1, activation="sigmoid"))

    train(num_episodes)

    # 前面都是训练参数，训练好之后开始运行
    start_time = time.time()
    charge_num_count = 0
    residents, edge_devices, best_solution, solutions = optimize_edge_device_positions1(num_residents, area_size,
                                                                                        num_devices,
                                                                                        radius, max_battery,
                                                                                        warning_battery,
                                                                                        )

    end_time = time.time()
    print("充电次数：")
    print(charge_num_count)
    print("运行时间")
    print(end_time - start_time)
    # 可视化
    visualize(residents, edge_devices, best_solution, area_size, solutions)
