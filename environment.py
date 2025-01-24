# environment.py
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

import function
from DQN import DQNAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device in use:", device)


# 定义车辆类
class Vehicle:
    def __init__(self, vehicle_id, position, v2v_range, capacity, neighbor_vehicles_id, rsu_close_id, request_state,
                 cache_state):
        self.neighbor_vehicles_id = neighbor_vehicles_id  # 邻居车辆
        self.rsu_close_id = rsu_close_id  # 本地RSU
        self.capacity = capacity  # 容量
        self.vehicle_id = vehicle_id  # 车辆ID
        self.position = position  # 位置
        self.v2v_range = v2v_range  # v2v通信范围
        self.cache_state = cache_state  # 车辆缓存内容
        self.request_state = request_state  # 车辆请求状态 1*contents_nums个0 1

        self.speed = random.randint(5, 60)  # 初始速度 随机5~60迈
        self.direction = np.random.rand() * 2 * np.pi  # 初始移动方向为随机方向
        self.initial_position = position.copy()  # 保存初始位置

    # 在 Vehicle 类中添加更新位置的方法
    def update_position(self, time):
        # 根据速度和移动方向更新车辆位置
        position = [0, 0]
        delta_x = time * self.speed * np.cos(self.direction)
        delta_y = time * self.speed * np.sin(self.direction)
        position[0] = self.position[0] + delta_x
        position[1] = self.position[1] + delta_y
        return position

    def is_in_range(self, vehicle_position):
        # 判断车辆是否在车辆的通信范围内
        distance = math.sqrt(
            (self.position[0] - vehicle_position[0]) ** 2 + (self.position[1] - vehicle_position[1]) ** 2)
        return distance <= 2 * self.v2v_range

    def is_in_range2(self, vehicle_position, vehicle_position1):
        # 判断车辆是否在车辆的通信范围内
        distance = math.sqrt(
            (vehicle_position[0] - vehicle_position1[0]) ** 2 + (vehicle_position[1] - vehicle_position1[1]) ** 2)
        return distance <= 2 * self.v2v_range

    def get_neighbor_vehicles(self, all_vehicles):
        # 获取邻居车辆
        neighbor_vehicles_id = [vehicle.vehicle_id for vehicle in all_vehicles if vehicle != self and
                                self.is_in_range(vehicle.position)]
        return neighbor_vehicles_id

    def get_closest_rsu_id(self, all_rsus, v2r_range):
        # 获取最近的RSU
        close_rsus = [rsu for rsu in all_rsus if np.linalg.norm(self.position - rsu.position) < v2r_range]
        if close_rsus:
            closest_rsu = min(close_rsus, key=lambda rsu: np.linalg.norm(self.position - rsu.position))
            return closest_rsu.rsu_id
        else:
            return -1
        # closest_rsu = min(all_rsus, key=lambda rsu: np.linalg.norm(self.position - rsu.position))
        # return closest_rsu.rsu_id


# 定义RSU类
class RSU:
    def __init__(self, rsu_id, position, v2r_range, capacity, neighbor_RSUs_id, num_contents):
        self.neighbor_RSUs_id = neighbor_RSUs_id  # 邻居RSUs
        self.capacity = capacity  # 容量
        self.rsu_id = rsu_id  # RSU ID
        self.position = position  # 位置
        self.v2r_range = v2r_range  # v2r通信范围
        self.cache_state = [0] * num_contents  # 设置 RSU 的缓存状态为零

    def is_in_range(self, vehicle_position):
        # 判断车辆是否在RSU的通信范围内
        distance = math.sqrt(
            (self.position[0] - vehicle_position[0]) ** 2 + (self.position[1] - vehicle_position[1]) ** 2)
        return distance <= 2 * self.v2r_range

    def get_neighbor_RSUs(self, all_rsus):
        # 获取邻居 RSUs
        neighbor_RSUs_id = [rsu.rsu_id for rsu in all_rsus if rsu != self and self.is_in_range(rsu.position)]
        return neighbor_RSUs_id

    def update_cache_state(self, cached_contents, num_contents):
        # 更新 RSU 的缓存状态
        self.cache_state = [0] * num_contents
        for content in cached_contents:
            self.cache_state[content.content_id] = 1


class Content:
    def __init__(self, content_id, content_type):
        self.content_id = content_id
        self.content_type = content_type  # 内容类型
        self.content_size = self.generate_content_size()  # 生成内容大小
        self.request_size = random.randint(0, 5)  # 随机请求大小

    def generate_content_size(self):
        if self.content_type == "Map":
            return random.randint(100, 150)
        elif self.content_type == "Video":
            return random.randint(50, 100)
        elif self.content_type == "Music":
            return random.randint(10, 50)
        else:
            raise ValueError("Unsupported content type")


class ContentPool:
    def __init__(self, num_contents):
        self.num_contents = num_contents
        self.content_list = self.generate_content_list()
        self.display_content_pool()

    def generate_content_list(self):
        content_list = []
        for i in range(self.num_contents):
            content_id = i
            content_type = random.choice(["Map", "Video", "Music"])
            content = Content(content_id, content_type)
            content_list.append(content)
        return content_list

    def display_content_pool(self):
        for content in self.content_list:
            print(
                f"Content ID: {content.content_id}, Type: {content.content_type}, Size: {content.content_size}, Request Size: {content.request_size}")


class CacheEnvironment:
    def __init__(self, num_vehicles, num_rsus, num_contents, max_training_steps, time_slots):
        # 初始化环境参数
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.num_contents = num_contents
        self.content_pool = ContentPool(num_contents)  # 内容池list
        self.rsus = self.generate_rsus(num_rsus, num_contents)  # 生成RSUs
        self.vehicles = self.generate_vehicles(num_vehicles, num_contents)  # 生成车辆
        self.max_training_steps = max_training_steps  # 定义最大代数
        self.state = self.reset()
        # 设置 input_size 为初始状态大小
        input_size = self.state.size(0)
        # 设置 output_size 为全局 RSU 内容放置决策的数量
        output_size = 2 ** num_contents * num_rsus  # 这个值可能非常大，谨慎使用
        self.dqn_agent = DQNAgent(input_size, output_size)
        # 计数器
        self.current_step = 0

        self.time_slots = time_slots  # 时间槽的数量
        self.current_time_slot = 0  # 当前时间槽

    def generate_vehicles(self, num_vehicles, num_contents):
        vehicles = []
        for i in range(num_vehicles):
            vehicle_id = i
            position = np.random.randint(250, 750, size=(2,))  # 随机生成车辆位置
            v2v_range = 50  # v2v通信范围
            v2r_range = 200  # v2r通信范围
            capacity = 1  # 车辆容量
            request_state = [0] * num_contents
            cache_state = [0] * num_contents
            content_request = random.choice(self.content_pool.content_list)  # 随机生成车辆请求的内容
            content_cache = random.choice(self.content_pool.content_list)  # 随机生成车辆请求的内容
            # 更新请求状态
            if content_request:
                request_state[content_request.content_id] = 1
            if content_cache:
                cache_state[content_cache.content_id] = 1
            vehicle = Vehicle(vehicle_id, position, v2v_range, capacity, [], 0, request_state, cache_state)
            # 初始时邻居车辆为空，最近的RSU为空
            vehicles.append(vehicle)
        # 现在所有的车辆都已经生成完毕，可以更新邻居车辆和最近的RSU信息
        for vehicle in vehicles:
            vehicle.neighbor_vehicles_id = vehicle.get_neighbor_vehicles(vehicles)
            vehicle.rsu_close_id = vehicle.get_closest_rsu_id(self.rsus, v2r_range)
        return vehicles

    def generate_rsus(self, num_rsus, num_contents):
        rsus = []
        for i in range(num_rsus):
            rsu_id = i
            v2r_range = 200  # v2r通信范围
            while True:
                position = np.random.randint(0, 1000, size=(2,))  # 随机生成RSU位置
                capacity = 2  # RSU容量
                rsu = RSU(rsu_id, position, v2r_range, capacity, [], num_contents)
                # 检查与现有RSU的距离
                distances = [np.linalg.norm(rsu.position - existing.position) for existing in rsus]
                if all(distance >= 2 * v2r_range for distance in distances):
                    break  # 如果所有距离都大于v2r_range，跳出循环
            rsus.append(rsu)
        # 现在所有的RSU都已经生成完毕，可以更新邻居RSU信息
        for rsu in rsus:
            rsu.neighbor_RSUs = rsu.get_neighbor_RSUs(rsus)
        return rsus

    def reset(self):
        # 重置环境状态 应为初始状态
        vehicle_positions = [vehicle.initial_position for vehicle in self.vehicles]  # 使用初始位置
        rsu_positions = [rsu.position for rsu in self.rsus]  # RSU 位置
        for rsu in self.rsus:
            rsu.cache_state = [0] * self.num_contents  # 设置 RSU 的缓存状态为零
        rsu_cache_states = [rsu.cache_state for rsu in self.rsus]  # 所有 RSU 的缓存状态
        vehicle_request_states = [vehicle.request_state for vehicle in self.vehicles]  # 所有车辆的请求状态
        self.current_time_slot = 0
        # vehicle_positions_tensor = torch.tensor(vehicle_positions).view(-1)
        # rsu_positions_tensor = torch.tensor(rsu_positions).view(-1)
        #
        # # 将二维列表展平为一维
        # rsu_cache_states_tensor = torch.tensor(
        #     [content_id for sublist in rsu_cache_states for content_id in sublist]).view(-1)
        #
        # vehicle_request_states_tensor = torch.tensor(vehicle_request_states).view(-1)
        #
        # state_tensor = torch.cat([
        #     vehicle_positions_tensor,
        #     rsu_positions_tensor,
        #     rsu_cache_states_tensor,
        #     vehicle_request_states_tensor
        # ])

        vehicle_positions_tensor = torch.tensor(vehicle_positions, dtype=torch.float32).view(-1)
        rsu_positions_tensor = torch.tensor(rsu_positions, dtype=torch.float32).view(-1)

        # 将二维列表展平为一维
        rsu_cache_states_tensor = torch.tensor(
            [content_id for sublist in rsu_cache_states for content_id in sublist], dtype=torch.float32).view(-1)

        vehicle_request_states_tensor = torch.tensor(vehicle_request_states, dtype=torch.float32).view(-1)

        state_tensor = torch.cat([
            vehicle_positions_tensor,
            rsu_positions_tensor,
            rsu_cache_states_tensor,
            vehicle_request_states_tensor
        ])

        return state_tensor

    def step(self, action, rewards):

        # 时隙 + 1，更新车辆位置
        self.current_time_slot += 1
        # for vehicle in self.vehicles:
        #     vehicle.update_position()

        RSU_cache_content_number = 0
        # 根据动作更新RSU缓存的内容
        for rsu in self.rsus:
            contents_to_place = []  # 初始化空列表
            # 将整数动作转换为二进制形式
            action_binary = bin(action)[2:].zfill(len(self.rsus) * self.content_pool.num_contents)
            # 提取当前RSU的子动作
            start_index = rsu.rsu_id * self.content_pool.num_contents
            end_index = (rsu.rsu_id + 1) * self.content_pool.num_contents
            rsu_action_binary = action_binary[start_index:end_index]
            # 根据 rsu_action_binary 的值确定是第几个内容，然后将该内容添加到 contents_to_place
            for content_id, bit in enumerate(rsu_action_binary):
                if bit == '1':
                    content = self.content_pool.content_list[content_id]
                    contents_to_place.append(content)
            # 更新RSU缓存状态
            rsu.update_cache_state(contents_to_place, self.content_pool.num_contents)
            RSU_cache_content_number = RSU_cache_content_number + sum(rsu.cache_state)
            # print(f"RSU_id为{rsu.rsu_id},缓存状态为：{rsu.cache_state}")

        # print(f"所有RSU共缓存{RSU_cache_content_number}个内容")
        # 计算每辆车 获取内容所消耗的时延、能耗
        delay_transmission, consumption_energy = function.calculate_reward(self)
        # print(f"总时延为{delay_transmission},总能耗为：{consumption_energy}")
        # 判断是否超出rsu范围
        reward = 0
        reward -= delay_transmission + consumption_energy + RSU_cache_content_number * 100
        rewards.append(reward)

        # # 更新计数器
        # self.current_step += 1

        # 检查时间槽是否耗尽
        done = self.current_time_slot >= self.time_slots

        # 返回新状态、奖励、是否结束等信息
        return self.get_state(), rewards, done

    def get_state(self):
        vehicle_positions = [vehicle.position for vehicle in self.vehicles]
        rsu_positions = [rsu.position for rsu in self.rsus]
        rsu_cache_states = [rsu.cache_state for rsu in self.rsus]
        vehicle_request_states = [vehicle.request_state for vehicle in self.vehicles]

        vehicle_positions_tensor = torch.tensor(vehicle_positions, dtype=torch.float32).view(-1)
        rsu_positions_tensor = torch.tensor(rsu_positions, dtype=torch.float32).view(-1)

        # 将二维列表展平为一维
        rsu_cache_states_tensor = torch.tensor(
            [content_id for sublist in rsu_cache_states for content_id in sublist], dtype=torch.float32).view(-1)

        vehicle_request_states_tensor = torch.tensor(vehicle_request_states, dtype=torch.float32).view(-1)

        state_tensor = torch.cat([
            vehicle_positions_tensor,
            rsu_positions_tensor,
            rsu_cache_states_tensor,
            vehicle_request_states_tensor
        ])

        return state_tensor

    def plot_environment(self):
        # 提取车辆和RSU的位置
        vehicle_positions = [vehicle.position for vehicle in self.vehicles]
        rsu_positions = [rsu.position for rsu in self.rsus]

        # 将位置分别拆分为 x 和 y 坐标
        vehicle_x, vehicle_y = zip(*vehicle_positions)
        rsu_x, rsu_y = zip(*rsu_positions)

        # 绘制车辆和RSU的位置
        plt.scatter(vehicle_x, vehicle_y, marker='o', label='Vehicles')
        plt.scatter(rsu_x, rsu_y, marker='s', label='RSUs')

        # 添加标签和图例
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Vehicle and RSU Positions')
        plt.legend()

        # 显示图形
        plt.show()
