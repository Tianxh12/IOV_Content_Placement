# 计算两个位置之间的欧氏距离
import numpy as np


def calculate_distance(position1, position2):
    diff_x = position1[0] - position2[0]
    diff_y = position1[1] - position2[1]
    return (diff_x ** 2 + diff_y ** 2) ** 0.5


def calculate_reward(self):
    total_request_to_rsu = 0  # 向RSU请求的总次数
    hit = 0  # 本地RSU缓存，即命中次数
    sum_done = 0
    # 遍历所有车辆
    sum_delay_transmission = []
    sum_consumption_energy = []
    for vehicle in self.vehicles:
        # 获取车辆请求的内容
        content = [content for content in self.content_pool.content_list
                   if vehicle.request_state[content.content_id] == 1]
        request_content = content[0]
        is_cloud = 0

        # 定义车辆请求位置
        content_from = ""
        # 检查车辆自身缓存状态
        if vehicle.cache_state[request_content.content_id] == 1:
            """车辆自身缓存了请求的内容，给予奖励  自身缓存并不能检测该方案的优劣，因此时延、能耗设置为0"""
            # 计算内容传输时延
            delay_transmission = 0
            # 计算内容传输能耗
            consumption_energy = 0
            content_from = "自身缓存"
            """End"""
        else:
            """Start 参数定义 """
            p_v_transmission = 1  # 车辆传输功率 通常在几百毫瓦（mW）到几瓦（W）之间
            p_v_storge = 2  # 车辆存储功率
            r_v_storge = 100  # 车辆存储内容的速率MB/s
            """End"""

            # 测试加入V2V后的变化
            # vehicle.neighbor_vehicles_id = []

            # 逐个给车辆发送请求询问是否存在所需内容
            delay_request = 0
            consumption_transmission = 0
            delay_process = 0
            consumption_process = 0
            is_have = 0
            v2v_vehicle_id = -1
            for vehicle_id in vehicle.neighbor_vehicles_id:

                """Start 计算向车辆请求、处理时延能耗"""
                # 距离
                distance = calculate_distance(vehicle.position, self.vehicles[vehicle_id].position)
                r_v2v = V2V_speed_calculate(distance)
                distance = distance - 10
                r_v2v_test = V2V_speed_calculate(distance)
                # 请求时延 = 请求大小 / 速率 + 传播时延= 距离 / 传播速度   其中，距离是车辆与目标之间的距离，传播速度是信号在传播媒介中的传播速度
                # 在自由空间中，电磁波（包括无线通信信号）的传播速度近似为光速，约为 3×10^8 米/秒
                delay_request = delay_request + request_content.request_size / r_v2v + distance / 3 / 10e8
                # 请求传输能耗
                consumption_transmission = consumption_transmission + p_v_transmission * delay_request
                # 车辆 处理请求时延、能耗
                delay_process_item1, consumption_process_item1 = Vehicle_process(request_content)
                # 加上之前的时延、能耗
                delay_process = delay_process + delay_process_item1
                consumption_process = consumption_process + consumption_process_item1
                """End"""

                if self.vehicles[vehicle_id].cache_state[request_content.content_id] == 1:
                    # 找到缓存有请求内容的车辆就调出循环
                    is_have = 1
                    content_from = "V2V" + str(vehicle_id)
                    v2v_vehicle_id = vehicle_id
                    break
            # 此处已找到缓存有该内容的 V2V车辆  累加起来所有发送请求的时延、能耗 为以下：
            vehicle_request_delay = delay_request + delay_process
            vehicle_request_consumption = consumption_transmission + consumption_process
            # print(f"车辆请求时延:{vehicle_request_delay},车辆请求能耗：{vehicle_request_consumption}")

            if is_have:
                """Start 计算车辆存储、内容传输所耗时延、能耗"""
                # 车辆存储内容时延、能耗
                delay_storge = request_content.content_size / r_v_storge
                consumption_storge = p_v_storge * delay_storge

                # 内容传输时延、能耗
                delay_content = request_content.content_size / r_v2v
                consumption_content = delay_content * p_v_transmission

                # 总 = 请求 + 内容传输 + 存储
                delay_transmission = vehicle_request_delay + delay_content + delay_storge
                consumption_energy = (vehicle_request_consumption + consumption_content
                                      + consumption_storge)
                """End"""

                # 计算V2V 什么时候脱离
                V2V_time_to_loss = time_to_loss_of_communication(vehicle.position, self.vehicles[vehicle_id].position,
                                                                 vehicle.speed, self.vehicles[vehicle_id].speed,
                                                                 vehicle.direction, self.vehicles[vehicle_id].direction,
                                                                 2 * vehicle.v2v_range)

                if V2V_time_to_loss < delay_transmission:
                    """V2V未完成 所消耗时延、能耗"""
                    # 已传输内容大小
                    completed_content = V2V_time_to_loss * r_v2v
                    surplus_content = request_content.content_size - completed_content
                    # 能耗
                    detach_consumption = completed_content * p_v_transmission
                    # 更新车辆位置
                    position = vehicle.update_position(V2V_time_to_loss)

                    # V2V 总时延 = 车辆请求时延 + 内容传输时延
                    V2V_delay = vehicle_request_delay + V2V_time_to_loss
                    V2V_consumption = vehicle_request_consumption + detach_consumption
                    """End"""

                    # 获取位置更新后的最近的RSU_id
                    close_rsus = [rsu for rsu in self.rsus if
                                  np.linalg.norm(position - rsu.position) < 2 * self.rsus[0].v2r_range]
                    if close_rsus:
                        closest_rsu = min(close_rsus, key=lambda rsu: np.linalg.norm(position - rsu.position))
                        rsu_close_id = closest_rsu.rsu_id
                    else:
                        rsu_close_id = -1

                    # 检查RSU是否存在
                    if rsu_close_id == -1:
                        is_cloud = 1  # 更新后的位置不在任何一个rsu通讯范围内，自身向云服务器请求
                    else:
                        rsu_close = self.rsus[rsu_close_id]
                        # 距离
                        distance = calculate_distance(position, rsu_close.position)
                        r_v2r = V2R_speed_calculate(distance)
                        # 请求时延 = 请求大小 / 速率 + 传播时延= 距离 / 传播速度   其中，距离是车辆与目标之间的距离，传播速度是信号在传播媒介中的传播速度
                        # 在自由空间中，电磁波（包括无线通信信号）的传播速度近似为光速，约为 3×10^8 米/秒
                        delay_request_V2R = request_content.request_size / r_v2r + distance / 3 / 10e8
                        # 请求传输能耗
                        consumption_transmission = p_v_transmission * delay_request
                        # RSU 处理请求时延、能耗
                        delay_process, consumption_process = RSU_process(request_content)
                        # 检查最近 RSU 缓存状态
                        if rsu_close.cache_state[request_content.content_id] == 1:
                            # 最近 RSU 缓存了请求的内容，给予奖励
                            hit += 1  # 本地RSU缓存内容，即命中！
                            p_r_transmission = 8  # RSU传输功率 通常在几瓦（W）到数十瓦（W）之间
                            # 内容传输时延

                            delay_content = surplus_content / r_v2r
                            delay_transmission = (V2V_delay + delay_request_V2R + delay_process + delay_content
                                                  + delay_storge)

                            consumption_content = delay_content * p_r_transmission

                            # 计算内容传输能耗
                            consumption_energy = (
                                    V2V_consumption + consumption_transmission + consumption_process
                                    + consumption_content + consumption_storge)
                            content_from = "V2V未完成，RSU接力"
                        else:
                            # 云服务器
                            is_cloud = 1
            else:
                total_request_to_rsu += 1  # 向rsu发送请求次数

                # 检查最近 RSU 缓存状态
                rsu_close = self.rsus[vehicle.rsu_close_id]

                # 参数
                p_r_transmission = 8  # RSU传输功率 通常在几瓦（W）到数十瓦（W）之间
                optical_fiber = 125  # MBps 光纤通信速度  即 RSU间通信速度
                # 距离
                distance = calculate_distance(vehicle.position, rsu_close.position)
                r_v2r = V2R_speed_calculate(distance)

                # 请求时延 = 请求大小 / 速率 + 传播时延= 距离 / 传播速度   其中，距离是车辆与目标之间的距离，传播速度是信号在传播媒介中的传播速度
                # 在自由空间中，电磁波（包括无线通信信号）的传播速度近似为光速，约为 3×10^8 米/秒
                delay_request = request_content.request_size / r_v2r + distance / 3 / 10e8
                # 请求传输能耗
                consumption_transmission = p_v_transmission * delay_request

                # RSU 处理请求时延、能耗
                delay_process, consumption_process = RSU_process(request_content)

                # 车辆存储内容时延、能耗
                delay_storge = request_content.content_size / r_v_storge
                consumption_storge = p_v_storge * delay_storge

                if rsu_close.cache_state[request_content.content_id] == 1:
                    # 最近 RSU 缓存了请求的内容，给予奖励
                    hit += 1  # 本地RSU缓存内容，即命中！
                    # 内容传输时延
                    delay_content = request_content.content_size / r_v2r
                    delay_transmission = (vehicle_request_delay + delay_request + delay_process + delay_content
                                          + delay_storge)

                    consumption_content = delay_content * p_r_transmission

                    # 计算内容传输能耗
                    consumption_energy = (vehicle_request_consumption + consumption_transmission + consumption_process
                                          + consumption_content + consumption_storge)
                    sum_done += 1
                    content_from = "本地RSU"
                else:
                    # 最近 RSU 也没有缓存请求的内容
                    # 找最近rsu的临近RSU
                    if rsu_close.neighbor_RSUs:
                        for neighbor_id in rsu_close.neighbor_RSUs:
                            # 根据rsu_id找rsu
                            neighbor_rsus = []
                            for rsu in self.rsus:
                                if neighbor_id == rsu.rsu_id:
                                    neighbor_rsus.append(rsu)
                            # RSU 处理请求时延、能耗
                            delay_process_item, consumption_process_item = RSU_process(request_content)
                            delay_process += delay_process_item
                            consumption_process += consumption_process_item
                            for neighbor_rsu in neighbor_rsus:
                                if neighbor_rsu.cache_state[request_content.content_id] == 1:
                                    # 内容传输时延
                                    delay_content_R2R = request_content.content_size / optical_fiber
                                    consumption_content_R2R = delay_content_R2R * p_r_transmission
                                    content_from = "临近RSU" + str(neighbor_rsu.rsu_id)
                                    break
                        delay_content_V2R = request_content.content_size / r_v2r
                        consumption_content_V2R = delay_content_V2R * p_r_transmission
                        delay_transmission = (vehicle_request_delay + delay_request + delay_process + delay_content_R2R
                                              + delay_content_V2R + delay_storge)
                        consumption_energy = (
                                vehicle_request_consumption + consumption_transmission + consumption_process
                                + consumption_content_R2R + consumption_content_V2R + consumption_storge)
                    else:
                        # 向云服务器请求
                        is_cloud = 1

                # 计算V2V 什么时候脱离
                V2R_time_to_loss = time_to_loss_of_communication(vehicle.position, rsu_close.position,
                                                                 vehicle.speed, 0,
                                                                 vehicle.direction, 0,
                                                                 rsu_close.v2r_range)
                if is_cloud != 1 and V2R_time_to_loss < delay_transmission:
                    """V2R 未完成 所消耗时延、能耗"""
                    # 已传输内容大小
                    completed_content = V2R_time_to_loss * r_v2r
                    surplus_content = request_content.content_size - completed_content

                    # 能耗
                    detach_consumption = completed_content * p_v_transmission

                    # V2R 总时延 = 车辆请求时延 + 内容传输时延
                    V2R_delay = vehicle_request_delay + V2R_time_to_loss
                    V2R_consumption = vehicle_request_consumption + detach_consumption
                    """End"""

                    # 更新车辆位置
                    position = vehicle.update_position(V2R_time_to_loss)

                    # 获取最近的RSU
                    close_rsus = [rsu for rsu in self.rsus if
                                  np.linalg.norm(position - rsu.position) < self.rsus[0].v2r_range]
                    if close_rsus:
                        closest_rsu = min(close_rsus, key=lambda rsu: np.linalg.norm(position - rsu.position))
                        rsu_close_id = closest_rsu.rsu_id
                    else:
                        rsu_close_id = -1

                    # 检查RSU是否存在
                    if rsu_close_id == -1:
                        is_cloud = 1  # 更新后的位置不在任何一个rsu通讯范围内，自身向云服务器请求
                    else:
                        # 请求的RSU一定会有，因为从本地传输给下一个RSU的
                        # 剩余内容传输时延
                        delay_content_R2R = surplus_content / optical_fiber
                        consumption_content_R2R = delay_content_R2R * p_r_transmission

                        delay_process, consumption_process = RSU_process_new(surplus_content)

                        delay_transmission = V2R_delay + delay_content_R2R + delay_process + delay_storge
                        consumption_energy = (V2R_consumption + consumption_content_R2R + consumption_process
                                              + consumption_storge)
                    content_from = "V2R未完成 然后又V2R"

        if is_cloud == 1:
            # 从云服务器获取内容
            # RSU 处理请求时延、能耗
            delay_transmission = 500
            consumption_energy = 500

        if is_cloud == 1:
            content_from = "云服务器"
        # hit_rate = hit / total_request_to_rsu
        # reward -= 0.4 * delay_transmission + 0.3 * consumption_energy  # 示例传输时延惩罚
        # print(
        #     f"车辆id:{vehicle.vehicle_id},请求内容大小为：{request_content.content_size},请求来源：{content_from}，所消耗时延："
        #     f"{delay_transmission},能耗：{consumption_energy}")
        sum_delay_transmission.append(delay_transmission)
        sum_consumption_energy.append(consumption_energy)
    return sum(sum_delay_transmission), sum(sum_consumption_energy)


def RSU_process(request_content):
    factor = 0.05  # 比例因子，调整请求内容大小对处理时间的影响程度
    delay_basic_process = 0.2  # 基本处理时间
    p_r_process = 1  # RSU处理功率
    # 请求处理时延
    delay_process = delay_basic_process + request_content.request_size * factor
    # RSU处理能耗
    consumption_process = delay_process * p_r_process
    return delay_process, consumption_process


def RSU_process_new(request_content):
    factor = 0.05  # 比例因子，调整请求内容大小对处理时间的影响程度
    delay_basic_process = 0.2  # 基本处理时间
    p_r_process = 1  # RSU处理功率
    # 请求处理时延
    delay_process = delay_basic_process + request_content * factor
    # RSU处理能耗
    consumption_process = delay_process * p_r_process
    return delay_process, consumption_process


def Vehicle_process(request_content):
    factor = 0.05  # 比例因子，调整请求内容大小对处理时间的影响程度
    delay_basic_process = 0.5  # 基本处理时间
    p_r_process = 0.5  # RSU处理功率
    # 请求处理时延
    delay_process = delay_basic_process + request_content.request_size * factor
    # RSU处理能耗
    consumption_process = delay_process * p_r_process
    return delay_process, consumption_process


def V2V_speed_calculate(distance):
    f = 1  # 频率 LF（30 kHz - 300 kHz）、MF（300 kHz - 3 MHz）、HF（3 MHz - 30 MHz）适用于远距离通信，但带宽相对较小，主要用于无线电广播、海事通信等。
    C = 10  # 通常在 0 到 40 之间，取决于频率的单位和具体的城市环境。
    p_v_transmission = 1  # 车辆传输功率 通常在几百毫瓦（mW）到几瓦（W）之间
    e = 5  # 天线增益 车辆和基站的天线增益可以在 2dBi 到 10dBi 的范围内
    w_vehicle = 5  # 比特每秒，如100 Mbps 来表示车辆可用的通信带宽。
    path_loss_exponent = 3  # 路径损耗模型中的指数，通常在2到4之间。
    # 路径损耗
    path_loss = 20 * np.log10(distance) + 20 * np.log10(f) + C
    # 信噪比
    SNR = p_v_transmission * e / path_loss * distance * path_loss_exponent
    # v2r速率
    r_v2v = w_vehicle * np.log2(1 + SNR)
    return r_v2v


def V2R_speed_calculate(distance):
    f = 1  # 频率 LF（30 kHz - 300 kHz）、MF（300 kHz - 3 MHz）、HF（3 MHz - 30 MHz）适用于远距离通信，但带宽相对较小，主要用于无线电广播、海事通信等。
    C = 10  # 通常在 0 到 40 之间，取决于频率的单位和具体的城市环境。
    p_v_transmission = 1  # 车辆传输功率 通常在几百毫瓦（mW）到几瓦（W）之间
    e = 5  # 天线增益 车辆和基站的天线增益可以在 2dBi 到 10dBi 的范围内
    w_rsu = 10  # 比特每秒，如100 Mbps 来表示车辆可用的通信带宽。
    path_loss_exponent = 3  # 路径损耗模型中的指数，通常在2到4之间。
    # 路径损耗
    path_loss = 20 * np.log10(distance) + 20 * np.log10(f) + C
    # 信噪比
    SNR = p_v_transmission * e / path_loss * distance * path_loss_exponent
    # v2r速率
    r_v2r = w_rsu * np.log2(1 + SNR)
    return r_v2r


"""计算车辆V2V通信时间"""


def distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


def are_vehicles_communicating(pos1, pos2, communication_range):
    return distance(pos1, pos2) <= communication_range


def update_position(pos, speed, direction, time):
    new_pos = pos + speed * direction * time
    return new_pos


def time_to_loss_of_communication(initial_pos1, initial_pos2, speed1, speed2, direction1, direction2,
                                  communication_range):
    time = 0
    max_simulation_steps = 1000  # 防止无限循环
    while time < max_simulation_steps:
        pos1 = update_position(initial_pos1, speed1, direction1, time)
        pos2 = update_position(initial_pos2, speed2, direction2, time)

        if not are_vehicles_communicating(pos1, pos2, communication_range):
            return time

        time += 1

    # 如果在最大步数内仍然通信，则返回一个大的值，表示未来很长时间内都保持通信
    return float('inf')


"""End"""
