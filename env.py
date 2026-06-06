import numpy as np
import torch

class Environment:
    def __init__(self,
                 num_cars=10,
                 highway_length=1000,
                 communications_range=500,
                 minimum_speed=17,
                 maximum_speed=33,
                 level_ratio=0.5,
                 lane_ratio=0.5,
                 transmitting_power=0.2,
                 band_size=1.8e5,
                 noise_power=7.2e-16,
                 steep_factor_k=1,
                 package_size_Z=1520,
                 maximum_delay=0.02,
                 reward_factor_1=0.5,
                 reward_factor_2=0.5,
                 sinr_threshold=100,
                 collision_minimum_time=2,
                 seed=42):
        self.num_cars = num_cars
        self.highway_length = highway_length
        self.communications_range = communications_range
        self.minimum_speed = minimum_speed
        self.maximum_speed = maximum_speed
        self.level_ratio = level_ratio
        self.lane_ratio = lane_ratio
        self.transmitting_power = transmitting_power
        self.band_size = band_size
        self.noise_power = noise_power
        self.steep_factor_k = steep_factor_k
        self.package_size_Z = package_size_Z
        self.maximum_delay = maximum_delay
        self.reward_factor_1 = reward_factor_1
        self.reward_factor_2 = reward_factor_2
        self.sinr_threshold = sinr_threshold
        self.collision_minimum_time = collision_minimum_time

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        np.random.seed(seed)

        # initialize state
        self.lanes = np.random.permutation([0]*int(num_cars*lane_ratio) + [1]*int(num_cars*(1-lane_ratio)))
        self.new_lanes = self.lanes.copy()
        self.levels = np.random.permutation([0]*int(num_cars*level_ratio) + [1]*int(num_cars*(1-level_ratio)))
        self.position_matrix = np.random.randint(0, highway_length, (num_cars))
        self.dis_matrix = np.zeros((num_cars, num_cars))
        self.update_dis()
        self.velocity_matrix = np.random.randint(minimum_speed, maximum_speed, (num_cars))
        self.new_velocity_matrix = self.velocity_matrix.copy()

    # channel / path loss
    def get_loss(self, dis, carrier_frequency):
        # simplified Friis + small shadowing
        return 32.4 + 20*np.log10(dis) + 20*np.log10(carrier_frequency), np.random.normal(0, 4)


    """
    UMi Street Canyon
    def get_loss(dis,carrier_frequency):
        d_breakpoint = 28.32
        if dis < 10:
            return friis_path_loss(dis,carrier_frequency),0
        elif dis < d_breakpoint:
            return path_loss_los(dis,carrier_frequency),shadow_loss_los()
        else:
            return path_loss_nlos(dis,carrier_frequency,d_breakpoint),shadow_loss_nlos()
        
    def friis_path_loss(dis,carrier_frequency):
        return 32.4 + 20*np.log10(dis) + 20*np.log10(carrier_frequency)  

    def path_loss_los(dis,carrier_frequency):
        return 32.4 + 21*np.log10(dis) + 20*np.log10(carrier_frequency)

    def path_loss_nlos(dis,carrier_frequency,d_breakpoint):
        return 32.4 + 40*np.log10(dis) + 20*np.log10(carrier_frequency) - 19*np.log10(d_breakpoint)

    def shadow_loss_los():
        return np.random.normal(0, 4)

    def shadow_loss_nlos():
        return np.random.normal(0, 7.82)
    """


    """

    RMa Urban Macro
    def get_loss(dis,carrier_frequency):
        d_breakpoint = 316.34
        if dis < 10:
            return friis_path_loss(dis,carrier_frequency),0
        else:
            los = np.random.binomial(1, np.e**(-(dis-10)/1000))
            if dis < d_breakpoint:
                path_loss = path_loss_los_1(dis,carrier_frequency)
                shadow_loss = shadow_loss_los_1()
            else:
                path_loss = path_loss_los_2(dis,carrier_frequency,d_breakpoint)
                shadow_loss = shadow_loss_los_2()
            if los:
                return path_loss, shadow_loss
            else:
                return max(path_loss_nlos(dis,carrier_frequency),path_loss),shadow_loss_nlos()
            
    def friis_path_loss(dis,carrier_frequency):
        return 32.4 + 20*np.log10(dis) + 20*np.log10(carrier_frequency)  

    def path_loss_los_1(dis,carrier_frequency):
        return 20 * np.log10(40*np.pi*dis*carrier_frequency/3) + 0.478 * np.log10(dis) - 0.7 + 0.0014 * dis

    def path_loss_los_2(dis,carrier_frequency,d_breakpoint):
        return path_loss_los_1(d_breakpoint,carrier_frequency) + 40 * np.log10(dis/d_breakpoint)

    def path_loss_nlos(dis,carrier_frequency):
        return 159.22 + 42.8*(np.log10(dis)-3) + 20*np.log10(carrier_frequency)

    def shadow_loss_los_1():
        return np.random.normal(0, 4)

    def shadow_loss_los_2():
        return np.random.normal(0, 6)

    def shadow_loss_nlos():
        return np.random.normal(0, 8)

    """

    def get_channel_gain(self, i, j):
        antenna_gain = 3 # dB
        carrier_frequency = 5.9 # GHz
        dis = self.dis_matrix[i][j]
        path_loss, shadow_loss = self.get_loss(dis, carrier_frequency)
        channel_gain = 10**((2*antenna_gain - path_loss - shadow_loss)/10)
        return channel_gain

    def get_interferers(self, actions, i, j):
        interferers = []
        for k in range(self.num_cars):
           if k != i and k != j and actions[i] == actions[k] and self.dis_matrix[j][k] <= self.communications_range:
               interferers.append(k)
        return interferers

    def get_available_vehicles(self):
        available_vehicles = []
        transmissions = 0
        for i in range(self.num_cars):
            tmp = []
            for j in range(self.num_cars):
                if self.dis_matrix[i][j] <= self.communications_range and i != j:
                    tmp.append(j)
                    transmissions += 1
            available_vehicles.append(tmp)
        # store as instance attribute so other methods can use it directly
        self.available_vehicles = available_vehicles
        self.transmissions = transmissions
        return available_vehicles, transmissions

    def get_edge_index_and_attr(self):
        source_nodes = []
        target_nodes = []
        edge_attr = []
        for i in range(self.num_cars):
            for j in self.available_vehicles[i]:
                source_nodes.append(i)
                target_nodes.append(j)
                edge_attr.append([self.dis_matrix[i][j]])
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        return edge_index, edge_attr

    def get_sinr(self, actions, i, j):
        interferences = 0
        interferers = self.get_interferers(actions, i, j)
        for k in range(len(interferers)):
            interferences += self.get_channel_gain(j, interferers[k]) * self.transmitting_power
        sinr = self.get_channel_gain(i, j) * self.transmitting_power / (self.noise_power + interferences)
        return sinr

    def get_rate(self, actions, i, j):
        return self.band_size * np.log2(1 + self.get_sinr(actions, i, j))

    def get_weight(self, i, j):
        weight = max(int(self.lanes[i] == self.lanes[j]) * np.exp(self.steep_factor_k * (self.collision_minimum_time - (self.dis_matrix[i][j] / max(abs(self.velocity_matrix[i] - self.velocity_matrix[j]), 0.01)))), 1)
        return weight

    def get_reward(self, actions):
        reward = np.zeros(self.num_cars)
        successful_transmissions = 0
        for i in range(self.num_cars):
            for j in self.available_vehicles[i]:
                successful_transmissions += int(self.get_sinr(actions, i, j) >= self.sinr_threshold)
                reward[i] += self.get_weight(i, j) * (self.reward_factor_1 * int(self.get_sinr(actions, i, j) >= self.sinr_threshold) + self.reward_factor_2 * int(self.get_rate(actions, i, j) >= self.package_size_Z / self.maximum_delay))
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        return reward / 10, successful_transmissions

    def update_position(self):
        for i in range(self.num_cars):
            self.position_matrix[i] += self.velocity_matrix[i]
            if self.position_matrix[i] >= self.highway_length:
                self.velocity_matrix[i] = np.random.randint(self.minimum_speed, self.maximum_speed)
                self.new_velocity_matrix[i] = self.velocity_matrix[i]
                self.position_matrix[i] = self.velocity_matrix[i]

    def update_dis(self):
        for i in range(self.num_cars):
            for j in range(i, self.num_cars):
                if i != j:
                    self.dis_matrix[i][j] = self.dis_matrix[j][i] = max(abs(self.position_matrix[i] - self.position_matrix[j]), 1e-2)

    def update_velocity_and_lanes(self):
        for i in range(self.num_cars):
            if self.levels[i]:
                action, lane = self.get_level_2_action(i)
            else:
                action, lane = self.get_level_1_action(i)
            self.new_velocity_matrix[i]  += 2 * (action - 2)
            self.new_velocity_matrix[i] = np.clip(self.new_velocity_matrix[i], self.minimum_speed, self.maximum_speed)
            self.new_lanes[i] = lane

    # level-k decision helpers
    def get_score(self, i, action, actions, assumed_lanes):
        score = 0
        actions[i] = action
        assumed_position = self.position_matrix.copy()
        assumed_velocity = self.velocity_matrix.copy()
        assumed_dis = self.dis_matrix.copy()
        assumed_velocity += 2 * (actions - 2)
        assumed_velocity = np.clip(assumed_velocity, self.minimum_speed, self.maximum_speed)
        assumed_position += assumed_velocity
        # update assumed distances
        for a in range(self.num_cars):
            for b in range(a, self.num_cars):
                if a != b:
                    assumed_dis[a][b] = assumed_dis[b][a] = max(abs(assumed_position[a] - assumed_position[b]), 1e-2)
        score += assumed_velocity[i]
        penalty, best_lane = min(((self.get_distance_penalty(i, assumed_dis, assumed_lanes, lane), lane) for lane in range(2)), key = lambda x: x[0])
        score -= penalty
        return score, best_lane

    def get_distance_penalty(self, i, dis_matrix, assumed_lanes, assumed_lane):
        penalty = 0
        assumed_lanes[i] = assumed_lane
        for j in range(self.num_cars):
            if i != j:
                if dis_matrix[i][j] < 100 and assumed_lanes[i] == assumed_lanes[j]:
                    penalty += 50 * (dis_matrix[i][j] / 100) ** 2
        return penalty

    def get_level_0_action(self, j):
        action = 3
        min_pos = self.highway_length
        min_pos_lane = 1 - self.lanes[j]
        for i in range(self.num_cars):
           if self.position_matrix[i] >= self.position_matrix[j] and i != j:
               if self.position_matrix[i] < min_pos:
                   min_pos = self.position_matrix[i]
                   min_pos_lane = self.lanes[i]
        return action, 1 - min_pos_lane

    def get_level_1_action(self, i):
        actions = np.zeros(self.num_cars, dtype=int)
        assumed_lanes = self.lanes.copy()
        for j in range(self.num_cars):
            actions[j], assumed_lanes[j] =  self.get_level_0_action(j)
        best_action, best_score, best_lane = max(((action, *self.get_score(i, action, actions, assumed_lanes)) for action in range(1,4)), key=lambda x: x[1])
        return best_action, best_lane

    def get_level_2_action(self, i):
        actions = np.zeros(self.num_cars, dtype=int)
        assumed_lanes = self.lanes.copy()
        for j in range(self.num_cars):
            actions[j], assumed_lanes[j] = self.get_level_1_action(j)
        best_action, best_score, best_lane = max(((action, *self.get_score(i, action, actions, assumed_lanes)) for action in range(1,4)), key=lambda x: x[1])
        return best_action, best_lane

