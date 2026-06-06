# 主程序：初始化环境、神经网络并运行训练循环
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from env import Environment
from networks import MLP, GCN, GAT, Qmix


if __name__ == '__main__':
    for alg in ['MLP', 'GCN', 'GAT']:
        writer = SummaryWriter('C:/Users/35511/Desktop/ddgn-new/runs/' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        eps = 0.99
        steps = 10000
        num_cars = 10
        num_resources = 6

        # create environment with same defaults as original
        env = Environment(num_cars=num_cars)
        mix_net = Qmix(num_cars).to(env.device)

        agents_net = []
        for i in range(num_cars):
            if alg == 'MLP':
                model = MLP(num_cars, 64, num_resources).to(env.device)
            # elif alg == 'GCN':
            #     model = GCN(num_cars, 64, num_resources).to(env.device)
            # elif alg == 'GAT':
            #     model = GAT(1, 64, num_resources).to(env.device)
            agents_net.append(model)

        agent_optimizer = torch.optim.AdamW([param for agent in agents_net for param in agent.parameters()], lr=0.001)
        mix_net_optimizer = torch.optim.AdamW(mix_net.parameters(), lr=0.01)
        add_linear = torch.nn.Linear(num_cars * num_resources, num_cars * num_resources).to(env.device)

        for step in range(steps):
            env.available_vehicles, transmissions = env.get_available_vehicles()

            if alg == 'MLP':
                q = []
                for i in range(num_cars):
                    obs = torch.tensor(env.dis_matrix[i] / env.highway_length, dtype=torch.float32, device=env.device)
                    q.append(agents_net[i](obs))

            q = torch.stack(q, dim=0).unsqueeze(0)
            q = add_linear(q.reshape(-1)).reshape(num_cars, num_resources)

            if np.random.rand() < eps:
                actions = torch.tensor(np.random.randint(0, num_resources, size=num_cars), dtype=torch.int64, device=env.device)
            else:
                action_probs = torch.nn.functional.softmax(q, dim=1)
                actions = torch.multinomial(action_probs, 1).reshape(-1)

            eps *= 0.99
            eps = max(eps, 0.01)

            reward, successful_transmissions = env.get_reward(actions.cpu().numpy())
            successful_rate = successful_transmissions / max(transmissions, 1)
            writer.add_scalar('successful_rate', successful_rate, step)

            chosen_Q = torch.gather(q, 1, actions.unsqueeze(1)).squeeze(1)
            q_pred = mix_net(chosen_Q)
            loss = torch.nn.functional.huber_loss(q_pred, sum(reward).detach())

            agent_optimizer.zero_grad()
            mix_net_optimizer.zero_grad()
            loss.backward()
            agent_optimizer.step()
            mix_net_optimizer.step()

            env.update_velocity_and_lanes()
            env.velocity_matrix = env.new_velocity_matrix.copy()
            env.lanes = env.new_lanes.copy()
            env.update_position()
            env.update_dis()

            print(sum(reward).item())
            writer.add_scalar('loss', loss.item(), step)
            writer.add_scalar('reward', sum(reward).item(), step)
