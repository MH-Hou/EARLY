import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset, random_split
from stable_baselines3 import PPO, A2C
from datetime import datetime

from nav_env import NavEnv
from nav_oracle_rrt import RRT_Oracle


TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len( self.observations)


def pretrain_agent(student,
                   env,
                   train_expert_dataset,
                   test_expert_dataset,
                   batch_size=64,
                   epochs=500,
                   scheduler_gamma=0.7,
                   learning_rate=1.0,
                   log_interval=100,
                   no_cuda=True,
                   seed=1,
                   test_batch_size=64,
                   ):
    use_cuda = not no_cuda and th.cuda.is_available()
    th.manual_seed(seed)
    device = th.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if isinstance(env.action_space, gym.spaces.Box):
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Extract initial policy
    model = student.policy.to(device)


    def train(model, device, train_loader, optimizer, epoch_id):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if isinstance(env.action_space, gym.spaces.Box):
                # A2C/PPO policy outputs actions, values, log_prob
                # SAC/TD3 policy outputs actions only
                if isinstance(student, (A2C, PPO)):
                    action, _, _ = model(data)
                else:
                    # SAC/TD3:
                    action = model(data)
                action_prediction = action.double()
            else:
                # Retrieve the logits for A2C/PPO when using discrete actions
                dist = model.get_distribution(data)
                action_prediction = dist.distribution.logits
                target = target.long()

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch_id,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        with th.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                if isinstance(env.action_space, gym.spaces.Box):
                    # A2C/PPO policy outputs actions, values, log_prob
                    # SAC/TD3 policy outputs actions only
                    if isinstance(student, (A2C, PPO)):
                        action, _, _ = model(data)
                    else:
                        # SAC/TD3:
                        action = model(data)
                    action_prediction = action.double()
                else:
                    # Retrieve the logits for A2C/PPO when using discrete actions
                    dist = model.get_distribution(data)
                    action_prediction = dist.distribution.logits
                    target = target.long()

                test_loss = criterion(action_prediction, target)
        test_loss /= len(test_loader.dataset)
        print(f"Test set: Average loss: {test_loss:.4f}")

    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
    # and testing
    train_loader = th.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = th.utils.data.DataLoader(
        dataset=test_expert_dataset, batch_size=test_batch_size, shuffle=True, **kwargs,
    )

    # Define an Optimizer and a learning rate schedule.
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Now we are finally ready to train the policy model.
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # Implant the trained policy network back into the RL student agent
    student.policy = model

    return student


def main():
    env = NavEnv(render=False)
    a2c_student = PPO('MlpPolicy', env, seed=1, gamma=1.0, verbose=1, policy_kwargs=dict(net_arch=[64, 64]))

    expert_observations = None
    expert_actions = None

    rrt_oracle = RRT_Oracle()
    total_trajs_num = 100
    init_xs = np.linspace(start=0.01, stop=19.99, num=total_trajs_num)
    traj_id = 0
    for init_x in init_xs:
        init_y = 4.0
        pos_init = (init_x, init_y)
        pos_goal = (10.0, 16.0)
        path = rrt_oracle.path_planning(pos_init=pos_init,
                                        pos_goal=pos_goal)

        # 2d np array of (total_step_num, action/state_dimension)
        oracle_states, oracle_actions = rrt_oracle.recover_demo(path=path,
                                                                delta_t=1.0,
                                                                pos_goal=pos_goal)
        if expert_observations is None:
            expert_observations = np.concatenate((np.ones((1, 4)) * np.inf,
                                                  oracle_states))
            pure_expert_observations = oracle_states
        else:
            expert_observations = np.concatenate((expert_observations,
                                                  np.ones((1, 4)) * np.inf,
                                                  oracle_states))
            pure_expert_observations = np.concatenate((pure_expert_observations, oracle_states))

        if expert_actions is None:
            expert_actions = np.concatenate((np.ones((1, 2)) * np.inf,
                                             oracle_actions))
            pure_expert_actions = oracle_actions
        else:
            expert_actions = np.concatenate((expert_actions,
                                             np.ones((1, 2)) * np.inf,
                                             oracle_actions))
            pure_expert_actions = np.concatenate((pure_expert_actions, oracle_actions))

        print("Solved path {}".format(traj_id + 1))
        # print(np.shape(expert_observations))
        print("**************************************")
        traj_id += 1

    # save demos
    np.savetxt("rrt_oracle/demos/oracle_states.csv", expert_observations, delimiter=" ")
    np.savetxt("rrt_oracle/demos/oracle_actions.csv", expert_actions, delimiter=" ")
    np.savetxt("rrt_oracle/demos/pure_oracle_states.csv", pure_expert_observations, delimiter=" ")
    np.savetxt("rrt_oracle/demos/pure_oracle_actions.csv", pure_expert_actions, delimiter=" ")
    print("Demos saved!")


    """expert_dataset = ExpertDataSet(pure_expert_observations, pure_expert_actions)
    train_size = int(0.8 * len(expert_dataset))
    test_size = len(expert_dataset) - train_size
    train_expert_dataset, test_expert_dataset = random_split(
        expert_dataset, [train_size, test_size]
    )
    print("test_expert_dataset: ", len(test_expert_dataset))
    print("train_expert_dataset: ", len(train_expert_dataset))

    print("Going to pretrain PPO with BC ... ")
    student = pretrain_agent(student=a2c_student,
                             env=env,
                             train_expert_dataset=expert_dataset,
                             test_expert_dataset=expert_dataset,
                             epochs=500,
                             scheduler_gamma=0.1,
                             learning_rate=1.0,
                             log_interval=100,
                             no_cuda=True,
                             seed=1,
                             batch_size=30,
                             test_batch_size=30
                             )

    student.save("rrt_oracle/bc_wo_ppo" + TIMESTAMP)

    env.close()
    print("BC without PPO training finished!")"""



if __name__ == '__main__':
    main()