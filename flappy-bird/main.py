import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch 
from torch import nn
import itertools
from replay_mem import ReplayMemory
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(self):
        # Hardcoded hyperparameters
        self.replay_memory_size = 100000
        self.batch_size = 32
        self.epsilon_init = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05
        self.loss_function = torch.nn.MSELoss()
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(policy_dqn.parameters())

    def run(self, istraining=True, render=False):
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]
        policy_dqn = DQN(num_states, num_actions).to(device)
        target_dqn = DQN(num_states, num_actions).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        step_count=0
        reward_per_episode = []
        epsilon_history = []

        if istraining:
            memory = ReplayMemory(self.replay_memory_size)

        epsilon = self.epsilon_init

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated:
                if istraining and np.random.random() < epsilon:
                    action = torch.tensor(env.action_space.sample())
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, _, info = env.step(action.item())
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                episode_reward += reward

                if istraining:
                    memory.append((state, action, new_state, reward, terminated))

                step_count+=1

                state = new_state

            reward_per_episode.append(episode_reward)
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            if(len(memory) >= self.batch_size):
                batch = memory.sample(self.batch_size)
                self.optimize_model(batch,policy_dqn, target_dqn)

                if step_count % 10 == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0
            
                

        env.close()
        print("flappy-bird game starting...")


def optimize(self, batch, policy_dqn, target_dqn):
    for state, action, new_state, reward, terminated in batch:
        if terminated:
            target_dqn_value = reward
        else:
            with torch.no_grad():
                target_q=reward + self.discount_factor * torch.max(target_dqn(new_state)).max()

        current=policy_dqn(state)
        loss=self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
def main():
    agent = Agent()
    agent.run(istraining=True, render=True)

if __name__ == "__main__":
    main()
