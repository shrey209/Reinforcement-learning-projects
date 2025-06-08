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
        
        self.replay_memory_size = 10000
        self.batch_size = 32
        self.epsilon_init = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05
        self.discount_factor = 0.99
        self.loss_fn = torch.nn.MSELoss()
        self.learning_rate = 0.001

    def run(self, istraining=True, render=False):
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        policy_dqn = DQN(num_states, num_actions).to(device)
        target_dqn = DQN(num_states, num_actions).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        step_count = 0
        reward_per_episode = []
        epsilon_history = []

        if istraining:
            memory = ReplayMemory(self.replay_memory_size)

        epsilon = self.epsilon_init

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated:
                if istraining and np.random.random() < epsilon:
                    action = torch.tensor(env.action_space.sample(), device=device)
                else:
                    with torch.no_grad():
                        q_values = policy_dqn(state.unsqueeze(0))
                        action = torch.argmax(q_values, dim=1).squeeze()

                new_state, reward, terminated, _, _ = env.step(action.item())
                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                episode_reward += reward

                if istraining:
                    memory.append((state, action, new_state, reward, terminated))

                step_count += 1
                state = new_state

            reward_per_episode.append(episode_reward)
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            if istraining and len(memory) >= self.batch_size:
                batch = memory.sample(self.batch_size)
                self.optimize_model(batch, policy_dqn, target_dqn)

                if step_count % 10 == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        env.close()
        print("flappy-bird game starting...")

def optimize_model(self, mini_batch, policy_dqn, target_dqn):

      
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1-terminations) * self.discount_factor_g * \
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                # Calculate target Q values (expected returns)
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
                '''
                    target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                        .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                            [0]             ==> tensor([3,6])
                '''

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases

def main():
    agent = Agent()
    agent.run(istraining=True, render=True)

if __name__ == "__main__":
    main()
