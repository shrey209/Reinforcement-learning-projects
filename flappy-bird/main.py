import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch
import itertools
from replay import ReplayMemory

device = "cuda" if torch.cuda.is_available() else "cpu"


class Agent:
    def run(self, istraining=True, render=False):
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]
        policy_dqn = DQN(num_states, num_actions).to_device(device)

        if istraining:
            memory = ReplayMemory(10000)

        for episode in itertools.count(1000):
            state, _ = env.reset()
            while True:
                action = env.action_space.sample()
                new_state, reward, terminated, _, info = env.step(action)

                if istraining:
                    memory.append((state, action, new_state, reward, terminated))

                state = new_state

                if terminated:
                    break

        env.close()
        print("flappy-bird game starting...")


def main():
    # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    agent = Agent()
    agent.run(istraining=True, render=True)


if __name__ == "__main__":
    main()
