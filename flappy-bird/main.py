
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:

    def run(self,istraining=True, render=False):
         env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
         num_actions=env.action_space.n
         num_states = env.observation_space.shape[0]
         policy_dqn = DQN(num_states, num_actions).to_device(device)
         obs, _ = env.reset()
         while True:
    
             action = env.action_space.sample()

  
             obs, reward, terminated, _, info = env.step(action)
    
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