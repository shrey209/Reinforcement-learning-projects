
import flappy_bird_gymnasium
import gymnasium


def main():
    # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    env = gymnasium.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset()
    while True:
    
        action = env.action_space.sample()

  
        obs, reward, terminated, _, info = env.step(action)
    
        if terminated:
            break

    env.close()
    print("flappy-bird game starting...")




if __name__ == "__main__":
    main()