from environment import GridEnvironment
from agent import QLearningAgent
import pickle
import matplotlib.pyplot as plt
import time


# def normalise(reward):
#     norm_reward = (reward + 1250) / (100 + 1250)
#     return norm_reward

def shuffle_in(value, lst):
    if len(lst) > 0:
        lst.pop(0)  # Remove the first item
    lst.append(value)  # Add the new value at the end
    return lst

def train_agent(episodes=10000,
                gamma=0.75,
                epsilon=1.0,
                epsilon_decay=0.9995,
                epsilon_min=0.2,
                replay_interval=10,
                save_interval=20,
                max_moves=50,
                moving_average=50):
    env = GridEnvironment()
    agent = QLearningAgent(state_size=env.get_flattened_state().shape[1], action_size=4)
    game_evolution = []  # List to store the board states for visualization
    avg_rewards = []
    rewards = []

    # Enable interactive mode
    plt.ion()

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Initialize an empty list to store loss values
    loss_values = []

    # Set up the initial plot
    line, = ax.plot(loss_values)
    ax.set_xlim(0, 100)
    ax.set_ylim(-100, 100)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')

    # Record the start time
    start_time = time.time()

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_evolution = []  # List to store each step of the current episode
        move_count = 0  # Initialize move counter

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done = env.step(action)
            next_state = next_state.reshape(1, -1)

            agent.remember(state, action, reward, next_state, done)

            agent.train(state, action, reward, next_state, done, gamma)

            state = next_state
            total_reward += reward
            move_count += 1  # Increment move counter

            # Save the current state of the grid for visualization
            #episode_evolution.append((env.grid.copy(), env.agent_pos))

            if done or move_count >= max_moves:
                if len(avg_rewards) < moving_average:
                    avg_rewards.append(total_reward)
                else:
                    shuffle_in(total_reward,avg_rewards)
                
                rewards.append(sum(avg_rewards)/len(avg_rewards))
                
                # Update the data of the line object
                line.set_xdata(range(len(rewards)))
                line.set_ydata(rewards)
                
                # Adjust the axis limits if necessary
                ax.set_xlim(0, len(rewards))
                
                # Redraw the plot
                plt.draw()
                plt.pause(0.1)

                # Calculate how much time has passed
                elapsed_time = time.time() - start_time
                print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}, Moves: {move_count}, Avg Reward: {sum(avg_rewards)/len(avg_rewards)}, elapsed time = {elapsed_time}")

                break
            
        if (e + 1) % replay_interval == 0:
            agent.replay(len(agent.replay_buffer),0.8)

        # Store the episode evolution every 10 episodes
        if (e + 1) % save_interval == 0:
            game_evolution.append(episode_evolution)
            # Save the model every 10 episodes
            model_filename = f"qlearning_model_episode_{e+1}.keras"
            agent.model.save(model_filename)
            print(f"Model saved to {model_filename}")


        # if (e + 1) % save_interval == 0:
        #     # Save the game evolution
        #     with open(f"game_evolution.pkl", "wb") as f:
        #         pickle.dump(game_evolution, f)
        #     print(f"Game evolution saved to game_evolution_episode_{e+1}.pkl")

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
    
    # Keep the plot open after the loop ends
    plt.ioff()
    plt.show()
    return agent

# Run the training
agent = train_agent()
