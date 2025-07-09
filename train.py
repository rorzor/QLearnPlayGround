from environment import GridEnvironment
from agent import QLearningAgent
import matplotlib.pyplot as plt
import time
from settings import *

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
                save_interval=20,
                max_moves=50,
                moving_average=50):
    env = GridEnvironment()
    agent = QLearningAgent(state_size=env.get_flattened_state().shape[1], action_size=4)
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
        move_count = 0  # Initialize move counter

        while not done:
            # run through the episode
            action = agent.act(state, epsilon)  # Choose action based on epsilon-greedy policy
            # print(f"Episode {e+1}, Action: {action}, State: {state}")
            next_state, reward, done = env.step(action) # Take action in the environment
            next_state = next_state.reshape(1, -1) 

            agent.remember(state, action, reward, next_state, done)     # Store the experience in replay buffer
            agent.train(state, action, reward, next_state, done, gamma) # Train the agent with the experience

            state = next_state
            total_reward += reward
            move_count += 1  # Increment move counter

            # Handle data and reset at end of episode
            if done or move_count >= max_moves:
                # print("Episode finished or max moves reached. Updating plot...")
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

        # train through a replay after each episode
        if len(agent.replay_buffer) > BATCH_MULTIPLIER * BATCH_SIZE and (e + 1) % REPLAY_INTERVAL == 0:
            print(f"Training agent with replay buffer size: {len(agent.replay_buffer)}")
            agent.replay(BATCH_SIZE, gamma)

        # Update target model every N episodes
        if (e + 1) % TARGET_UPDATE_INTERVAL == 0:
            agent.update_target_model()
            print(f"Target model updated at episode {e+1}")

        # Store the episode evolution every 10 episodes
        if (e + 1) % save_interval == 0:
            # Save the model every N episodes
            #model_filename = f"qlearning_model_episode_{e+1}.keras"
            model_filename = f"qlearning_model.keras"

            agent.model.save(model_filename)
            print(f"Model saved to {model_filename}")

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        if sum(avg_rewards)/len(avg_rewards) > 95 and episodes > 50:
            break
    
    # Keep the plot open after the loop ends
    plt.ioff()
    plt.show()
    return agent

# Run the training
agent = train_agent()
