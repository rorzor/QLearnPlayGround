import pickle

def load_and_visualize_game_evolution(file_path):
    # Load the pickle file
    with open(file_path, 'rb') as file:
        game_evolution = pickle.load(file)
    
    # Visualize the game evolution
    for episode_idx, episode in enumerate(game_evolution):
        print(f"Visualizing Episode {episode_idx + 1}:")
        for step_idx, (grid, agent_pos) in enumerate(episode):
            print(f"Step {step_idx + 1}:")
            grid_display = grid.copy()
            grid_display[agent_pos] = 1  # Represent the agent's position
            print(grid_display)
            print("\n")

# Load and visualize the game evolution from the provided file
file_path = 'game_evolution.pkl'
load_and_visualize_game_evolution(file_path)