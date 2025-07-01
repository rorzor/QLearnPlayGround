import pygame
import numpy as np
import tensorflow as tf
import keras

from agent import QLearningAgent
from environment import GridEnvironment

# Initialize Pygame
pygame.init()

# Define Colors
GRAY = (200, 200, 200)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

# Define Grid and Cell Size
GRID_SIZE = 3
CELL_SIZE = 100
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE

# Create the Screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Agent Simulation")

# Load the Trained Model
model_path = "qlearning_model.keras"  # Replace with your model file
model = keras.models.load_model(model_path)

# Create the Environment
env = GridEnvironment()

# Define Directions for Visualization
ARROW_OFFSETS = {
    0: (0, -1),  # Up
    1: (0, 1),   # Down
    2: (-1, 0),  # Left
    3: (1, 0)    # Right
}

def draw_grid(env, action=None):
    screen.fill(WHITE)
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GRAY, rect)
            pygame.draw.rect(screen, WHITE, rect, 2)
            
            # Draw Goal
            if (i, j) == env.goal_pos:
                pygame.draw.rect(screen, GREEN, rect)
            # Draw Trap
            elif (i, j) == env.trap_pos:
                pygame.draw.rect(screen, RED, rect)
            # Draw Agent
            if (i, j) == env.agent_pos:
                pygame.draw.circle(screen, BLUE, rect.center, CELL_SIZE // 4)

    # Draw the decision arrow if an action was taken
    if action is not None:
        player_pos = np.flip(env.agent_pos)
        arrow_start = np.array(player_pos) * CELL_SIZE + np.array([CELL_SIZE // 2, CELL_SIZE // 2])
        arrow_end = arrow_start + np.array(ARROW_OFFSETS[action]) * (CELL_SIZE // 2)
        pygame.draw.line(screen, BLUE, arrow_start, arrow_end, 5)

    pygame.display.flip()

def main():
    clock = pygame.time.Clock()
    running = True
    state = env.reset().reshape(1, -1)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Agent takes action
                    q_values = model(state)
                    action = np.argmax(q_values[0])

                if event.key == pygame.K_UP:
                    action = 0
                if event.key == pygame.K_DOWN:
                    action = 1
                if event.key == pygame.K_LEFT:
                    action = 2
                if event.key == pygame.K_RIGHT:
                    action = 3

                next_state, reward, done = env.step(action)
                state = next_state.reshape(1, -1)
                    
                draw_grid(env, action)
                    
                if done:
                    print(f"Episode finished with reward: {reward}")
                    state = env.reset().reshape(1, -1)
                    draw_grid(env)

        clock.tick(5)  # Control the speed of the simulation
    
    pygame.quit()

if __name__ == "__main__":
    main()
