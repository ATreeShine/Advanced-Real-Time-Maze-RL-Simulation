import os
# Use the dummy video driver for headless environments.
os.environ["SDL_VIDEODRIVER"] = "dummy"

import eventlet
eventlet.monkey_patch()

import threading, time, random, sys, io, base64
from collections import deque
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
import pygame
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# Global Configurations
# ---------------------------
cell_size = 20
maze_width, maze_height = 31, 31  # Grid dimensions (cells). Use odd numbers.
maze_pixel_width = maze_width * cell_size
maze_pixel_height = maze_height * cell_size
hidden_panel_width = 200         # Panel to visualize hidden activations.
info_panel_height = 40           # Extra vertical space for text.
total_width = maze_pixel_width + hidden_panel_width
total_height = maze_pixel_height + info_panel_height

# Global variable to hold the current frame (as a base64 PNG string)
current_frame = None
frame_lock = threading.Lock()

# ---------------------------
# Maze Generation & Environment
# ---------------------------
class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Create a grid full of walls.
        self.grid = [['#' for _ in range(width)] for _ in range(height)]
        self.generate_maze()
        # Define start and exit positions.
        self.start = (1, 1)
        self.exit = (height - 2, width - 2)
        self.grid[self.start[0]][self.start[1]] = 'S'
        self.grid[self.exit[0]][self.exit[1]] = 'E'

    def generate_maze(self):
        # Maze generation via recursive backtracking.
        stack = [(1, 1)]
        self.grid[1][1] = ' '  # Mark as open.
        while stack:
            y, x = stack[-1]
            neighbors = []
            # Look two cells away in each direction.
            for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                ny, nx = y + dy, x + dx
                if 0 < ny < self.height - 1 and 0 < nx < self.width - 1:
                    if self.grid[ny][nx] == '#':
                        neighbors.append((ny, nx, dy, dx))
            if neighbors:
                ny, nx, dy, dx = random.choice(neighbors)
                # Carve a passage between.
                self.grid[y + dy // 2][x + dx // 2] = ' '
                self.grid[ny][nx] = ' '
                stack.append((ny, nx))
            else:
                stack.pop()

    def draw(self, surface):
        # Draw the maze on the given Pygame surface.
        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
                cell = self.grid[r][c]
                if cell == '#':
                    color = (0, 0, 0)         # Walls: Black
                elif cell == 'S':
                    color = (0, 255, 0)       # Start: Green
                elif cell == 'E':
                    color = (255, 0, 0)       # Exit: Red
                else:
                    color = (255, 255, 255)   # Open passage: White
                pygame.draw.rect(surface, color, rect)

class MazeEnv:
    def __init__(self, maze):
        self.maze = maze
        self.reset()

    def reset(self):
        self.agent_pos = self.maze.start
        return self.get_state()

    def step(self, action):
        y, x = self.agent_pos
        # Action mapping: 0=up, 1=down, 2=left, 3=right.
        if action == 0:
            new_pos = (y - 1, x)
        elif action == 1:
            new_pos = (y + 1, x)
        elif action == 2:
            new_pos = (y, x - 1)
        elif action == 3:
            new_pos = (y, x + 1)
        else:
            new_pos = self.agent_pos

        # Check boundaries and wall collisions.
        if (0 <= new_pos[0] < self.maze.height and 
            0 <= new_pos[1] < self.maze.width and 
            self.maze.grid[new_pos[0]][new_pos[1]] != '#'):
            self.agent_pos = new_pos  # Valid move.
        # Else, remain at current position.

        reward = -1  # Step penalty.
        done = False
        if self.agent_pos == self.maze.exit:
            reward = 100  # Reward for reaching the exit.
            done = True
        return self.get_state(), reward, done

    def get_state(self):
        # Returns a 3‑channel state (shape: [3, maze_height, maze_width]):
        # Channel 0: Walls (1 if wall, 0 otherwise)
        # Channel 1: Agent (1 if agent is here)
        # Channel 2: Exit (1 if exit cell)
        state = np.zeros((3, self.maze.height, self.maze.width), dtype=np.float32)
        for r in range(self.maze.height):
            for c in range(self.maze.width):
                if self.maze.grid[r][c] == '#':
                    state[0, r, c] = 1.0
        ay, ax = self.agent_pos
        state[1, ay, ax] = 1.0
        ey, ex = self.maze.exit
        state[2, ey, ex] = 1.0
        return state

# ---------------------------
# Replay Buffer for Experience Replay
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.stack(states), actions, rewards, np.stack(next_states), dones
    def __len__(self):
        return len(self.buffer)

# ---------------------------
# Deep Q-Network (DQN) with Hidden Layers
# ---------------------------
class DQN(nn.Module):
    def __init__(self, input_channels, num_actions, height, width):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.height = height
        self.width = width
        self.fc = nn.Linear(64 * height * width, num_actions)

    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.conv1(x)
        x = self.relu1(x)
        hidden = self.conv2(x)  # Hidden activations.
        x = self.relu2(hidden)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        q_values = self.fc(x_flat)
        return q_values, x  # Return Q-values and hidden activations.

# ---------------------------
# DQN Agent Wrapper
# ---------------------------
class DQNAgent:
    def __init__(self, state_shape, num_actions, device):
        self.device = device
        self.num_actions = num_actions
        self.policy_net = DQN(state_shape[0], num_actions, state_shape[1], state_shape[2]).to(device)
        self.target_net = DQN(state_shape[0], num_actions, state_shape[1], state_shape[2]).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.replay_buffer = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.update_target_every = 1000
        self.step_count = 0

    def select_action(self, state):
        self.step_count += 1
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values, _ = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.from_numpy(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Compute current Q values.
        q_values, _ = self.policy_net(states)
        state_action_values = q_values.gather(1, actions)

        # Compute next Q values using target network.
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_states)
            max_next_q_values = next_q_values.max(1, keepdim=True)[0]
        expected_state_action_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Periodically update the target network.
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

# ---------------------------
# Utility: Update Global Frame (for Streaming)
# ---------------------------
def update_frame(surface):
    global current_frame
    data = pygame.image.tostring(surface, 'RGB')
    img = Image.frombytes('RGB', surface.get_size(), data)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    b64_frame = base64.b64encode(buf.read()).decode('utf-8')
    with frame_lock:
        current_frame = b64_frame

# ---------------------------
# Simulation Loop: Training & Demonstration
# ---------------------------
def simulation_loop():
    global current_frame
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 16)
    # Create an off‑screen surface (with extra width for hidden-layer viz).
    screen = pygame.Surface((total_width, total_height))

    # Initialize Maze and Environment.
    maze = Maze(maze_width, maze_height)
    env = MazeEnv(maze)

    # Set up device and agent.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_shape = (3, maze_height, maze_width)
    num_actions = 4
    agent = DQNAgent(state_shape, num_actions, device)

    num_episodes = 300
    max_steps = 500
    global_score = 0

    print("Training agent in the maze...")
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        episode_score = 0
        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.push_transition(state, action, reward, next_state, done)
            loss_val = agent.update()
            state = next_state
            episode_score += reward
            global_score += reward

            # --------- Drawing the Frame ---------
            screen.fill((50, 50, 50))
            # Left panel: Draw the maze.
            maze.draw(screen)
            # Draw the agent (blue square).
            ay, ax = env.agent_pos
            agent_rect = pygame.Rect(ax * cell_size, ay * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (0, 0, 255), agent_rect)
            # Info text (score, episode, epsilon).
            info_text = font.render(f"Ep:{ep} Step:{step}  EpScore:{episode_score}  Global:{global_score}  ε:{agent.epsilon:.2f}", True, (255, 255, 255))
            screen.blit(info_text, (5, maze_pixel_height + 5))

            # Right panel: Hidden layer visualization.
            # Get hidden activations from the current state.
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            with torch.no_grad():
                _, hidden_activations = agent.policy_net(state_tensor)
            # hidden_activations: (1, 64, maze_height, maze_width)
            hidden_np = hidden_activations.cpu().numpy()[0]  # shape: (64, H, W)
            # Average over channels → (H, W)
            hidden_avg = np.mean(hidden_np, axis=0)
            # Normalize to [0, 255] using np.ptp.
            norm_hidden = (hidden_avg - hidden_avg.min()) / (np.ptp(hidden_avg) + 1e-5)
            hidden_img = (norm_hidden * 255).astype(np.uint8)
            # Convert to 3‑channel image.
            hidden_img_color = np.stack([hidden_img] * 3, axis=-1)
            # Create a surface from the numpy array (note transpose for Pygame).
            hidden_surface = pygame.surfarray.make_surface(np.transpose(hidden_img_color, (1, 0, 2)))
            # Scale the hidden surface to the panel size.
            hidden_panel = pygame.transform.smoothscale(hidden_surface, (hidden_panel_width, maze_pixel_height))
            # Blit the hidden panel onto the right side.
            screen.blit(hidden_panel, (maze_pixel_width, 0))

            update_frame(screen)
            time.sleep(0.02)  # For smooth animation.
            if done:
                break
        print(f"Episode {ep} finished with score {episode_score}")

    # ----- Demonstration Phase -----
    print("Training complete! Starting demonstration with learned policy.")
    agent.epsilon = 0.0  # Disable exploration.
    state = env.reset()
    for step in range(max_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        state = next_state

        screen.fill((50, 50, 50))
        maze.draw(screen)
        ay, ax = env.agent_pos
        agent_rect = pygame.Rect(ax * cell_size, ay * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (0, 0, 255), agent_rect)
        demo_text = font.render(f"Demo Mode - Step: {step}", True, (255, 255, 255))
        screen.blit(demo_text, (5, maze_pixel_height + 5))
        # Hidden layer visualization (same as above).
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        with torch.no_grad():
            _, hidden_activations = agent.policy_net(state_tensor)
        hidden_np = hidden_activations.cpu().numpy()[0]
        hidden_avg = np.mean(hidden_np, axis=0)
        norm_hidden = (hidden_avg - hidden_avg.min()) / (np.ptp(hidden_avg) + 1e-5)
        hidden_img = (norm_hidden * 255).astype(np.uint8)
        hidden_img_color = np.stack([hidden_img] * 3, axis=-1)
        hidden_surface = pygame.surfarray.make_surface(np.transpose(hidden_img_color, (1, 0, 2)))
        hidden_panel = pygame.transform.smoothscale(hidden_surface, (hidden_panel_width, maze_pixel_height))
        screen.blit(hidden_panel, (maze_pixel_width, 0))

        update_frame(screen)
        time.sleep(0.05)
        if done:
            break
    print("Demonstration complete. Press Ctrl+C to exit.")
    while True:
        update_frame(screen)
        time.sleep(0.1)

# ---------------------------
# Flask & SocketIO for Real-Time Streaming
# ---------------------------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

@app.route("/")
def index():
    # A simple HTML page that displays the real-time simulation.
    return render_template_string('''
    <!doctype html>
    <html>
      <head>
         <title>Advanced Real-Time Maze RL Simulation</title>
         <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
         <style>
            body { background-color: #333; color: #fff; text-align: center; }
         </style>
      </head>
      <body>
         <h1>Advanced Real-Time Maze RL Simulation</h1>
         <img id="frame" width="{{ width }}" height="{{ height }}" style="border: 2px solid #fff;">
         <script>
            var socket = io();
            socket.on('frame', function(data) {
               document.getElementById("frame").src = "data:image/png;base64," + data;
            });
         </script>
      </body>
    </html>
    ''', width=total_width, height=total_height)

def frame_emitter():
    # Emit the latest frame at ~30 FPS.
    while True:
        eventlet.sleep(1/30.0)
        with frame_lock:
            frame_data = current_frame
        if frame_data is not None:
            socketio.emit("frame", frame_data)

# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == "__main__":
    # Start the simulation loop in a background thread.
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()
    # Start the frame emitter background task.
    socketio.start_background_task(frame_emitter)
    # Run the Flask-SocketIO server on port 9999 (accessible externally).
    socketio.run(app, host="0.0.0.0", port=9999)
