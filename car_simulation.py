import os
# Use the dummy video driver for headless environments.
os.environ["SDL_VIDEODRIVER"] = "dummy"

import eventlet
eventlet.monkey_patch()

import threading, time, random, sys, io, base64, math
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
cell_size = 10  # cell size for drawing
grid_width, grid_height = 51, 51  # odd dimensions for symmetry.
grid_pixel_width = grid_width * cell_size
grid_pixel_height = grid_height * cell_size
hidden_panel_width = 200         # Panel to visualize hidden activations.
info_panel_height = 40           # Extra vertical space for text.
total_width = grid_pixel_width + hidden_panel_width
total_height = grid_pixel_height + info_panel_height

NUM_CARS = 2  # number of simultaneous agents

# Global variable to hold the current frame (as a base64 PNG string)
current_frame = None
frame_lock = threading.Lock()

# ---------------------------
# Race Track Generation & Environment
# ---------------------------
class RaceTrack:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Create grid: 'R' for road, '#' for off-road, 'S' for start.
        self.grid = [['#' for _ in range(width)] for _ in range(height)]
        self.generate_track()

    def generate_track(self):
        cx, cy = self.width // 2, self.height // 2
        self.track_radius = min(cx, cy) - 3
        road_width = 6
        inner_radius = self.track_radius - road_width // 2
        outer_radius = self.track_radius + road_width // 2
        for r in range(self.height):
            for c in range(self.width):
                dx = c - cx + 0.5
                dy = r - cy + 0.5
                dist = math.sqrt(dx*dx + dy*dy)
                if inner_radius <= dist <= outer_radius:
                    self.grid[r][c] = 'R'
        # Mark the start line (top of circle)
        start_r = cy - self.track_radius
        start_c = cx
        self.grid[start_r][start_c] = 'S'
        self.start = (start_r, start_c)

    def draw(self, surface):
        # Draw the grid onto the provided surface.
        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
                cell = self.grid[r][c]
                if cell == '#':
                    color = (34, 139, 34)  # off-road: Forest Green
                elif cell == 'R':
                    color = (169, 169, 169)  # road: Dark Gray
                elif cell == 'S':
                    color = (255, 0, 0)      # start line: Red
                pygame.draw.rect(surface, color, rect)

class RaceTrackEnv:
    def __init__(self, track: RaceTrack):
        self.track = track
        self.cx, self.cy = track.width // 2, track.height // 2
        self.reset()

    def reset(self):
        start_r, start_c = self.track.start
        self.car_pos = np.array([start_c + 0.5, start_r + 0.5], dtype=np.float32)
        self.car_angle = 0.0  # tangent at start (pointing right)
        self.left_start_zone = False
        self.laps = 0
        self.steps = 0
        return self.get_state()

    def step(self, action):
        # Action: 0 = turn left, 1 = straight, 2 = turn right.
        turn_angle = math.radians(15)
        if action == 0:
            self.car_angle -= turn_angle
        elif action == 2:
            self.car_angle += turn_angle
        speed = 0.8
        dx = speed * math.cos(self.car_angle)
        dy = speed * math.sin(self.car_angle)
        self.car_pos += np.array([dx, dy], dtype=np.float32)
        self.steps += 1

        grid_x = int(self.car_pos[0])
        grid_y = int(self.car_pos[1])
        if not (0 <= grid_x < self.track.width and 0 <= grid_y < self.track.height):
            return self.get_state(), -50, True

        cell = self.track.grid[grid_y][grid_x]
        if cell in ['R', 'S']:
            reward = 1
        else:
            return self.get_state(), -5, True

        if cell == 'S' and self.left_start_zone:
            reward += 100
            self.laps += 1
            done = True
        else:
            done = False
        if cell != 'S':
            self.left_start_zone = True
        if self.steps >= 500:
            done = True

        return self.get_state(), reward, done

    def get_state(self):
        # 3-channel state: road mask, car position, start line.
        state = np.zeros((3, self.track.height, self.track.width), dtype=np.float32)
        for r in range(self.track.height):
            for c in range(self.track.width):
                if self.track.grid[r][c] in ['R', 'S']:
                    state[0, r, c] = 1.0
                if self.track.grid[r][c] == 'S':
                    state[2, r, c] = 1.0
        grid_x = int(self.car_pos[0])
        grid_y = int(self.car_pos[1])
        if 0 <= grid_y < self.track.height and 0 <= grid_x < self.track.width:
            state[1, grid_y, grid_x] = 1.0
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
        x = self.conv1(x)
        x = self.relu1(x)
        hidden = self.conv2(x)
        x = self.relu2(hidden)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        q_values = self.fc(x_flat)
        return q_values, x

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
        self.total_laps = 0
        self.total_score = 0

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

        q_values, _ = self.policy_net(states)
        state_action_values = q_values.gather(1, actions)
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_states)
            max_next_q_values = next_q_values.max(1, keepdim=True)[0]
        expected_state_action_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
# Simulation Loop: Ultra-Optimized Multi-Car Training & Demonstration
# ---------------------------
def simulation_loop():
    global current_frame
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 12)
    # Off-screen display surface.
    screen = pygame.Surface((total_width, total_height))

    # Create a shared track and pre-render its static parts.
    track = RaceTrack(grid_width, grid_height)
    track_surface = pygame.Surface((grid_pixel_width, grid_pixel_height))
    track.draw(track_surface)  # Pre-render track once.

    # Create environments and agents.
    envs = [RaceTrackEnv(track) for _ in range(NUM_CARS)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_shape = (3, grid_height, grid_width)
    num_actions = 3  # left, straight, right.
    agents = [DQNAgent(state_shape, num_actions, device) for _ in range(NUM_CARS)]
    states = [env.reset() for env in envs]
    episode_scores = [0] * NUM_CARS
    episodes_done = [0] * NUM_CARS
    total_episodes = 300

    colors = [(random.randint(50,255), random.randint(50,255), random.randint(50,255)) for _ in range(NUM_CARS)]
    print("Ultra‑optimized training of 20 cars...")

    # Main training loop.
    while any(e < total_episodes for e in episodes_done):
        for i in range(NUM_CARS):
            if episodes_done[i] >= total_episodes:
                continue
            action = agents[i].select_action(states[i])
            next_state, reward, done = envs[i].step(action)
            agents[i].push_transition(states[i], action, reward, next_state, done)
            agents[i].update()
            states[i] = next_state
            episode_scores[i] += reward
            if done:
                agents[i].total_laps += envs[i].laps
                agents[i].total_score += episode_scores[i]
                episodes_done[i] += 1
                states[i] = envs[i].reset()
                episode_scores[i] = 0
        # --------- Drawing the Frame (optimized) ---------
        screen.fill((0, 0, 0))
        screen.blit(track_surface, (0, 0))
        for i in range(NUM_CARS):
            car_x = int(envs[i].car_pos[0] * cell_size)
            car_y = int(envs[i].car_pos[1] * cell_size)
            pygame.draw.circle(screen, colors[i], (car_x, car_y), cell_size // 2)
        info_text = font.render(f"Episodes: {max(episodes_done)} / {total_episodes} per car", True, (255,255,255))
        screen.blit(info_text, (5, grid_pixel_height + 5))
        update_frame(screen)
        eventlet.sleep(0.001)  # Minimal sleep for ultra-fast simulation

    # Pick best agent by total laps.
    best_index = max(range(NUM_CARS), key=lambda i: agents[i].total_laps)
    print(f"Training complete! Best car #{best_index} with {agents[best_index].total_laps} laps.")
    
    # ----- Demonstration Phase -----
    print("Starting ultra-fast demo with the best car.")
    demo_env = RaceTrackEnv(track)
    demo_agent = agents[best_index]
    demo_agent.epsilon = 0.0  # disable exploration
    state = demo_env.reset()
    demo_steps = 0
    while demo_steps < 500:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        action = demo_agent.select_action(state)
        next_state, reward, done = demo_env.step(action)
        state = next_state

        screen.fill((0, 0, 0))
        screen.blit(track_surface, (0, 0))
        car_x = int(demo_env.car_pos[0] * cell_size)
        car_y = int(demo_env.car_pos[1] * cell_size)
        pygame.draw.circle(screen, (0, 0, 255), (car_x, car_y), cell_size // 2)
        demo_text = font.render(f"Demo (Car #{best_index}) - Step: {demo_steps}  Laps: {demo_env.laps}", True, (255,255,255))
        screen.blit(demo_text, (5, grid_pixel_height + 5))
        # Hidden layer visualization.
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        with torch.no_grad():
            _, hidden_activations = demo_agent.policy_net(state_tensor)
        hidden_np = hidden_activations.cpu().numpy()[0]
        hidden_avg = np.mean(hidden_np, axis=0)
        norm_hidden = (hidden_avg - hidden_avg.min()) / (np.ptp(hidden_avg) + 1e-5)
        hidden_img = (norm_hidden * 255).astype(np.uint8)
        hidden_img_color = np.stack([hidden_img] * 3, axis=-1)
        hidden_surface = pygame.surfarray.make_surface(np.transpose(hidden_img_color, (1, 0, 2)))
        hidden_panel = pygame.transform.smoothscale(hidden_surface, (hidden_panel_width, grid_pixel_height))
        screen.blit(hidden_panel, (grid_pixel_width, 0))
        update_frame(screen)
        eventlet.sleep(0.001)
        demo_steps += 1
    print("Demonstration complete. Press Ctrl+C to exit.")
    while True:
        update_frame(screen)
        eventlet.sleep(0.01)

# ---------------------------
# Flask & SocketIO for Real-Time Streaming
# ---------------------------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

@app.route("/")
def index():
    return render_template_string('''
    <!doctype html>
    <html>
      <head>
         <title>Ultra-Optimized Multi-Car Race Track RL Simulation</title>
         <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
         <style>
            body { background-color: #333; color: #fff; text-align: center; }
         </style>
      </head>
      <body>
         <h1>Ultra-Optimized Multi-Car Race Track RL Simulation</h1>
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
    while True:
        eventlet.sleep(1/60.0)
        with frame_lock:
            frame_data = current_frame
        if frame_data is not None:
            socketio.emit("frame", frame_data)

if __name__ == "__main__":
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()
    socketio.start_background_task(frame_emitter)
    socketio.run(app, host="0.0.0.0", port=9999)
