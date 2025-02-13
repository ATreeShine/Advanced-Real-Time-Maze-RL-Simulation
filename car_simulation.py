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
cell_size = 10  # smaller cell size for higher resolution track
grid_width, grid_height = 51, 51  # Use odd numbers for symmetry.
grid_pixel_width = grid_width * cell_size
grid_pixel_height = grid_height * cell_size
hidden_panel_width = 200         # Panel to visualize hidden activations.
info_panel_height = 40           # Extra vertical space for text.
total_width = grid_pixel_width + hidden_panel_width
total_height = grid_pixel_height + info_panel_height

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
        # Create a grid representing the environment.
        # Each cell will be marked as:
        # 'R' for road, '#' for off-road, and 'S' for start line.
        self.grid = [['#' for _ in range(width)] for _ in range(height)]
        self.generate_track()

    def generate_track(self):
        cx, cy = self.width // 2, self.height // 2
        # Define track parameters.
        self.track_radius = min(cx, cy) - 3
        road_width = 6
        inner_radius = self.track_radius - road_width // 2
        outer_radius = self.track_radius + road_width // 2
        for r in range(self.height):
            for c in range(self.width):
                # Compute distance from center (using cell center coordinates).
                dx = c - cx + 0.5
                dy = r - cy + 0.5
                dist = math.sqrt(dx*dx + dy*dy)
                if inner_radius <= dist <= outer_radius:
                    self.grid[r][c] = 'R'
        # Mark the start line. Here we choose the top of the circle.
        start_r = cy - self.track_radius
        start_c = cx
        self.grid[start_r][start_c] = 'S'
        self.start = (start_r, start_c)

    def draw(self, surface):
        # Draw the track on the given Pygame surface.
        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
                cell = self.grid[r][c]
                if cell == '#':
                    color = (34, 139, 34)  # Off-road: Forest Green
                elif cell == 'R':
                    color = (169, 169, 169)  # Road: Dark Gray
                elif cell == 'S':
                    color = (255, 0, 0)   # Start line: Red
                pygame.draw.rect(surface, color, rect)

class RaceTrackEnv:
    def __init__(self, track: RaceTrack):
        self.track = track
        self.cx, self.cy = track.width // 2, track.height // 2
        self.reset()

    def reset(self):
        # Place the car at the start line (using continuous coordinates).
        start_r, start_c = self.track.start
        # Convert grid cell to continuous coordinates (center of cell).
        self.car_pos = np.array([start_c + 0.5, start_r + 0.5], dtype=np.float32)
        # For a circular track, set initial orientation tangent to the circle.
        # At the top of the circle the tangent is horizontal (pointing right).
        self.car_angle = 0.0  # in radians
        # For lap detection, flag when the car leaves the start zone.
        self.left_start_zone = False
        # Count number of laps completed.
        self.laps = 0
        self.steps = 0
        return self.get_state()

    def step(self, action):
        # Action mapping: 0 = turn left, 1 = go straight, 2 = turn right.
        turn_angle = math.radians(15)
        if action == 0:
            self.car_angle -= turn_angle
        elif action == 2:
            self.car_angle += turn_angle
        # Always move forward by a fixed step.
        speed = 0.8
        dx = speed * math.cos(self.car_angle)
        dy = speed * math.sin(self.car_angle)
        new_pos = self.car_pos + np.array([dx, dy], dtype=np.float32)
        self.car_pos = new_pos
        self.steps += 1

        # Determine grid cell of car.
        grid_x = int(self.car_pos[0])
        grid_y = int(self.car_pos[1])
        # Check boundaries.
        if not (0 <= grid_x < self.track.width and 0 <= grid_y < self.track.height):
            reward = -50
            done = True
            return self.get_state(), reward, done

        cell = self.track.grid[grid_y][grid_x]
        # If car is on road, reward a small positive step; else, penalize.
        if cell in ['R', 'S']:
            reward = 1
        else:
            reward = -5
            done = True
            return self.get_state(), reward, done

        # Check for lap completion.
        # When the car is in the start cell and it has left the start zone.
        if cell == 'S' and self.left_start_zone:
            reward += 100  # Lap reward.
            self.laps += 1
            done = True
        else:
            done = False

        # Once the car leaves the start cell, mark that it has left.
        if cell != 'S':
            self.left_start_zone = True

        # Also end episode if too many steps.
        if self.steps >= 500:
            done = True

        return self.get_state(), reward, done

    def get_state(self):
        # Build a 3‑channel state representation as a numpy array (shape: [3, height, width]).
        # Channel 0: Road mask (1 if road or start, 0 otherwise).
        # Channel 1: Car position (a single 1 at the car’s cell).
        # Channel 2: Start line (1 if start cell, 0 otherwise).
        state = np.zeros((3, self.track.height, self.track.width), dtype=np.float32)
        for r in range(self.track.height):
            for c in range(self.track.width):
                if self.track.grid[r][c] in ['R', 'S']:
                    state[0, r, c] = 1.0
                if self.track.grid[r][c] == 'S':
                    state[2, r, c] = 1.0
        # Mark the car position.
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
    font = pygame.font.SysFont("Arial", 14)
    # Create an off‑screen surface (with extra width for hidden-layer viz).
    screen = pygame.Surface((total_width, total_height))

    # Initialize Race Track and Environment.
    track = RaceTrack(grid_width, grid_height)
    env = RaceTrackEnv(track)

    # Set up device and agent.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_shape = (3, grid_height, grid_width)
    num_actions = 3  # turn left, straight, turn right.
    agent = DQNAgent(state_shape, num_actions, device)

    num_episodes = 300
    global_score = 0
    attempt_counter = 0

    print("Training agent on the circular race track...")
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        episode_score = 0
        attempt_counter += 1
        for step in range(500):
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
            screen.fill((0, 0, 0))
            # Left panel: Draw the track.
            track.draw(screen)
            # Draw the car (blue circle).
            car_x = int(env.car_pos[0] * cell_size)
            car_y = int(env.car_pos[1] * cell_size)
            pygame.draw.circle(screen, (0, 0, 255), (car_x, car_y), cell_size // 2)
            # Info text.
            info_text = font.render(f"Attempt:{attempt_counter}  Steps:{step}  Score:{episode_score}  Global:{global_score}  ε:{agent.epsilon:.2f}  Laps:{env.laps}", True, (255, 255, 255))
            screen.blit(info_text, (5, grid_pixel_height + 5))

            # Right panel: Hidden layer visualization.
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            with torch.no_grad():
                _, hidden_activations = agent.policy_net(state_tensor)
            hidden_np = hidden_activations.cpu().numpy()[0]  # shape: (64, H, W)
            hidden_avg = np.mean(hidden_np, axis=0)
            norm_hidden = (hidden_avg - hidden_avg.min()) / (np.ptp(hidden_avg) + 1e-5)
            hidden_img = (norm_hidden * 255).astype(np.uint8)
            hidden_img_color = np.stack([hidden_img] * 3, axis=-1)
            hidden_surface = pygame.surfarray.make_surface(np.transpose(hidden_img_color, (1, 0, 2)))
            hidden_panel = pygame.transform.smoothscale(hidden_surface, (hidden_panel_width, grid_pixel_height))
            screen.blit(hidden_panel, (grid_pixel_width, 0))

            update_frame(screen)
            time.sleep(0.01)  # Faster update for smoother animation.
            if done:
                break
        print(f"Attempt {attempt_counter} finished with score {episode_score}, laps: {env.laps}")

    # ----- Demonstration Phase -----
    print("Training complete! Starting demonstration with learned policy.")
    agent.epsilon = 0.0  # Disable exploration.
    state = env.reset()
    for step in range(500):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        state = next_state

        screen.fill((0, 0, 0))
        track.draw(screen)
        car_x = int(env.car_pos[0] * cell_size)
        car_y = int(env.car_pos[1] * cell_size)
        pygame.draw.circle(screen, (0, 0, 255), (car_x, car_y), cell_size // 2)
        demo_text = font.render(f"Demo Mode - Step: {step}  Laps: {env.laps}", True, (255, 255, 255))
        screen.blit(demo_text, (5, grid_pixel_height + 5))
        # Hidden layer visualization.
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        with torch.no_grad():
            _, hidden_activations = agent.policy_net(state_tensor)
        hidden_np = hidden_activations.cpu().numpy()[0]
        hidden_avg = np.mean(hidden_np, axis=0)
        norm_hidden = (hidden_avg - hidden_avg.min()) / (np.ptp(hidden_avg) + 1e-5)
        hidden_img = (norm_hidden * 255).astype(np.uint8)
        hidden_img_color = np.stack([hidden_img] * 3, axis=-1)
        hidden_surface = pygame.surfarray.make_surface(np.transpose(hidden_img_color, (1, 0, 2)))
        hidden_panel = pygame.transform.smoothscale(hidden_surface, (hidden_panel_width, grid_pixel_height))
        screen.blit(hidden_panel, (grid_pixel_width, 0))

        update_frame(screen)
        time.sleep(0.03)
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
         <title>Advanced Real-Time Race Track RL Simulation</title>
         <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
         <style>
            body { background-color: #333; color: #fff; text-align: center; }
         </style>
      </head>
      <body>
         <h1>Advanced Real-Time Race Track RL Simulation</h1>
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
    # Run the Flask-SocketIO server on port 9999.
    socketio.run(app, host="0.0.0.0", port=9999)
