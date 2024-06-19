import pygame
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torchvision import transforms
import os
import sys
from tqdm import tqdm

# Initialize Pygame
pygame.init()




# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 800 * 1.5, 600 * 1.5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
DARK_GREEN = (0, 100, 0)

# Game settings
SPEED_MULTIPLIER = 2
DIFFICULTY_LEVEL = 1
MODEL_PATH = "dqn_model.pth"
MODE = "train"
EPOCH_IN_REAL_TIME = False


# Final dimensions for state representation
FINAL_WIDTH, FINAL_HEIGHT = 84, 84

# Game settings
PLAYER_RADIUS = 25
PLAYER_SPEED = 10
PLAYER_ATTACK_POWER = 8
PLAYER_ATTACK_COOLDOWN = 0.3

MONSTER_SIZE = 50
MONSTER_BASE_SPEED = 1.1
MONSTER_ATTACK_COOLDOWN = {"type1": 1, "type2": 3, "type3": 5}
MONSTER_PROJECTILE_SPEED = 1

MONSTER_INCREASE_PCT = 5
RESPAWN_REDUCTION_PCT = 5

PROJECTILE_SIZE = 15
PROJECTILE_SPEED = 1

# respawn data
MONSTER_RESPAWN_TIME = 8
MONSTERS_PER_RESPAWN = [1,2,3]
WAVE_INTERVAL = MONSTER_RESPAWN_TIME * 5



# Reward weights
PENALTY_HP_LOSS = -1000
SURVIVAL_TIME_WEIGHT = 1.0
MONSTERS_KILLED_WEIGHT = 1

# Hyperparameters
LEARNING_RATE = 0.003
BATCH_SIZE = 64
NUM_EPOCHS = 100
EPISODES_PER_EPOCH = 1000


# Exploration parameters
EPSILON_START = 1
EPSILON_FINAL = 0.1
EPSILON_DECAY = 0.9999

# FPS Settings
FPS = 120

# Define the CNN model using PyTorch
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=4)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=4, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=1)
        conv_out_size = self._get_conv_out(input_shape)
        self.fc1 = nn.Linear(conv_out_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.pool1(self.conv1(o))
        o = self.pool2(self.conv2(o))
        o = self.conv3(o)
        o = o.view(o.size(0), -1)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.pool1(self.conv1(x)))
        x = torch.nn.functional.leaky_relu(self.pool2(self.conv2(x)))
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        return self.fc3(x)

class Player:
    def __init__(self, speed_multiplier):
        self.initial_pos = [SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 2 * PLAYER_RADIUS]
        self.pos = self.initial_pos.copy()
        self.speed_multiplier = speed_multiplier
        self.hp = 10000
        self.attack_power = PLAYER_ATTACK_POWER
        self.attack_cooldown = PLAYER_ATTACK_COOLDOWN
        self.last_attack_time=0

    def move(self, direction):
        if direction == 0:
            self.pos[0] -= PLAYER_SPEED
        elif direction == 1:
            self.pos[0] += PLAYER_SPEED
        elif direction == 2:
            self.pos[1] -= PLAYER_SPEED
        elif direction == 3:
            self.pos[1] += PLAYER_SPEED

        self.pos[0] = max(PLAYER_RADIUS, min(SCREEN_WIDTH - PLAYER_RADIUS, self.pos[0]))
        self.pos[1] = max(PLAYER_RADIUS, min(SCREEN_HEIGHT - PLAYER_RADIUS, self.pos[1]))

    def move_manual(self, keys):
        if keys[pygame.K_LEFT]:
            self.pos[0] -= PLAYER_SPEED
            return 2  # Left
        if keys[pygame.K_RIGHT]:
            self.pos[0] += PLAYER_SPEED
            return 3  # Right
        if keys[pygame.K_UP]:
            self.pos[1] -= PLAYER_SPEED
            return 0  # Up
        if keys[pygame.K_DOWN]:
            self.pos[1] += PLAYER_SPEED
            return 1  # Down
        return 4  # No movement

    def draw(self, screen):
        pygame.draw.circle(screen, BLUE, (int(self.pos[0]), int(self.pos[1])), PLAYER_RADIUS)

    def shoot(self, projectiles, monsters,current_time):
         if current_time - self.last_attack_time >= self.attack_cooldown:
            target_monster = self.get_closest_monster(monsters)
            if target_monster:
                direction = np.array([target_monster.pos[0] - self.pos[0], target_monster.pos[1] - self.pos[1]])
                norm = np.linalg.norm(direction)
                if norm != 0:
                    direction = direction / norm
                projectiles.append([self.pos[0], self.pos[1], "blaster", direction, "player"])
            self.last_attack_time=current_time


    def get_closest_monster(self, monsters):
        if not monsters:
            return None
        distances = [((self.pos[0] - monster.pos[0]) ** 2 + (self.pos[1] - monster.pos[1]) ** 2, monster) for monster in monsters]
        _, closest_monster = min(distances, key=lambda x: x[0])
        return closest_monster

    def reset_position(self):
        self.pos = self.initial_pos.copy()

class Monster:
    def __init__(self, x_pos, y_pos, monster_type, speed, hp, attack_power, cooldown):
        self.pos = [x_pos, y_pos]
        self.type = monster_type
        self.speed = speed
        self.hp = hp
        self.attack_power = attack_power
        self.cooldown = cooldown
        self.last_attack_time = 0  # Track the last attack time

    def move(self, player_pos):
        direction = np.array(player_pos) - np.array(self.pos)
        norm = np.linalg.norm(direction)
        if norm != 0:
            direction = direction / norm
        self.pos[0] += self.speed * direction[0]
        self.pos[1] += self.speed * direction[1]

    def draw(self, screen):
        pygame.draw.rect(screen, RED, (int(self.pos[0]), int(self.pos[1]), MONSTER_SIZE, MONSTER_SIZE))

    def attack(self, projectiles, player_pos,current_time):
        pass

class Type1Monster(Monster):
    def __init__(self, x_pos, y_pos, difficulty):
        speed = MONSTER_BASE_SPEED * (1 + difficulty * 0.1)
        hp = 20 * (1 + difficulty * 0.1)
        attack_power = 5 * (1 + difficulty * 0.1)
        cooldown = MONSTER_ATTACK_COOLDOWN["type1"]
        super().__init__(x_pos, y_pos, "type1", speed, hp, attack_power, cooldown)

class Type2Monster(Monster):
    def __init__(self, x_pos, y_pos, difficulty):
        speed = MONSTER_BASE_SPEED * 1.0 * (1 + difficulty * 0.1)
        hp = 15 * (1 + difficulty * 0.1)
        attack_power = 8 * (1 + difficulty * 0.1)
        cooldown = MONSTER_ATTACK_COOLDOWN["type2"]
        super().__init__(x_pos, y_pos, "type2", speed, hp, attack_power, cooldown)

    def attack(self, projectiles, player_pos,current_time):
        if current_time - self.last_attack_time >= self.cooldown:
            direction = np.array(player_pos) - np.array([self.pos[0] + MONSTER_SIZE // 2, self.pos[1] + MONSTER_SIZE])
            norm = np.linalg.norm(direction)
            if norm != 0:
                direction = direction / norm
            projectiles.append([self.pos[0] + MONSTER_SIZE // 2, self.pos[1] + MONSTER_SIZE, "attack_projectile", direction, "monster"])
            self.last_attack_time = current_time

class Type3Monster(Monster):
    def __init__(self, x_pos, y_pos, difficulty):
        speed = MONSTER_BASE_SPEED * 0.8 * (1 + difficulty * 0.1)
        hp = 30 * (1 + difficulty * 0.1)
        attack_power = 3 * (1 + difficulty * 0.1)
        cooldown = MONSTER_ATTACK_COOLDOWN["type3"]
        super().__init__(x_pos, y_pos, "type3", speed, hp, attack_power, cooldown)

    def attack(self, projectiles, player_pos,current_time):
        if current_time - self.last_attack_time >= self.cooldown:
            direction = np.array(player_pos) - np.array([self.pos[0] + MONSTER_SIZE // 2, self.pos[1] + MONSTER_SIZE])
            norm = np.linalg.norm(direction)
            if norm != 0:
                direction = direction / norm
            angle_offsets = [0, np.pi / 6, -np.pi / 6]  # Main direction, 30 degrees to the right, 30 degrees to the left
            for angle in angle_offsets:
                rotated_direction = np.array([np.cos(angle) * direction[0] - np.sin(angle) * direction[1],
                                              np.sin(angle) * direction[0] + np.cos(angle) * direction[1]])
                projectiles.append([self.pos[0] + MONSTER_SIZE // 2, self.pos[1] + MONSTER_SIZE, "attack_projectile", rotated_direction, "monster"])
            self.last_attack_time = current_time

class Game:
    def __init__(self, difficulty=DIFFICULTY_LEVEL, speed_multiplier=SPEED_MULTIPLIER, num_epochs=NUM_EPOCHS, episodes_per_epoch=EPISODES_PER_EPOCH, model_path=MODEL_PATH, mode=MODE, monster_increase_pct=MONSTER_INCREASE_PCT, respawn_reduction_pct=RESPAWN_REDUCTION_PCT, epsilon_start=EPSILON_START, epsilon_final=EPSILON_FINAL, epsilon_decay=EPSILON_DECAY, penalty_hp_loss=PENALTY_HP_LOSS, survival_time_weight=SURVIVAL_TIME_WEIGHT, monsters_killed_weight=MONSTERS_KILLED_WEIGHT, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epoch_in_real_time=EPOCH_IN_REAL_TIME):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Simple RL Game")
        self.clock = pygame.time.Clock()
        self.mode = mode  # Set the mode
        self.epoch_in_real_time = epoch_in_real_time

        self.speed_multiplier = speed_multiplier

        self.player = Player(self.speed_multiplier)
        self.monsters = []
        self.projectiles = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_shape = (4, FINAL_WIDTH, FINAL_HEIGHT)
        self.action_space = 5  # Including "no move" action
        self.model = DQN(self.state_shape, self.action_space).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.replay_memory = deque(maxlen=2000)
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.num_epochs = num_epochs
        self.episodes_per_epoch = episodes_per_epoch
        self.episodes = self.num_epochs * self.episodes_per_epoch
        self.batch_size = batch_size
        self.difficulty = difficulty
        self.monster_increase_pct = monster_increase_pct / 100
        self.respawn_reduction_pct = respawn_reduction_pct / 100
        self.monsters_per_respawn = MONSTERS_PER_RESPAWN
        self.arrows = ["up", "down", "left", "right", "no_move"]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((FINAL_WIDTH, FINAL_HEIGHT)),
            transforms.ToTensor()
        ])
        self.last_monster_spawn_time = 0
        self.monster_respawn_time = MONSTER_RESPAWN_TIME
        self.wave_start_time = 0
        self.initial_wave_time = 0
        self.model_path = model_path
        if self.model_path:
            self.load_model(self.model_path)
        self.total_survival_time = 0  # Initialize total survival time
        self.total_monsters_killed = 0  # Initialize total monsters killed

        # Reward parameters
        self.penalty_hp_loss = penalty_hp_loss
        self.survival_time_weight = survival_time_weight
        self.monsters_killed_weight = monsters_killed_weight
        

    def create_monster(self, monster_type):
        while True:
            x_pos = random.randint(0, SCREEN_WIDTH - MONSTER_SIZE)
            y_pos = random.randint(0, SCREEN_HEIGHT - MONSTER_SIZE)
            if np.linalg.norm(np.array([x_pos, y_pos]) - np.array(self.player.pos)) > 5 * PLAYER_RADIUS:
                break
        if monster_type == 1:
            return Type1Monster(x_pos, y_pos, self.difficulty)
        elif monster_type == 2:
            return Type2Monster(x_pos, y_pos, self.difficulty)
        elif monster_type == 3:
            return Type3Monster(x_pos, y_pos, self.difficulty)

    def draw_projectiles(self):
        for projectile in self.projectiles:
            if projectile[4] == "player":
                end_pos = (projectile[0] + 15 * projectile[3][0], projectile[1] + 15 * projectile[3][1])
                pygame.draw.line(self.screen, DARK_GREEN, (int(projectile[0]), int(projectile[1])), (int(end_pos[0]), int(end_pos[1])), 5)
            else:
                pygame.draw.rect(self.screen, BLACK, (int(projectile[0]), int(projectile[1]), PROJECTILE_SIZE, PROJECTILE_SIZE))

    def update_monster_positions(self,current_time):
        new_projectiles = []
        for monster in self.monsters:
            monster.move(self.player.pos)
            monster.attack(new_projectiles, self.player.pos,current_time)
        self.monsters = [monster for monster in self.monsters if monster.pos[1] < SCREEN_HEIGHT and monster.hp > 0]
        self.projectiles += new_projectiles

    def update_projectile_positions(self):
        new_projectiles = []
        for p in self.projectiles:
            if p[2] == "blaster":
                direction = np.array(p[3])
                p[0] += direction[0] * PROJECTILE_SPEED
                p[1] += direction[1] * PROJECTILE_SPEED
                new_projectiles.append(p)
                for monster in self.monsters:
                    if self.detect_collision((p[0], p[1]), monster.pos, MONSTER_SIZE):
                        monster.hp -= self.player.attack_power
                        if monster.hp <= 0:
                            self.monsters.remove(monster)
                            self.total_monsters_killed += 1  # Increment monsters killed
                        if new_projectiles:
                            new_projectiles.pop()  # Remove projectile when it hits a monster
            elif p[4] == "monster":
                p[0] += p[3][0] * MONSTER_PROJECTILE_SPEED
                p[1] += p[3][1] * MONSTER_PROJECTILE_SPEED
                if not self.detect_collision(self.player.pos, (p[0], p[1]), PROJECTILE_SIZE):
                    new_projectiles.append(p)

        self.projectiles = new_projectiles

    def collision_check(self):
        for monster in self.monsters:
            if self.detect_collision(self.player.pos, monster.pos, MONSTER_SIZE):
                self.player.hp -= monster.attack_power
                return True, "collided with a monster"
        for projectile in self.projectiles:
            if projectile[4] == "monster" and self.detect_collision(self.player.pos, projectile[:2], PROJECTILE_SIZE):
                self.player.hp -= 10  # Example value for projectile damage
                return True, "hit by a projectile"
        return False, ""

    def detect_collision(self, player_pos, obj_pos, obj_size):
        # has to be improved to detect in a radius, now it's detecting square
        p_x, p_y = player_pos
        o_x, o_y = obj_pos
        return (o_x - PLAYER_RADIUS <= p_x < o_x + obj_size + PLAYER_RADIUS) and \
               (o_y - PLAYER_RADIUS <= p_y < o_y + obj_size + PLAYER_RADIUS)

    def get_game_image(self):
        data = pygame.surfarray.array3d(pygame.display.get_surface())
        data = np.transpose(data, (1, 0, 2))
        data = self.transform(data).squeeze().numpy()  # Ensure the output shape is [FINAL_WIDTH, FINAL_HEIGHT]
        return data

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon and self.mode == "train":
            return np.random.choice(5)  # Random action including "no move"
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.max(1)[1].item()

    def draw_arrows(self, chosen_action):
        arrow_positions = {
            0: (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100),
            1: (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50),
            2: (SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT - 75),
            3: (SCREEN_WIDTH // 2 + 50, SCREEN_HEIGHT - 75),
            4: (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 75)  # No move
        }
        for i, pos in arrow_positions.items():
            color = YELLOW if i == chosen_action else BLACK
            if i == 4:  # No move
                pygame.draw.circle(self.screen, color, pos, 10)
            else:
                pygame.draw.polygon(self.screen, color, self.get_arrow_points(pos, i))

    def get_arrow_points(self, center, direction):
        if direction == 0:  # Up
            return [(center[0], center[1] - 10), (center[0] - 10, center[1] + 10), (center[0] + 10, center[1] + 10)]
        elif direction == 1:  # Down
            return [(center[0], center[1] + 10), (center[0] - 10, center[1] - 10), (center[0] + 10, center[1] - 10)]
        elif direction == 2:  # Left
            return [(center[0] - 10, center[1]), (center[0] + 10, center[1] - 10), (center[0] + 10, center[1] + 10)]
        elif direction == 3:  # Right
            return [(center[0] + 10, center[1]), (center[0] - 10, center[1] - 10), (center[0] - 10, center[1] + 10)]

    def draw_timer(self, screen, survival_time):
        font = pygame.font.SysFont(None, 36)
        text = font.render(f"Total Surviving Time: {survival_time:.2f} seconds", True, BLACK)
        screen.blit(text, (10, 10))

    def draw_hp(self, screen, hp):
        font = pygame.font.SysFont(None, 36)
        text = font.render(f"HP: {hp}", True, BLACK)
        screen.blit(text, (10, 50))

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path))
            self.model.eval()
            print(f"Model loaded from {file_path}")
        else:
            print(f"No model found at {file_path}, starting with a new model.")

    def reset_game(self):
        self.player.reset_position()
        self.monsters = []
        self.projectiles = []
        self.last_monster_spawn_time = 0
        self.monster_respawn_time = MONSTER_RESPAWN_TIME
        self.wave_start_time = 0
        self.initial_wave_time = 0
        self.total_survival_time = 0  # Reset survival time display
        self.total_monsters_killed = 0  # Reset monsters killed
        self.player.hp = 100  # Reset player HP

    def game_loop(self, real_time=False):
        self.projectiles = []
        done = False
        start_time = time.time()
        death_reason = None

        initial_state = self.get_game_image()
        state = np.stack([initial_state] * 4, axis=0)  # Stack 4 frames to create initial state, shape [4, FINAL_WIDTH, FINAL_HEIGHT]

        current_speed_multiplier = self.speed_multiplier if not real_time else 1.0

        # First wave of monsters comes immediately
        for monster_type in self.monsters_per_respawn:
            self.monsters.append(self.create_monster(monster_type))

        while not done:
            self.screen.fill(WHITE)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            keys = pygame.key.get_pressed()
            current_time= (time.time()-start_time)*current_speed_multiplier

            if self.mode == "manual":
                action = self.player.move_manual(keys)
                if action == 4:
                    self.update_monster_positions(current_time)
                    self.update_projectile_positions()
                    collision, reason = self.collision_check()
                    if collision:
                        survival_time = (current_time) 
                        reward = self.penalty_hp_loss + (survival_time * self.survival_time_weight) + (self.total_monsters_killed * self.monsters_killed_weight)  # Custom reward
                        self.replay_memory.append((state, action, reward, state, done))
                        self.save_model(self.model_path)  # Save the model only once per round
                    if self.player.hp <= 0:
                        monsters_killed = self.total_monsters_killed
                        self.reset_game()
                        done = True
                        return state, action, reward, state, done, survival_time, death_reason, monsters_killed  # Returning survival time and death reason
                    self.player.draw(self.screen)
                    for monster in self.monsters:
                        monster.draw(self.screen)
                    self.draw_projectiles()
                    self.player.shoot(self.projectiles, self.monsters,current_time)
                    self.draw_arrows(action)
                    self.draw_timer(self.screen, current_time)
                    self.draw_hp(self.screen, self.player.hp)
                    pygame.display.update()
                    self.clock.tick(FPS)
                    continue  # Skip rest of loop if no movement key is pressed
            else:
                

                for _ in range(int(current_speed_multiplier)):
                    action = self.choose_action(state)
                    self.player.move(action)


                    if current_time - self.wave_start_time >= WAVE_INTERVAL:
                        print('new wave time=',current_time)
                        self.monster_respawn_time = MONSTER_RESPAWN_TIME  # Reset respawn time
                        self.wave_start_time = current_time
                        self.last_monster_spawn_time = current_time
                        for monster in self.monsters:
                            self.monsters.remove(monster)
                    elif current_time - self.last_monster_spawn_time >= self.monster_respawn_time:
                        print('new respawn time=',current_time)
                        print('last respawn',self.last_monster_spawn_time)
                        print('last wave',self.wave_start_time)
                        for monster_type in self.monsters_per_respawn:
                            self.monsters.append(self.create_monster(monster_type))
                        self.last_monster_spawn_time = current_time
                    else:
                        pass
                    self.update_monster_positions(current_time)
                    self.update_projectile_positions()

                    collision, reason = self.collision_check()
                    if collision:
                        survival_time = current_time
                        self.total_survival_time += survival_time
                        reward = self.penalty_hp_loss + (survival_time * self.survival_time_weight * current_time) + (self.total_monsters_killed * self.monsters_killed_weight)  # Custom reward
                        self.replay_memory.append((state, action, reward, state, done))
                        if self.player.hp <= 0:
                            monsters_killed = self.total_monsters_killed
                            self.reset_game()
                            done = True
                            return state, action, reward, state, done, survival_time, death_reason, monsters_killed  # Returning survival time and death reason

            self.player.draw(self.screen)
            for monster in self.monsters:
                monster.draw(self.screen)
            self.draw_projectiles()
            self.player.shoot(self.projectiles, self.monsters,current_time)
            self.draw_arrows(action)

            next_frame = self.get_game_image()
            next_state = np.concatenate((state[1:], np.expand_dims(next_frame, axis=0)), axis=0)  # Shape [4, FINAL_WIDTH, FINAL_HEIGHT]

            survival_time = current_time 
            reward = (survival_time * self.survival_time_weight * current_speed_multiplier) + (self.total_monsters_killed * self.monsters_killed_weight)  # Custom reward

            self.replay_memory.append((state, action, reward, next_state, done))

            state = next_state

            self.draw_timer(self.screen, survival_time*current_speed_multiplier)
            self.draw_hp(self.screen, self.player.hp)

            pygame.display.update()
            self.clock.tick(FPS)
            monsters_killed = self.total_monsters_killed
        return state, action, reward, next_state, done, survival_time, death_reason, monsters_killed  # Returning survival time and death reason

    def train_model(self, gamma=0.99):
        if len(self.replay_memory) < self.batch_size:
            print('Insufficient actions for batch')
            return
        mini_batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        states = states.view(self.batch_size, 4, FINAL_WIDTH, FINAL_HEIGHT)
        next_states = next_states.view(self.batch_size, 4, FINAL_WIDTH, FINAL_HEIGHT)

        q_values = self.model(states)
        next_q_values = self.model(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + gamma * next_q_value * (1 - dones)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        epoch_results = []

        try:
            for epoch in range(self.num_epochs):
                epoch_survival_times = []
                epoch_monsters_killed = []
                deaths_by_projectile = 0
                deaths_by_monster = 0

                for i, episode in enumerate(tqdm(range(self.episodes_per_epoch), desc=f"Epoch {epoch + 1}/{self.num_epochs}")):
                    real_time = self.epoch_in_real_time and i == 0  # Real-time speed for the first play of the epoch
                    result = self.game_loop(real_time=real_time)
                    state, action, reward, next_state, done, survival_time, death_reason, monsters_killed = result
                    self.replay_memory.append((state, action, reward, next_state, done))
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                    self.train_model()

                    epoch_survival_times.append(survival_time)
                    epoch_monsters_killed.append(monsters_killed)

                    if death_reason == "hit by a projectile":
                        deaths_by_projectile += 1
                    elif death_reason == "collided with a monster":
                        deaths_by_monster += 1

                if len(epoch_survival_times) == 0:
                    continue  # Skip empty epochs

                average_survival_time = sum(epoch_survival_times) / len(epoch_survival_times)
                average_monsters_killed = sum(epoch_monsters_killed) / len(epoch_monsters_killed)
                max_survival_time = max(epoch_survival_times)
                min_survival_time = min(epoch_survival_times)
                total_deaths = deaths_by_projectile + deaths_by_monster
                death_by_projectile_pct = (deaths_by_projectile / total_deaths) * 100 if total_deaths > 0 else 0
                death_by_monster_pct = (deaths_by_monster / total_deaths) * 100 if total_deaths > 0 else 0

                epoch_results.append({
                    "average_survival_time": average_survival_time,
                    "average_monsters_killed": average_monsters_killed,
                    "max_survival_time": max_survival_time,
                    "min_survival_time": min_survival_time,
                    "death_by_projectile_pct": death_by_projectile_pct,
                    "death_by_monster_pct": death_by_monster_pct
                })

                print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Survival Time: {average_survival_time:.2f} seconds, Average Monsters Killed: {average_monsters_killed:.2f}, Max Survival Time: {max_survival_time:.2f} seconds, Min Survival Time: {min_survival_time:.2f} seconds, % Deaths by Projectile: {death_by_projectile_pct:.2f}%, % Deaths by Monster: {death_by_monster_pct:.2f}%, epsilon={self.epsilon:.4f}")

            print("Training finished. Here are the epoch results:")
            for epoch, result in enumerate(epoch_results):
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Survival Time: {result['average_survival_time']:.2f} seconds, Average Monsters Killed: {result['average_monsters_killed']:.2f}, Max Survival Time: {result['max_survival_time']:.2f} seconds, Min Survival Time: {result['min_survival_time']:.2f} seconds, % Deaths by Projectile: {result['death_by_projectile_pct']:.2f}%, % Deaths by Monster: {result['death_by_monster_pct']:.2f}%")

        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            if self.model_path:
                self.save_model(self.model_path)
            sys.exit()

        if self.model_path:
            self.save_model(self.model_path)

        pygame.quit()

    def test(self):
        total_survival_time = 0
        total_reward = 0
        total_monsters_killed = 0
        deaths_by_projectile = 0
        deaths_by_monster = 0

        try:
            while True:
                result = self.game_loop()
                if result is None:
                    break
                state, action, reward, next_state, done, survival_time, reason, monsters_killed = result
                total_survival_time += survival_time
                total_reward += reward
                total_monsters_killed += monsters_killed
                if reason == "hit by a projectile":
                    deaths_by_projectile += 1
                elif reason == "collided with a monster":
                    deaths_by_monster += 1
                if done:
                    break

            print(f"Total survival time in testing mode: {total_survival_time:.2f} seconds, Total Reward: {total_reward:.2f}, Total Monsters Killed: {total_monsters_killed}, Deaths by Projectile: {deaths_by_projectile}, Deaths by Monster: {deaths_by_monster}")
        except KeyboardInterrupt:
            print(f"Testing interrupted. Total survival time: {total_survival_time:.2f} seconds, Total Reward: {total_reward:.2f}")
            sys.exit()

        pygame.quit()


if __name__ == "__main__":
    game = Game(difficulty=DIFFICULTY_LEVEL, speed_multiplier=SPEED_MULTIPLIER, num_epochs=NUM_EPOCHS, episodes_per_epoch=EPISODES_PER_EPOCH, model_path=MODEL_PATH, mode=MODE, monster_increase_pct=MONSTER_INCREASE_PCT, respawn_reduction_pct=RESPAWN_REDUCTION_PCT, epsilon_start=EPSILON_START, epsilon_final=EPSILON_FINAL, epsilon_decay=EPSILON_DECAY, penalty_hp_loss=PENALTY_HP_LOSS, survival_time_weight=SURVIVAL_TIME_WEIGHT, monsters_killed_weight=MONSTERS_KILLED_WEIGHT, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epoch_in_real_time=EPOCH_IN_REAL_TIME)

    if MODE == "train":
        game.train()
    elif MODE == "test":
        game.test()
    elif MODE == "manual":
        game.game_loop()
