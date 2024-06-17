import pygame
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torchvision import transforms

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Game settings
PLAYER_SIZE = 50
PLAYER_SPEED = 10
MONSTER_SIZE = 50
MONSTER_BASE_SPEED = 5
PROJECTILE_SIZE = 10
PROJECTILE_SPEED = 10

# Define the CNN model using PyTorch
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class Player:
    def __init__(self):
        self.pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT - 2 * PLAYER_SIZE]
        self.speed = PLAYER_SPEED
        self.hp = 100
        self.attack_power = 10
        self.attack_speed = 1.0
        self.weapon_cooldowns = {"short_ranged": 0, "long_ranged": 0, "mid_ranged": 0}

    def move(self, direction):
        if direction == 0:
            self.pos[0] -= self.speed
        elif direction == 1:
            self.pos[0] += self.speed
        elif direction == 2:
            self.pos[1] -= self.speed
        elif direction == 3:
            self.pos[1] += self.speed
        
        self.pos[0] = max(0, min(SCREEN_WIDTH - PLAYER_SIZE, self.pos[0]))
        self.pos[1] = max(0, min(SCREEN_HEIGHT - PLAYER_SIZE, self.pos[1]))

    def draw(self, screen):
        pygame.draw.rect(screen, BLACK, (self.pos[0], self.pos[1], PLAYER_SIZE, PLAYER_SIZE))

    def shoot(self, projectiles):
        if self.weapon_cooldowns["short_ranged"] <= 0:
            projectiles.append([self.pos[0] + PLAYER_SIZE // 2, self.pos[1], "short_ranged"])
            self.weapon_cooldowns["short_ranged"] = 10
        if self.weapon_cooldowns["long_ranged"] <= 0:
            projectiles.append([self.pos[0] + PLAYER_SIZE // 2, self.pos[1], "long_ranged"])
            self.weapon_cooldowns["long_ranged"] = 30
        if self.weapon_cooldowns["mid_ranged"] <= 0:
            projectiles.append([self.pos[0] + PLAYER_SIZE // 2 - 15, self.pos[1], "mid_ranged"])
            projectiles.append([self.pos[0] + PLAYER_SIZE // 2, self.pos[1], "mid_ranged"])
            projectiles.append([self.pos[0] + PLAYER_SIZE // 2 + 15, self.pos[1], "mid_ranged"])
            self.weapon_cooldowns["mid_ranged"] = 20

        for weapon in self.weapon_cooldowns:
            if self.weapon_cooldowns[weapon] > 0:
                self.weapon_cooldowns[weapon] -= 1

class Monster:
    def __init__(self, x_pos, y_pos, monster_type, speed, hp, attack_power, attack_speed):
        self.pos = [x_pos, y_pos]
        self.type = monster_type
        self.speed = speed
        self.hp = hp
        self.attack_power = attack_power
        self.attack_speed = attack_speed
        self.cooldown = 0

    def move(self):
        self.pos[1] += self.speed

    def draw(self, screen):
        pygame.draw.rect(screen, BLACK, (self.pos[0], self.pos[1], MONSTER_SIZE, MONSTER_SIZE))

    def attack(self, projectiles):
        pass

class Type1Monster(Monster):
    def __init__(self, x_pos, y_pos, difficulty):
        speed = MONSTER_BASE_SPEED * (1 + difficulty * 0.1)
        hp = 20 * (1 + difficulty * 0.1)
        attack_power = 5 * (1 + difficulty * 0.1)
        attack_speed = 1.0
        super().__init__(x_pos, y_pos, "type1", speed, hp, attack_power, attack_speed)

class Type2Monster(Monster):
    def __init__(self, x_pos, y_pos, difficulty):
        speed = MONSTER_BASE_SPEED * 1.5 * (1 + difficulty * 0.1)
        hp = 15 * (1 + difficulty * 0.1)
        attack_power = 8 * (1 + difficulty * 0.1)
        attack_speed = 1.2
        super().__init__(x_pos, y_pos, "type2", speed, hp, attack_power, attack_speed)

class Type3Monster(Monster):
    def __init__(self, x_pos, y_pos, difficulty):
        speed = MONSTER_BASE_SPEED * 0.5 * (1 + difficulty * 0.1)
        hp = 30 * (1 + difficulty * 0.1)
        attack_power = 3 * (1 + difficulty * 0.1)
        attack_speed = 0.8
        super().__init__(x_pos, y_pos, "type3", speed, hp, attack_power, attack_speed)

    def attack(self, projectiles):
        if self.cooldown <= 0:
            projectiles.append([self.pos[0] + MONSTER_SIZE // 2, self.pos[1] + MONSTER_SIZE])
            projectiles.append([self.pos[0] + MONSTER_SIZE // 2 - 15, self.pos[1] + MONSTER_SIZE])
            projectiles.append([self.pos[0] + MONSTER_SIZE // 2 + 15, self.pos[1] + MONSTER_SIZE])
            self.cooldown = 30
        else:
            self.cooldown -= 1

class Game:
    def __init__(self, difficulty=0, play_speed=1.0):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Simple RL Game")
        self.clock = pygame.time.Clock()
        self.player = Player()
        self.monsters = []
        self.projectiles = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_shape = (4, 84, 84)
        self.action_space = 4
        self.model = DQN(self.state_shape, self.action_space).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.replay_memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.episodes = 1000
        self.difficulty = difficulty
        self.play_speed = play_speed
        self.arrows = ["up", "down", "left", "right"]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])

    def create_monster(self):
        monster_class = random.choice([Type1Monster, Type2Monster, Type3Monster])
        x_pos = random.randint(0, SCREEN_WIDTH - MONSTER_SIZE)
        y_pos = 0
        return monster_class(x_pos, y_pos, self.difficulty)

    def draw_projectiles(self):
        for projectile in self.projectiles:
            pygame.draw.rect(self.screen, RED, (projectile[0], projectile[1], PROJECTILE_SIZE, PROJECTILE_SIZE))

    def update_monster_positions(self):
        new_projectiles = []
        for monster in self.monsters:
            monster.move()
            if isinstance(monster, Type3Monster):
                monster.attack(new_projectiles)
        self.monsters = [monster for monster in self.monsters if monster.pos[1] < SCREEN_HEIGHT]
        self.projectiles += new_projectiles

    def update_projectile_positions(self):
        self.projectiles = [[p[0], p[1] + PROJECTILE_SPEED] for p in self.projectiles if p[1] < SCREEN_HEIGHT]

    def collision_check(self):
        for monster in self.monsters:
            if self.detect_collision(self.player.pos, monster.pos, MONSTER_SIZE):
                return True
        for projectile in self.projectiles:
            if self.detect_collision(self.player.pos, projectile, PROJECTILE_SIZE):
                return True
        return False

    def detect_collision(self, player_pos, obj_pos, obj_size):
        p_x, p_y = player_pos
        o_x, o_y = obj_pos
        return (o_x <= p_x < o_x + obj_size or p_x <= o_x < p_x + PLAYER_SIZE) and \
               (o_y <= p_y < o_y + obj_size or p_y <= o_y < p_y + PLAYER_SIZE)

    def get_game_image(self):
        data = pygame.surfarray.array3d(pygame.display.get_surface())
        data = np.transpose(data, (1, 0, 2))
        data = self.transform(data).squeeze().numpy()  # Ensure the output shape is [84, 84]
        return data

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(4)  # Random action
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.max(1)[1].item()

 

    def draw_arrows(self, chosen_action):
        arrow_positions = {
            0: (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100),
            1: (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50),
            2: (SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT - 75),
            3: (SCREEN_WIDTH // 2 + 50, SCREEN_HEIGHT - 75)
        }
        for i, pos in arrow_positions.items():
            color = YELLOW if i == chosen_action else BLACK
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

    def game_loop(self):
        self.monsters.append(self.create_monster())
        start_time = time.time()
        
        initial_state = self.get_game_image()
        state = np.stack([initial_state] * 4, axis=0)  # Stack 4 frames to create initial state, shape [4, 84, 84]

        self.projectiles = []
        done = False

        while not done:
            self.screen.fill(WHITE)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
            
            action = self.choose_action(state)
            self.player.move(action)
            
            if random.random() < 0.1:
                self.monsters.append(self.create_monster())
            
            self.update_monster_positions()
            self.update_projectile_positions()
            
            if self.collision_check():
                done = True
            
            # Draw player, monsters, projectiles, and arrows
            self.player.draw(self.screen)
            for monster in self.monsters:
                monster.draw(self.screen)
            self.draw_projectiles()
            self.player.shoot(self.projectiles)
            self.draw_arrows(action)
            
            next_frame = self.get_game_image()
            next_state = np.concatenate((state[1:], np.expand_dims(next_frame, axis=0)), axis=0)  # Shape [4, 84, 84]
            
            reward = time.time() - start_time
            
            self.replay_memory.append((state, action, reward, next_state, done))
            
            state = next_state
            
            pygame.display.update()
            self.clock.tick(30 * self.play_speed)
        
        return state, action, reward, next_state, done  # Return the correct state shape
    
    def train_model(self, gamma=0.99, batch_size=1):
        if len(self.replay_memory) < batch_size:
            return
        mini_batch = random.sample(self.replay_memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Ensure states and next_states have the shape [batch_size, 4, 84, 84]
        states = states.view(batch_size, 4, 84, 84)
        next_states = next_states.view(batch_size, 4, 84, 84)

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
        for episode in range(self.episodes):
            state, action, reward, next_state, done = self.game_loop()
            self.replay_memory.append((state, action, reward, next_state, done))
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.train_model(batch_size=1)
            print(f"Episode {episode + 1}/{self.episodes}, Epsilon: {self.epsilon:.2f}")

        pygame.quit()

if __name__ == "__main__":
    difficulty_level = 1  # Set difficulty level here
    play_speed = 2.0  # Set play speed multiplier here
    game = Game(difficulty=difficulty_level, play_speed=play_speed)
    game.train()
