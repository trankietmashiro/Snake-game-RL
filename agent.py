import torch
import random
import numpy as np 
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self, QTrainer):
        self.n_games = 0
        self.epsilon = 1 # randomness
        self.gamma = 0.9 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr = LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, next_action, reward, next_state, done, use_qlearning):
        self.memory.append((state, action, next_action, reward, next_state, done, use_qlearning)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample  = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, next_actions, rewards, next_states, dones, use_qlearnings = zip(*mini_sample)
        self.trainer.train_step(states, actions,, next_actions, rewards, next_states, dones, use_qlearnings)

    def train_short_memory(self, state, action, next_action, reward, next_state, done, use_qlearning):
        self.trainer.train_step(state, action, next_action, reward, next_state, done, use_qlearning)

    def decay_epsilon(self, decay_rate=0.95, min_epsilon=0.05):
        return max(min_epsilon, self.epsilon * (decay_rate**self.n_games))

    def get_action(self, state):
        final_move = [0, 0, 0]
        epsilon = self.decay_epsilon()
        if random.random() < epsilon:
            move = random.randint(0, 2)  # random action
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        final_move[move] = 1      

        return final_move

