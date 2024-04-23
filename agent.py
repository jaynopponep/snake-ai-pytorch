import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
# MAX_MEMORY: Max size of the deque
# BATCH_SIZE: # of experiences sampled from memory during training
# LEARNING_RATE: Iteration steps in learning to optimize the model


class Agent:
    def __init__(self, load_model=False):
        self.n_games = 0
        self.epsilon = 0  # Controls randomness
        self.gamma = 0.9  # Discount rate, must be smaller than one (usually 0.8-0.9)
        self.memory = deque(maxlen=MAX_MEMORY)  # pop left when old memory overloads
        if load_model:
            print("Loading existing model...")
            self.model = Linear_QNet.load("model/model_epoch_95.pth")
            print("Model loaded successfully.")
        else:
            print("Initializing new model...")
            self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)
        # Above are simple initializations to the agent class

    def get_state(self, game):
        head = game.snake[0]  # List, first item is our head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        # Creates the pointers for all directions based on the head of the snake

        dir_l = game.direction == Direction.LEFT  # boolean functions for checking directions
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        # Sets one of these directions true, others false based on the directions found in the game.py

        state = [
            # Danger when straight direction. Checks if the pointers are at collision on any direction
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger for right turns
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger for left turns
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Boolean flags for any directions that are in danger (Im assuming)
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location based on head of snake
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # pop left if max memory reached
    # Remembers the previous state including reward, last action, etc to learn from that state.

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
    # Checks if memory is exceeding batch size. If so, we sample off the memory based on batch size (which is smaller
        # than memory size) Otherwise, it will sample off the entire memory.

    # Once we take our "mini_sample" whether it be the entire memory or not, we extract or unzip
        # each experience as a tuple, resulting in separate tuples for experiences
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        # We call the train_step function to train based on each tuple for the agent to train on
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    # Trains on the most recent experience

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        # Epsilon is controlled by the amount of times the game is played
        # The more games played, the lower the randomness, making the choices
        # made to be more specific to the previous trained and learned strategies
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        # Above represents exploration. If a random integer from 0 to 200 is less than epsilon
        # Then it will go with that random integer and try to explore strategies
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        # Otherwise, we make a move based on exploitation
        # The else statement is a more educated move used with torch
        # predict() is called based on the current state
        # move is set to the highest prediction value in torch.argmax()
        # Element is chosen to be set equal to 1
        return final_move


def train(epochs, load_model=False):
    agent = Agent(load_model=load_model)
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = SnakeGameAI()
    # Above is to plot all information in a graph about SnakeGameAI's progress

    for epoch in range(epochs):
        while True:
            # declares current state as the old state
            state_old = agent.get_state(game)

            # determines the next move based on the old state
            final_move = agent.get_action(state_old)

            # perform move & get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # train with short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                break
        plot(plot_scores, plot_mean_scores)
        agent.model.save(f"model_epoch_{epoch+1}.pth")


if __name__ == '__main__':
    load_existing_model = True
    train(epochs=105, load_model=load_existing_model)

