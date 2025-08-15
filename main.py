import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from game import SnakeGameAI
from agent import Agent
from helper import plot

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, next_action, reward, next_state, done, use_qlearning=1):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        next_action = torch.tensor(next_action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            next_action = next_action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone().detach()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                if use_qlearning:
                    Q_next = torch.max(self.model(next_state[idx]))
                    Q_new = reward[idx] + self.gamma * Q_next
                else: #SARSA
                    Q_next = self.model(next_state[idx])[torch.argmax(next_action[idx]).item()]
                    Q_new = reward[idx] + self.gamma * Q_next
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(QTrainer)
    game = SnakeGameAI()

    use_qlearning = 0  # 0 = SARSA, 1 = Q-learning

    state_old = agent.get_state(game)
    move = agent.get_action(state_old)

    while agent.n_games < 200:
        reward, done, score = game.play_step(move)
        state_new = agent.get_state(game)

        if use_qlearning:
            # For Q-learning: just get the greedy next action for update, not for execution
            next_move = 0
        else:
            # For SARSA: pick next action from current policy
            next_move = agent.get_action(state_new)

        # Train step
        agent.train_short_memory(state_old, move, next_move, reward, state_new, done, use_qlearning)

        # Remember
        agent.remember(state_old, move, next_move, reward, state_new, done, use_qlearning)

        state_old = state_new

        if done:
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
            plot(plot_scores, plot_mean_scores)
        
        # For the next iteration:
        if use_qlearning:
            # Q-learning picks next execution move after update
            move = agent.get_action(state_new)
        else:
            # SARSA already picked next_move
            move = next_move

def demo():
    agent = Agent(QTrainer)
    agent.model.load()   # load your saved model
    agent.epsilon = 0    # full exploitation, no randomness

    game = SnakeGameAI()

    while True:
        state = agent.get_state(game)
        move = agent.get_action(state)  # will be greedy because epsilon = 0
        reward, done, score = game.play_step(move)

        if done:
            print("Score:", score)
            game.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Snake AI with Q-learning or SARSA")
    parser.add_argument("--algo", choices=["qlearning", "sarsa"], default="qlearning", help="Algorithm to use")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes to train")
    args = parser.parse_args()

    use_qlearning = 1 if args.algo == "qlearning" else 0
    train(use_qlearning, args.episodes)
    demo()

