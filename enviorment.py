import random
import gymnasium as gym
import numpy as np

env = gym.make('Taxi-v3', render_mode="human")
env.reset()

state_space = env.observation_space.n
action_space = env.action_space.n

Qtable_taxi = np.zeros((state_space, action_space))

def epsilon_greedy_policy(Qtable, state, epsilon):
  if random.uniform(0,1) > epsilon:
    action = np.argmax(Qtable[state][:])
  else:
    action = env.action_space.sample()
  return action

# Training parameters
n_training_episodes = 1500   # Total training episodes
learning_rate = 0.7           # Learning rate

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# EVAL_SEED
eval_seed = 52;
# Set the random seed for reproducibility
random.seed(eval_seed)
np.random.seed(eval_seed)

# Environment parameters
env_id = "Taxi-v3"           # Name of the environment
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate

# Exploration parameters
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05           # Minimum exploration probability
decay_rate = 0.05            # Exponential decay rate for exploration prob


def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable, learning_rate, gamma):
    for episode in range(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        state, info = env.reset()
        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)
            Qtable[state][action] += learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])
            if terminated or truncated:
                break
            state = new_state
    return Qtable

Qtable_taxi = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_taxi, learning_rate, gamma)

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, info = env.reset(seed=seed[episode]) if seed else env.reset()
        total_rewards_ep = 0
        for step in range(max_steps):
            action = np.argmax(Q[state][:])
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward
            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    return np.mean(episode_rewards), np.std(episode_rewards)

mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_taxi, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
