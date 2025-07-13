import numpy as np

def epsilon_greedy(Q_values, epsilon):
    """
    Selects an action using ε-greedy strategy.

    Parameters:
    - Q_values: list or array of estimated action values (Q_t(a))
    - epsilon: float, probability of choosing a random action

    Returns:
    - action: int, index of selected action
    """
    if np.random.rand() < epsilon:
        # Explore: choose a random action
        action = np.random.randint(len(Q_values))
    else:
        # Exploit: choose the best known action
        action = np.argmax(Q_values)
    return action


def update_q_value(Q, N, action, reward):
    """
    Updates the Q-value for a given action using the sample-average method.

    Parameters:
    - Q: list of current Q-values
    - N: list of counts of how many times each action has been taken
    - action: int, the action that was just taken
    - reward: float, the reward received for that action

    Returns:
    - Updated Q and N lists
    """
    N[action] += 1  # Increment count for this action
    Q[action] += (1 / N[action]) * (reward - Q[action])  # Update Q-value
    return Q, N


# Set up
n_actions = 10
Q = np.zeros(n_actions)     # Q-values: estimated value of each action
N = np.zeros(n_actions)     # Count of times each action was taken
epsilon = 0.1               # Exploration rate
true_values = np.random.normal(0, 1, n_actions)  # True action values (unknown to agent)

# Run the agent for 1000 steps
rewards = []
for step in range(1000):
    action = epsilon_greedy(Q, epsilon)
    reward = np.random.normal(true_values[action], 1)  # Simulate reward
    Q, N = update_q_value(Q, N, action, reward)
    rewards.append(reward)

# Print results
print("Estimated Q-values:", Q)
print("True action values:", true_values)
print("Total reward:", sum(rewards))
