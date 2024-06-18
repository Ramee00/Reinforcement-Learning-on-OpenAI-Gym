import gym
import numpy as np
import matplotlib.pyplot as plt

train_history = {'episodes': [], 'reward': []}
test_history = {'episodes': [], 'reward': []}

class QLearningAgent():
    
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.LEARNING_RATE = 0.2
        self.GAMMA = 0.90 # discount rate for future reward
        self.EPISODES = 10000
        self.DISCRETE_OS_SIZE = [20, 20] # discrete observation space size
        self.DISCRETE_OS_WIN_SIZE = (self.env.observation_space.high - self.env.observation_space.low)/self.DISCRETE_OS_SIZE

        # Exploration settings
        self.EPSILON = 1  # exploration rate
        self.EPSILON_MIN = 0.001 # min exploration rate
        self.EPSILON_DECAY = 0.9 # exploration decay rate

        # Creating a Q-Table for each state-action pair
        self.q_table = np.random.uniform(low=-2, high=0, size=(self.DISCRETE_OS_SIZE + [self.env.action_space.n]))

    ## Discretize state
    def convert_discrete(self,state):
        discrete_state = (state - self.env.observation_space.low)/self.DISCRETE_OS_WIN_SIZE
        return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table
    
    ## Action chosen based on exploration or exploitation
    def act(self,state):
        if np.random.random() > self.EPSILON:
            return np.argmax(self.q_table[state])
        else:
            return np.random.randint(0, self.env.action_space.n)

    ## Train the agent by updating Q table
    def train(self):
        num_solved_streaks = 0
        for episode in range(self.EPISODES):
            state = self.env.reset()
            discrete_state = self.convert_discrete(state)
            done = False
            episode_reward = 0

            while not done:
                action = self.act(discrete_state)
                self.env.render()
                new_state, reward, done, _ = self.env.step(action)
                new_discrete_state = self.convert_discrete(new_state)
                episode_reward += reward

                # If simulation did not end yet after last step - update Q table
                if not done:

                    # Maximum possible Q value for new state
                    max_future_q = np.max(self.q_table[new_discrete_state])

                    # Current Q value (for current state and performed action)
                    current_q = self.q_table[discrete_state + (action,)]

                    # New Q value for current state and action
                    new_q = (1 - self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (reward + self.GAMMA * max_future_q)

                    # Update Q table with new Q value
                    self.q_table[discrete_state + (action,)] = new_q

                elif new_state[0] >= self.env.goal_position:
                    num_solved_streaks+=1
                    print('episode: {}, position: {}, episode rewards: {}'.format(episode+1, new_state[0],episode_reward))
                    print('num_solved_streaks:{0}'.format(num_solved_streaks))
                    train_history['episodes'].append(episode+1)
                    train_history['reward'].append(episode_reward)
                    self.q_table[discrete_state + (action,)] = 0
                
                else:
                    num_solved_streaks=0
                    print('episode: {}, position: {}, episode rewards: {}'.format(episode, new_state[0],episode_reward))
                    train_history['episodes'].append(episode+1)
                    train_history['reward'].append(episode_reward)
                discrete_state = new_discrete_state

            if num_solved_streaks == 100:
                print('Goal achieved consecutively for 100 times')
                break
            
            # Decay exploration rate
            if self.EPSILON > self.EPSILON_MIN:
                self.EPSILON *= self.EPSILON_DECAY

    ## Test the agent with learned Q table
    def test(self):
        total_score = 0
        for episode in range(100):
            state = self.env.reset()
            discrete_state = self.convert_discrete(state)
            done = False
            episode_reward = 0
            while not done:
                self.env.render()
                action = self.act(discrete_state)
                new_state, reward, done, _ = self.env.step(action)
                discrete_state =  self.convert_discrete(new_state)
                episode_reward -= 1

                if done:
                    total_score += episode_reward
                    print('episode: {}, position: {}, episode rewards: {}'.format(episode+1, new_state[0],episode_reward))
                    test_history['episodes'].append(episode+1)
                    test_history['reward'].append(episode_reward)
                    break
        print('Average score: {}'.format(total_score/100))
        if total_score/100 > -200:
            print('Solved')
        else:
            print('Failed to solve')

if __name__ == "__main__":
    agent = QLearningAgent()
    print('Training ...')
    agent.train()
    print('Testing...')
    agent.test()

    ## Plot results
    plt.plot(train_history['episodes'],train_history['reward'])
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Mountaincar - Q Learning (Training)')
    plt.show()
    plt.plot(test_history['episodes'],test_history['reward'])
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Mountaincar - Q Learning (Testing)')
    plt.show()
