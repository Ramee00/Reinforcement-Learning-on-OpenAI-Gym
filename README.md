# Reinforcement Learning with OpenAI Gym Environments

## Overview

This project demonstrates the application of reinforcement learning to solve the Mountain Car,Bipedal Walker and cartpole problem using OpenAI's Gym environment. The project includes two main components:


## Environment Description

The Mountain Car environment in OpenAI Gym is a classic reinforcement learning problem where a car must drive up a steep hill. The goal is to reach the flag at the top of the right hill. However, the car's engine is not strong enough to directly reach the top, so it must learn to build momentum by driving back and forth.
The BipedalWalker-v3 environment in OpenAI Gym simulates a two-legged robot learning to walk across varied terrain. The robot has a torso and two legs, and the goal is to teach it to move forward without falling. The environment provides feedback in the form of rewards based on how well the robot walks and maintains balance.
The CartPole-v1 environment is a classic problem in reinforcement learning. It involves balancing a pole on top of a moving cart. The goal is to prevent the pole from falling over by applying left or right forces to the cart.

## How to Run the Code

1. **Install Dependencies**: Make sure you have Python installed along with the necessary packages. You can install the required packages using pip:
   ```bash
   pip install gym numpy matplotlib
   ```

2. **Run Random Actions Script**: To see the Mountain Car environment with random actions, execute:
   ```bash
   python mountaincar_no_rl.py
   ```

3. **Run Q-Learning Script**: To train and test the Q-learning agent, execute:
   ```bash
   python mountaincar_QL.py
   ```

## Visualizing Results

The Q-learning script will display two plots:
- **Training Rewards**: Shows the cumulative rewards during the training episodes.
- **Testing Rewards**: Shows the rewards obtained during the testing episodes after training is completed.

These plots help visualize the agent's learning progress and its performance.

## Conclusion

This project provides a hands-on approach to understanding reinforcement learning and how Q-learning can be applied to solve the Mountain Car,Bipedal Walker and Cartpole problem. By comparing random actions with the learned policy, you can see the significant impact of reinforcement learning in achieving goal-oriented behavior in complex environments.
