import gym 
import highway_env
env=gym.make('highway-v0')
env.reset()
for _ in range(150):
	env.render()
	env.step(env.action_space.sample())

