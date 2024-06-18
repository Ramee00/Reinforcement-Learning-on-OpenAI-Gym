import gym
env = gym.make("MountainCar-v0",render_mode='human')
env.reset()
for _ in range(2000):
	env.render()
	env.step(env.action_space.sample())
