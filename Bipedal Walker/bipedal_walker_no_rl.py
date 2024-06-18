import gym 
env=gym.make('BipedalWalker-v3',render_mode='human')
env.reset()
for _ in range(300):
	env.render()
	env.step(env.action_space.sample())

