"""
Project: Playing Sonic the Hedgehog 2 with Gym Retro and NEAT
Purpose: Demonstrate Sonic in OpenAi Gym Retro environment
Created by: John McMeen
Helpful commands
  Import Roms: python -m retro.import roms/sega_classics
"""

import retro  # pip install gym-retro

# create a gym retro environment
environment = retro.make('SonicTheHedgehog2-Genesis', 'EmeraldHillZone.Act1')
print(environment.action_space)
environment.reset()

done = False
while not done:
    # render a frame
    environment.render()

    # get a random action from the available action space
    action = environment.action_space.sample()

    # left action
    # action = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0];

    # print the action to see it in action
    print(action)

    # increment the environment by one step
    observation, reward, done, info = environment.step(action)
