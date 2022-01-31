# Novum Alpha
# Ursa Blue 2020
# Random agent performing actions. No learning.

import retro

def main():
    #env = retro.make(game='Airstriker-Genesis', record='.')

    env = retro.make(game='Airstriker-Genesis')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()

