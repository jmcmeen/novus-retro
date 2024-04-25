"""
Project: GymRetro + NEAT
Purpose: A generic loader for GymRetro with NEAT
Input files: 
    config-retro - Simulation configuration file
    config-neat - NEAT configuration file
"""

import retro 
import numpy as np 
import cv2 
import neat 
import pickle
import os
import time
from datetime import datetime
from retro_config_parser import RetroConfigParser

config_file = os.path.join(os.path.dirname(__file__), 'config-sonic')

rcp = RetroConfigParser(config_file)

# create retro environment: game, state, scenario (defines rewards)
environment = retro.make(game=rcp.game, state=rcp.state, scenario=rcp.scenario)

# create log file
f=open(f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-neat_log.csv", "w")

# eval genomes takes several genomes, evaluates its fitness, and returns it
def eval_genomes(genomes, config):
    genome_count = 0
    # for each genome in the population
    for genome_id, genome in genomes:
        tic = time.perf_counter()
        # reset environment to initial state
        observation = environment.reset()

        # shape/resolution of image created by emulator
        inx, iny, inc = environment.observation_space.shape

        # scale down observation
        inx = int(inx / 8)
        iny = int(iny / 8)

        if rcp.network_type == "recurrent":
            network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        elif rcp.network_type == "feedforward":
            network = neat.nn.FeedForwardNetwork.create(genome, config)
        else:
            raise ValueError("network_type must be 'recurrent' or 'feedforward'")

        # set up some variables to track fitness
        fitness = 0
        current_max_fitness = 0
        counter = 0
        current_frames = 0

        # optionally create another window for the "neural network's vision"
        if rcp.show_computer_view:
            cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        finished = False

        while not finished:
            # render the game
            if rcp.show_game_view:
                environment.render()

            # increment frame counter
            current_frames += 1

            # resize and reshape the observation image
            observation = cv2.resize(observation, (inx, iny))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (inx, iny))

            # optional update "neural network's vision"
            if rcp.show_computer_view:
                cv2.imshow('main', observation)
                cv2.waitKey(1)

            # create a single array from 2d pixel data
            img_array = np.ndarray.flatten(observation)

            # create controller actions from input
            if(rcp.frames_to_skip == 0 or current_frames % rcp.frames_to_skip == 0 or current_frames == 1):
                actions = network.activate(img_array)

            # map activation output to 0 or 1
            actions = np.where(np.array(actions) <= 0.0, 0.0, 1.0).tolist()

            # take a peek at controller actions after translation
            if rcp.show_controls: 
                print(actions)

            # increment the emulator state
            observation, reward, done, info = environment.step(actions)

            # update fitness with reward from environment
            fitness += reward

            # give it X steps without improvement to improve fitness or restart
            if fitness > current_max_fitness:
                current_max_fitness = fitness
                counter = 0
            else:
                counter += 1

            if done or counter == rcp.steps_to_kill:
                finished = True
                print(genome_id, current_max_fitness)

        genome_count = genome_count + 1
        toc = time.perf_counter()
        f.write(f'{genome_count}, {toc - tic:0.4f}, {fitness}\n')

        # set the fitness for this genome
        genome.fitness = fitness

#if main
if __name__ == '__main__':
    # NEAT configuration, all defaults except a config file is provided
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config-neat')

    # NEAT output
    population = neat.Population(config) 

    if rcp.checkpoint > 0:
        #check if file exists
        if os.path.isfile(f'{rcp.game}-{rcp.state}-{rcp.scenario}-{rcp.checkpoint}'):
            population = neat.Checkpointer.restore_checkpoint(f'{rcp.game}-{rcp.state}-{rcp.scenario}-{rcp.checkpoint}')

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(generation_interval=rcp.checkpoint_interval, 
                                              filename_prefix=f'{rcp.game}-{rcp.state}-{rcp.scenario}-'))

    # the winning network, run for x generations
    winner = population.run(eval_genomes, rcp.num_generations)

    # save the winning network to a binary file to reload later
    with open(f'{rcp.game}-{rcp.state}-{rcp.scenario}-{rcp.num_generations}.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

    exit()
