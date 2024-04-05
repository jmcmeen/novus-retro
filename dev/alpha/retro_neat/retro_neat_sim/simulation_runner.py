import multiprocessing
import os
import pickle
import cv2
import neat  # pip install neat-python
import numpy as np
import retro
import gc

from .population_extended import PopulationExtended
from .parallel_extended import ParallelEvaluatorExtended


def simulation_runner_render(network_file, neat_config_file, game, state, scenario, network_type, color_mode):
    environment = retro.make(game=game, state=state, scenario=scenario, record='.')

    # reset environment to initial state
    observation = environment.reset()

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    # local_dir = os.path.dirname(__file__)
    # config_path = os.path.join(local_dir, 'config-neat')
    # NEAT configuration, all defaults except a config file is provided
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         neat_config_file)

    # Open pkl file, binary serialization of neural network
    with open(network_file, 'rb') as input_file:
        genome = pickle.load(input_file)

    # shape/resolution of image created by emulator
    inx, iny, inc = environment.observation_space.shape

    # scale down observation
    inx = int(inx / 8)
    iny = int(iny / 8)

    # create NEAT network
    if network_type == "recurrent":
        network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
    elif network_type == "feedforward":
        network = neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        pass  # TODO throw an exception

    fitness = 0
    finished = False
    while not finished:
        # resize and reshape the observation image
        observation = cv2.resize(observation, (inx, iny))

        if color_mode == "bw":
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        elif color_mode == "rgb":
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)  # creates more inputs than bw

        #observation = np.reshape(observation, (inx, iny))  # not sure this line even needs to be here

        # create a single array from 2d pixel data
        img_array = np.ndarray.flatten(observation)

        # create controller actions from input
        actions = network.activate(img_array)

        # map relu activation output to 0 or 1
        actions = np.where(np.array(actions) <= 0.0, 0.0, 1.0).tolist()

        # increment the emulator state
        observation, reward, done, info = environment.step(actions)

        # claim the reward
        fitness += reward

        if done:
            finished = True

    # closing the environment and forcing garbage collector to run to flush bk2 file
    environment.close()
    gc.collect()

    # #forces bk2 file
    # environment.reset()
    # environment.close()
    #
    # temporary_file = glob.glob("*.bk2")[0]
    # filename = str(fitness) + ".bk2"
    # os.rename(temporary_file, filename)
    #
    # # TODO this is super hacky and needs to be thought of in the next version
    # os.system("python -m retro.scripts.playback_movie " + filename)
    #
    # os.remove(filename)

    # set the fitness for this genome
    return fitness


def simulation_runner_playback(network_file, neat_config_file, game, state, scenario, network_type, color_mode,
                               render_computer_view=False, controller_actions=False):
    environment = retro.make(game=game, state=state, scenario=scenario)

    # reset environment to initial state
    observation = environment.reset()

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    # local_dir = os.path.dirname(__file__)
    # config_path = os.path.join(local_dir, 'config-neat')
    # NEAT configuration, all defaults except a config file is provided
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         neat_config_file)

    # Open pkl file, binary serialization of neural network
    with open(network_file, 'rb') as input_file:
        genome = pickle.load(input_file)

    # shape/resolution of image created by emulator
    inx, iny, inc = environment.observation_space.shape

    # scale down observation
    inx = int(inx / 8)
    iny = int(iny / 8)

    # create NEAT network
    if network_type == "recurrent":
        network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
    elif network_type == "feedforward":
        network = neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        pass  # TODO throw an exception

    # optionally create another window for the "neural network's vision"
    if render_computer_view:
        cv2.namedWindow("main", cv2.WINDOW_NORMAL)

    fitness = 0
    finished = False
    while not finished:
        environment.render()
        # resize and reshape the observation image
        observation = cv2.resize(observation, (inx, iny))

        if color_mode == "bw":
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        elif color_mode == "rgb":
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)  # creates more inputs than bw

        #observation = np.reshape(observation, (inx, iny))  # not sure this line even needs to be here

        if render_computer_view:
            cv2.imshow('main', observation)
            cv2.waitKey(1)

        # create a single array from 2d pixel data
        img_array = np.ndarray.flatten(observation)

        # create controller actions from input
        actions = network.activate(img_array)

        # map relu activation output to 0 or 1
        actions = np.where(np.array(actions) <= 0.0, 0.0, 1.0).tolist()

        # take a peek at controller actions after translation
        if controller_actions:
            print(actions)

        # increment the emulator state
        observation, reward, done, info = environment.step(actions)

        # claim the reward
        fitness += reward

        if done:
            finished = True

    print("renaming pkl file")
    os.rename(network_file, network_file + 'p')

    # closing the environment and forcing garbage collector to run to flush bk2 file
    environment.close()
    gc.collect()

    # set the fitness for this genome
    return fitness


def simulation_runner_serial(neat_config_file, game, state, scenario, network_type, render_game, color_mode,
                             num_generations, render_computer_view=False, controller_actions=False,
                             checkpoint_file=None, display_reports=True, checkpoint_interval=1):
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    # local_dir = os.path.dirname(__file__)
    # config_path = os.path.join(local_dir, 'config-neat')
    # NEAT configuration, all defaults except a config file is provided
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         neat_config_file)

    # NEAT output
    population = PopulationExtended(config)  # using custom PopulationExtended

    if checkpoint_file:
        population = neat.Checkpointer.restore_checkpoint(checkpoint_file)

    if display_reports:
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.StatisticsReporter())

    if checkpoint_interval > 0:
        population.add_reporter(neat.Checkpointer(checkpoint_interval))

    winner = population.runSerial(evaluate_genomes, num_generations, game, state, scenario, network_type, render_game,
                                  color_mode, render_computer_view, controller_actions)

    # save the winning network to a binary file to reload later
    with open(str(population.best_genome.fitness) + '.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

    return population.best_genome.fitness


def simulation_runner_parallel(neat_config_file, game, state, scenario, network_type, color_mode, num_generations,
                               controller_actions=False, checkpoint_file=None, display_reports=True,
                               checkpoint_interval=1, num_threads=None):
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    # local_dir = os.path.dirname(__file__)
    # config_path = os.path.join(local_dir, 'config-neat')
    # NEAT configuration, all defaults except a config file is provided
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         neat_config_file)

    # NEAT output
    population = PopulationExtended(config)  # using custom PopulationExtended

    if checkpoint_file:
        population = neat.Checkpointer.restore_checkpoint(checkpoint_file)

    if display_reports:
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.StatisticsReporter())

    if checkpoint_interval > 0:
        population.add_reporter(neat.Checkpointer(checkpoint_interval))

    if num_threads:
        threads = num_threads
    else:
        threads = multiprocessing.cpu_count()

    # create a parallel evaluator that will spawn workers
    pe = ParallelEvaluatorExtended(threads, evaluate_genome)

    # the winning network up to x generations
    winner = population.runParallel(pe.evaluate, num_generations, game, state, scenario, network_type, color_mode,
                                    controller_actions)

    # save the winning network to a binary file to reload later
    with open(str(population.best_genome.fitness) + '.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

    return population.best_genome.fitness


def evaluate_genomes(genomes, config, game, state, scenario, network_type, render_game, color_mode,
                     render_computer_view, controller_actions):
    # create retro environment: game, state, scenario (defines rewards)
    environment = retro.make(game=game, state=state, scenario=scenario)

    # for each genome in the population
    for genome_id, genome in genomes:
        # reset environment to initial state
        observation = environment.reset()

        # shape/resolution of image created by emulator
        inx, iny, inc = environment.observation_space.shape

        # scale down observation
        inx = int(inx / 8)
        iny = int(iny / 8)

        # create NEAT network
        if network_type == "recurrent":
            network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        elif network_type == "feedforward":
            network = neat.nn.FeedForwardNetwork.create(genome, config)
        else:
            pass  # throw an exception

        # optionally create another window for the "neural network's vision"
        if render_computer_view:
            cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        fitness = 0
        finished = False
        while not finished:
            if render_game:
                environment.render()

            # resize and reshape the observation image
            observation = cv2.resize(observation, (inx, iny))

            if color_mode == "bw":
                observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            elif color_mode == "rgb":
                observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)  # creates more inputs than bw

            #observation = np.reshape(observation, (inx, iny))  # not sure this line even needs to be here

            if render_computer_view:
                cv2.imshow('main', observation)
                cv2.waitKey(1)

            # create a single array from 2d pixel data
            img_array = np.ndarray.flatten(observation)

            # create controller actions from input
            actions = network.activate(img_array)

            # map relu activation output to 0 or 1
            actions = np.where(np.array(actions) <= 0.0, 0.0, 1.0).tolist()

            # take a peek at controller actions after translation
            if controller_actions:
                print(actions)

            # increment the emulator state
            observation, reward, done, info = environment.step(actions)

            # claim the reward
            fitness += reward

            if done:
                finished = True
                print(genome_id, fitness)

            # set the fitness for this genome
            genome.fitness = fitness

    environment.close()


def evaluate_genome(genome, config, game, state, scenario, network_type, color_mode, controller_actions):
    # create retro environment: game, state, scenario (defines rewards)
    environment = retro.make(game=game, state=state, scenario=scenario)

    # reset environment to initial state
    observation = environment.reset()

    # shape/resolution of image created by emulator
    inx, iny, inc = environment.observation_space.shape

    # scale down observation
    inx = int(inx / 8)
    iny = int(iny / 8)

    # create NEAT network
    if network_type == "recurrent":
        network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
    elif network_type == "feedforward":
        network = neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        pass  # throw an exception

    fitness = 0
    finished = False
    while not finished:
        # resize and reshape the observation image
        observation = cv2.resize(observation, (inx, iny))

        if color_mode == "bw":
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        elif color_mode == "rgb":
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)  # creates more inputs than bw

        #observation = np.reshape(observation, (inx, iny))  # not sure this line even needs to be here

        # create a single array from 2d pixel data
        img_array = np.ndarray.flatten(observation)

        # create controller actions from input
        actions = network.activate(img_array)

        # map relu activation output to 0 or 1
        actions = np.where(np.array(actions) <= 0.0, 0.0, 1.0).tolist()

        # take a peek at controller actions after translation
        if controller_actions:
            print(actions)

        # increment the emulator state
        observation, reward, done, info = environment.step(actions)

        # claim the reward
        fitness += reward

        if done:
            finished = True

    environment.close()

    # set the fitness for this genome
    return fitness
