from alpha.retro_neat.retro_neat_sim.simulation_runner import simulation_runner_parallel

if __name__ == '__main__':
    simulation_runner_parallel(neat_config_file='neat-config',
                               game='SpaceHarrierII-Genesis',
                               state='Level1',
                               scenario='lives3.json',
                               network_type='recurrent',
                               color_mode='rgb',
                               num_generations=100000,
                               controller_actions=False,
                               checkpoint_file=None,
                               display_reports=True,
                               checkpoint_interval=25,
                               num_threads=None)

    exit()
