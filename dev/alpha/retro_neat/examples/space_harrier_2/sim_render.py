from alpha.retro_neat.retro_neat_sim.simulation_runner import simulation_runner_render
import gc


if __name__ == '__main__':
    fitness = simulation_runner_render(network_file="514100.0.pkl",
                                       neat_config_file='neat-config',
                                       game='SpaceHarrierII-Genesis',
                                       state='Level1',
                                       scenario='lives3.json',
                                       network_type='recurrent',
                                       color_mode='bw')
exit()
