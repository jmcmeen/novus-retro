from alpha.retro_neat.retro_neat_sim.simulation_runner import simulation_runner_playback

if __name__ == '__main__':
    simulation_runner_playback(network_file="18900.0.pkl",
                               neat_config_file='neat-config',
                               game='SpaceHarrierII-Genesis',
                               state='Level1',
                               scenario='lives3.json',
                               network_type='recurrent',
                               color_mode='bw',
                               render_computer_view=False,
                               controller_actions=False)

exit()
