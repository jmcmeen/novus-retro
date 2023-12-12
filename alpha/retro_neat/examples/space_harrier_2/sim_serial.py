from alpha.retro_neat.retro_neat_sim.simulation_runner import simulation_runner_serial

if __name__ == '__main__':
    simulation_runner_serial(neat_config_file='neat-config',
                             game='SpaceHarrierII-Genesis',
                             state='Level1',
                             scenario='lives3.json',
                             network_type='recurrent',
                             render_game=True,
                             color_mode='bw',
                             num_generations=1000000,
                             render_computer_view=False,
                             controller_actions=False,
                             checkpoint_file=None,
                             display_reports=True,
                             checkpoint_interval=20)

    exit()
