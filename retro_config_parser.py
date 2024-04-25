import configparser
import os
import sys

class RetroConfigParser:
    def __init__(self, config_file):
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)
        
        if not self.config.has_section('retro'):
            raise configparser.NoSectionError("No section 'retro' in configuration file")
        if not self.config.has_option('retro', 'game'):
            raise configparser.NoOptionError("No option 'game' in section 'retro' in configuration file")
        if not self.config.has_option('retro', 'state'):
            raise configparser.NoOptionError("No option 'state' in section 'retro' in configuration file")
        if not self.config.has_option('retro', 'scenario'):
            raise configparser.NoOptionError("No option 'scenario' in section 'retro' in configuration file")
        
        self.game = self.config.get('retro', 'game')
        self.state = self.config.get('retro', 'state')
        self.scenario = self.config.get('retro', 'scenario')
        self.network_type = self.config.get('neat', 'network_type', fallback='recurrent')
        self.checkpoint = self.config.getint('neat', 'checkpoint', fallback=0)
        self.checkpoint_interval = self.config.getint('neat', 'checkpoint_interval', fallback=25)
        self.num_generations = self.config.getint('simulation', 'num_generations', fallback=1000)
        self.steps_to_kill = self.config.getint('simulation', 'steps_to_kill', fallback=sys.maxsize)
        self.frames_to_skip = self.config.getint('simulation', 'frames_to_skip', fallback=0)
        self.show_computer_view = self.config.getboolean('interactive', 'show_computer_view', fallback=False)
        self.show_game_view = self.config.getboolean('interactive', 'show_game_view', fallback=False)
        self.show_controls = self.config.getboolean('interactive', 'show_controls', fallback=False)

