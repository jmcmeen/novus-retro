# Novus Retro

Train neural networks to play classic retro games using [NEAT](https://neat-python.readthedocs.io/) (NeuroEvolution of Augmenting Topologies) and [Stable Retro](https://stable-retro.farama.org/). The emulator renders game frames, which are downscaled to grayscale pixel arrays and fed as input to a NEAT neural network. The network outputs controller actions, and fitness is determined by in-game rewards.

Supports both serial and parallel (multiprocessing) training, configurable game/state/scenario via INI config files, and playback of trained networks.

## Scripts

| Script | Description |
| --- | --- |
| `retro_neat.py` | Serial NEAT trainer — evaluates genomes one at a time with optional visualization |
| `retro_neat_parallel.py` | Parallel NEAT trainer — uses `multiprocessing` to evaluate genomes across all CPU cores |
| `retro_neat_playback_winner.py` | Plays back a trained network and records to BK2 (convertible to MP4 via ffmpeg) |
| `retro_config_parser.py` | Configuration parser for game/simulation/NEAT settings |

## Configuration

Game and training parameters are set via INI config files (e.g. `config-sonic`, `config-airstriker`). Sections include:

- **[retro]** — `game`, `state`, `scenario` (which game, starting level, and reward definition)
- **[simulation]** — `num_generations`, `steps_to_kill`, `frames_to_skip`
- **[neat]** — `network_type` (feedforward/recurrent), `checkpoint`, `checkpoint_interval`
- **[interactive]** — flags for showing game view, computer vision view, and controller output

NEAT hyperparameters (population size, mutation rates, etc.) are configured separately in `config-neat`.

## Setup

Install dependencies from the requirements file:

```bash
pip install -r requirements.txt
```

Import ROMs (you will need to provide your own). Assuming they are in the `roms/` folder:

```bash
python -m retro.import roms
```

## Usage

Train with serial evaluation:

```bash
python retro_neat.py
```

Train with parallel evaluation:

```bash
python retro_neat_parallel.py
```

Play back a trained winner:

```bash
python retro_neat_playback_winner.py
```
