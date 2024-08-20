import pathlib
import json
import click
import hydra
from omegaconf import OmegaConf
import pandas as pd

from diffusion_policy.common.replay_buffer import ReplayBuffer

@click.command()
@click.option('-c', '--config', required=True)
def main(config):
    """
    Convert csv files containing joint position information to zarr,
    the data format used for training models in this repo.

    Args:
        config (str): data_conversion config that defines how to convert the csv files
            into the input/output formats for the diffusion policy model.    
    """

    cfg = OmegaConf.load(config)

    # Load the csv files
    data, episode_lengths = load_and_combine_data(cfg.csv_file_paths, cfg.episode_stats, cfg.decimation_factor)

    # Init replay buffer for handling output
    output_dir = pathlib.Path(cfg.output_path)
    assert output_dir.parent.is_dir()
    zarr_path = str(output_dir.joinpath(output_dir.name + '.zarr').absolute())
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode='w')

def load_and_combine_data(csv_file_paths: dict, episode_stats: dict[str, list], decimation_factor: int=1):
    """
    Load csv files from the given paths and combine them into a single dataframe.

    Args:
        csv_file_paths (dict): dictionary containing the paths to the csv files for each episode
        episode_stats (dict): dictionary containing the start and end indices for each episode
        decimation_factor (int): factor by which to decimate the data
    
    Returns:
        pandas.DataFrame: dataframe containing the combined data from all episodes
        list: list containing the lengths of each episode
    """

    chunks = []
    episode_lengths = []

    for start, end in zip(episode_stats['start'], episode_stats['end']):
        # load the required chunk of data and apply decimation in one step
        # remove first row (header) and 1st column (time)
        chunk_1 = pd.read_csv(csv_file_paths["file_1"],
                              skiprows = lambda x: x < start or ((x - start) % decimation_factor != 0),
                              nrows = (end - start) // decimation_factor + 1,
                              usecols = [1,2,3,4],
                              header = None)

        chunk_2 = pd.read_csv(csv_file_paths["file_2"],
                              skiprows = lambda x: x < start or ((x - start) % decimation_factor != 0),
                              nrows = (end - start) // decimation_factor + 1,
                              usecols = [1,2,3,4],
                              header = None)

        # confirm that the two chunks have the same length
        assert len(chunk_1) == len(chunk_2)

        # combine the two chunks
        chunk = pd.concat([chunk_1, chunk_2], axis=1, ignore_index=True)

        # confirm that the chunk has the correct length
        assert len(chunk) == len(chunk_1) == (end - start) // decimation_factor + 1

        chunks.append(chunk)
        episode_lengths.append(len(chunk))

    # combine all the chunks into a single dataframe
    data = pd.concat(chunks, axis=0, ignore_index=False)

    print("Data loaded and combined successfully!")

    return data, episode_lengths

if __name__ == '__main__':
    main()