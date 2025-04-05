import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset
import torch


def generate_padding_data(dataset_size, event_size):
    n_cat = 3
    time_window_length = 100
    max_turn_time = 60
    cat_label = torch.randint(n_cat, (dataset_size, event_size, 1))
    one_hot_cat = torch.zeros(dataset_size, event_size,
                              n_cat).scatter_(2, cat_label, 1)
    event_type = torch.randint(2, (dataset_size, event_size, 1))
    one_hot_event_type = torch.zeros(dataset_size, event_size,
                                     2).scatter_(2, event_type, 1)

    t = torch.rand(dataset_size, event_size, 1).sort(1)[0]
    turn_time = (max_turn_time / time_window_length) * torch.rand(dataset_size, event_size, 3)

    return torch.cat([t, one_hot_event_type, one_hot_cat, turn_time], dim=2).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--dataset_size", type=int, default=5000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    problem = 'padding'
    
    for graph_size in opts.graph_sizes:
        datadir = os.path.join(opts.data_dir, problem)
        os.makedirs(datadir, exist_ok=True)

        if opts.filename is None:
            filename = os.path.join(datadir, "{}{}_{}_seed{}.pkl".format(
                problem,
                graph_size, opts.name, opts.seed))
        else:
            filename = check_extension(opts.filename)

        assert opts.f or not os.path.isfile(check_extension(filename)), \
            "File already exists! Try running with -f option to overwrite."

        np.random.seed(opts.seed)
        dataset = generate_padding_data(opts.dataset_size, graph_size)

        print(dataset[0])

        save_dataset(dataset, filename)