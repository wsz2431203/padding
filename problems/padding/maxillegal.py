import argparse
import numpy as np
import os
import time
from datetime import timedelta
from scipy.spatial import distance_matrix
from subprocess import check_call, check_output, CalledProcessError
import torch
from tqdm import tqdm
import re
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import pickle

def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)


def get_costs(dataset, pi):
    # Check that tours are valid, i.e. contain 0 to n -1
    assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
    ).all(), "Invalid tour"

    n_events = dataset.size(-2)
    # Gather dataset in order of tour
    sorted_data = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))  # reorder row according to pi
    t = sorted_data[:, :, [0]]
    is_arrival = sorted_data[:, :, [1]]
    is_departure = sorted_data[:, :, [2]]
    one_hot_cat = sorted_data[:, :, 3:6]
    turn_time_by_cat = sorted_data[:, :, 6:9]
    is_arrival_departure = is_arrival.matmul(is_departure.transpose(1, 2))
    turn_time_by_connection = one_hot_cat.matmul(turn_time_by_cat.transpose(1, 2))
    dt = t.transpose(1, 2) - t
    all_connection_legality = (dt >= turn_time_by_connection).int()

    lost_connection = all_connection_legality.tril(-1) * is_arrival_departure
    illegal_connection = (1 - all_connection_legality).triu(1) * is_arrival_departure
    return lost_connection.sum(dim=(1, 2)) , None


def max_illegal(dataset):
    batch_size, graph_size, _ = dataset.size()

    t = dataset[:, :, [0]]
    is_arrival = dataset[:, :, [1]]
    is_departure = dataset[:, :, [2]]
    one_hot_cat = dataset[:, :, 3:6]
    turn_time_by_cat = dataset[:, :, 6:9]
    is_arrival_departure = is_arrival.matmul(is_departure.transpose(1, 2))
    turn_time_by_connection = one_hot_cat.matmul(turn_time_by_cat.transpose(1, 2))
    dt = t.transpose(1, 2) - t
    all_connection_legality = (dt >= turn_time_by_connection).int()
    is_illegal_arrival_departure = is_arrival_departure * (1 - all_connection_legality)
    has_illegal_connection, max_illegal_event = is_illegal_arrival_departure.flip(dims=[2]).max(dim=2)
    max_illegal_event = torch.where(has_illegal_connection == 1, graph_size - 1 - max_illegal_event, - 1)
    a, pi = (max_illegal_event + torch.arange(0, graph_size) / (2.0 * graph_size)).sort(dim=1)

    total_dist, _ = get_costs(dataset, pi)

    return total_dist, pi


def cplex_lp(dataset):
    batch_size, graph_size, _ = dataset.size()

    t = dataset[:, :, [0]]
    is_arrival = dataset[:, :, [1]]
    is_departure = dataset[:, :, [2]]
    one_hot_cat = dataset[:, :, 3:6]
    turn_time_by_cat = dataset[:, :, 6:9]
    is_arrival_departure = is_arrival.matmul(is_departure.transpose(1, 2))
    turn_time_by_connection = one_hot_cat.matmul(turn_time_by_cat.transpose(1, 2))
    dt = t.transpose(1, 2) - t
    all_connection_legality = (dt >= turn_time_by_connection).int()
    is_illegal_arrival_departure = is_arrival_departure * (1 - all_connection_legality)
    has_illegal_connection, max_illegal_event = is_illegal_arrival_departure.flip(dims=[2]).max(dim=2)
    max_illegal_event = torch.where(has_illegal_connection == 1, graph_size - 1 - max_illegal_event, - 1)
    a, pi = max_illegal_event.sort(dim=1)

    total_dist, _ = get_costs(dataset, pi)

    return total_dist, pi


def solve_all_nn(dataset_path, eval_batch_size=1024, no_cuda=False, dataset_n=None, progress_bar_mininterval=0.1):
    import torch
    from torch.utils.data import DataLoader
    from problems import PADDING
    from utils import move_to

    dataloader = DataLoader(
        PADDING.make_dataset(filename=dataset_path, num_samples=dataset_n if dataset_n is not None else 1000000),
        batch_size=eval_batch_size
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() and not no_cuda else "cpu")
    results = []
    for batch in tqdm(dataloader, mininterval=progress_bar_mininterval):
        start = time.time()
        batch = move_to(batch, device)

        lengths, pi = max_illegal(batch)
        lengths_check, _ = PADDING.get_costs(batch, pi)

        assert (torch.abs(lengths - lengths_check.data) < 1e-5).all()

        duration = time.time() - start
        results.extend(
            [(cost.item(), np.trim_zeros(pi.cpu().numpy(), 'b'), duration) for cost, pi in zip(lengths, pi)])

    return results, eval_batch_size


def solve_all_lp(dataset_path, eval_batch_size=1024, no_cuda=False, dataset_n=None, progress_bar_mininterval=0.1):
    import torch
    from torch.utils.data import DataLoader
    from problems import PADDING
    from utils import move_to

    dataloader = DataLoader(
        PADDING.make_dataset(filename=dataset_path, num_samples=dataset_n if dataset_n is not None else 1000000),
        batch_size=eval_batch_size
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() and not no_cuda else "cpu")
    results = []
    for batch in tqdm(dataloader, mininterval=progress_bar_mininterval):
        start = time.time()
        batch = move_to(batch, device)

        lengths, pi = cplex_lp(batch)
        lengths_check, _ = PADDING.get_costs(batch, pi)

        assert (torch.abs(lengths - lengths_check.data) < 1e-5).all()

        duration = time.time() - start
        results.extend(
            [(cost.item(), np.trim_zeros(pi.cpu().numpy(), 'b'), duration) for cost, pi in zip(lengths, pi)])

    return results, eval_batch_size


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("method",
                        help="Name of the method to evaluate, 'nn', 'gurobi' or '(nearest|random|farthest)_insertion'")
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA (only for Tsiligirides)')
    parser.add_argument('--disable_cache', action='store_true', help='Disable caching')
    parser.add_argument('--max_calc_batch_size', type=int, default=1000, help='Size for subbatches')
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    parser.add_argument('-n', type=int, help="Number of instances to process")
    parser.add_argument('--offset', type=int, help="Offset where to start processing")
    parser.add_argument('--results_dir', default='results', help="Name of results directory")

    opts = parser.parse_args()

    assert opts.o is None or len(opts.datasets) == 1, "Cannot specify result filename with more than one dataset"

    for dataset_path in opts.datasets:

        assert os.path.isfile(check_extension(dataset_path)), "File does not exist!"

        dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])

        if opts.o is None:
            results_dir = os.path.join(opts.results_dir, "tsp", dataset_basename)
            os.makedirs(results_dir, exist_ok=True)

            out_file = os.path.join(results_dir, "{}{}{}-{}{}".format(
                dataset_basename,
                "offs{}".format(opts.offset) if opts.offset is not None else "",
                "n{}".format(opts.n) if opts.n is not None else "",
                opts.method, ext
            ))
        else:
            out_file = opts.o

        assert opts.f or not os.path.isfile(
            out_file), "File already exists! Try running with -f option to overwrite."

        match = re.match(r'^([a-z_]+)(\d*)$', opts.method)
        assert match
        method = match[1]
        runs = 1 if match[2] == '' else int(match[2])

        if method == "nn":
            assert opts.offset is None, "Offset not supported for nearest neighbor"

            eval_batch_size = opts.max_calc_batch_size

            results, parallelism = solve_all_nn(
                dataset_path, eval_batch_size, opts.no_cuda, opts.n,
                opts.progress_bar_mininterval
            )
        elif method == "lp":
            eval_batch_size = opts.max_calc_batch_size
            results, parallelism = solve_all_lp(
                dataset_path, eval_batch_size, opts.no_cuda, opts.n,
                opts.progress_bar_mininterval
            )


        else:
            assert False, "Unknown method: {}".format(opts.method)

        costs, tours, durations = zip(*results)  # Not really costs since they should be negative
        print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
        print("Average serial duration: {} +- {}".format(
            np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
        print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
        print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

        save_dataset((results, parallelism), out_file)
