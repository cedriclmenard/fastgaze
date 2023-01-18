import torch
import torch.multiprocessing as mp


def multiprocess_run_on_dataset(n_processes: int, function, dataset: torch.utils.data.Dataset, *args):
    """ Run n_processes processes, applying the given function to every dataset sample, splitting the task
        between all processes.

    Args:
        n_processes (int): number of parallel processes to spawn
        function (function): function to run and collect return data
        dataset (torch.utils.data.Dataset): dataset to split by number of processes and run on. Should be the first argument to the function.
        args: arguments to pass after the dataset to the function.

    Returns:
        [variable type]: collected return data from function runs
    """

    # Split the dataset by the number of processes
    subsets = []
    len_subsets = int(len(dataset)/n_processes)+1
    for i in range(n_processes):
        start = i*len_subsets
        end = (i+1)*len_subsets
        if end > len(dataset):
            end = len(dataset)
        indices = list(range(start, end))
        subsets.append(torch.utils.data.Subset(dataset, indices))

    # Testing first
    # test = function(subsets[0], args)
    ctx = mp.get_context("spawn")
    # mp.set_start_method("spawn")
    with ctx.Pool(processes=n_processes) as pool:
        # func_args = (subsets[i], *args)
        multiple_results = [pool.apply_async(function, args=(subsets[i], *args)) for i in range(n_processes)]
        results = [res.get() for res in multiple_results]
    
    return results