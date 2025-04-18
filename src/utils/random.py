import random
import torch
import numpy as np
from torch.utils.data.sampler import RandomSampler
from typing import Optional, Iterator, Sized, Any
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    """
    Set the random seed for all relevant packages.

    Args:
        seed (int): The seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class SeededDataLoader(DataLoader):
    """
    A DataLoader that sets a random seed for each iteration.
    """

    def __init__(self, dataset: Dataset, seed: int = 0, **kwargs) -> None:
        """
        Initialize the SeededDataLoader.

        Args:
            dataset (Dataset): The dataset to load from.
            seed (int): The random seed to use for this DataLoader.
            **kwargs: Additional arguments to pass to the DataLoader.
        """
        super().__init__(dataset, **kwargs)
        self.seed = seed

    def __iter__(self) -> Iterator[Any]:
        """
        Create an iterator for the DataLoader that sets a random seed for each iteration.

        Returns:
            Iterator[Any]: An iterator that yields batches of data.
        """
        # Save original random states
        orig_random_state = random.getstate()
        orig_np_state = np.random.get_state()
        orig_torch_state = torch.get_rng_state()
        orig_cuda_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )

        # Set seed for this iteration
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Get iterator from parent class
        iterator = super().__iter__()

        # Yield all elements
        try:
            yield from iterator
        finally:
            # Restore original random states
            random.setstate(orig_random_state)
            np.random.set_state(orig_np_state)
            torch.set_rng_state(orig_torch_state)
            if torch.cuda.is_available() and orig_cuda_state is not None:
                torch.cuda.set_rng_state_all(orig_cuda_state)
