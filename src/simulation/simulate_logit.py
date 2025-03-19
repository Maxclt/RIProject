import numpy as np

from typing import Union, List, Tuple
from src.model.logit import RILogit
from numpy.random import default_rng

# TODO specify state of the world, a value for N, or a vector that contains each state of the world for each products, then use the conditionnal logit probabilities for that state of the world, use inverse sampling method , 1000
# Multiple individuals, (corresponds to demographic groups)


def get_dist_from_sim(simulation: np.ndarray):
    counts = np.apply_along_axis(np.bincount, axis=1, arr=simulation)
    return counts / simulation.shape[1]


class SimulateRILogit(RILogit):
    def __init__(
        self,
        characteristics: np.ndarray,
        ppi: np.ndarray,
        llambda,
        method="BA",
        stop_fun="DIE",
        **kwargs,
    ):
        # Shape
        self.n_i, self.n_var = characteristics.shape

    def simulate(self, states: Union[Tuple, List[Tuple]], n_sim) -> np.ndarray:
        """Simulate n_sim draws of product choices from
        the logit distribution in one or multiple given states
        using the inverse sampling method

        Args:
            states (Union[Tuple, List[Tuple]]): list of tuples or tuple that encode the state of each products
            n_sim (int): numbers of draws

        Returns:
            np.ndarray: matrix that returns the actions chosen of shape (n_sim x n_states)
        """

        if isinstance(states, tuple):
            num_states = [self.all_states.index(states)]
        elif isinstance(states, list):
            num_states = [self.all_states.index(state) for state in states]
        else:
            return KeyError

        num_states_len = len(num_states)

        # Generate uniform samples for each state
        rg = default_rng()
        u = rg.uniform(size=(n_sim, num_states_len))  # Shape (n_sim, n_states)

        # Retrieve logit distributions for selected states
        dist = self.get_logit()[num_states]  # Shape (n_states, num_products)

        # Compute CDF for each distribution
        cdf = np.cumsum(dist, axis=1)  # Shape (n_states, num_products)

        # Perform inverse transform sampling: search across columns for each row's uniform sample
        sim = np.array(
            [np.searchsorted(cdf[i], u[:, i]) for i in range(num_states_len)]
        )

        return sim, get_dist_from_sim(sim)

    def simulate_all(self):
        """Simulate using multiple u_mat values derived from characteristics."""
        results = [] * self.n_i

        for i, characteristic in enumerate(self.characteristics):
            u_mat = self.utilities(characteristic)

            # Reinitialize with new u_mat
            super().__init__(
                u_mat, self.ppi, self.llambda, self.method, self.stop_fun, **self.kwargs
            )

            # Run the simulation for the given characteristic
            sim_result, _ = self.simulate(self.all_states, n_sim=1)
            results[i] = sim_result[0]

        return results
