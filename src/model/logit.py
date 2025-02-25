import numpy as np

from abc import ABC
from typing import Callable, Union
from tqdm import tqdm


class Logit(ABC):

    def __init__(
        self,
        U: np.ndarray,
        ppi: np.ndarray,
        llambda: float,
        method: str = "BA",
        stop_fun: Union[str, float] = "DIE",
    ):
        """Initiate the matrix that defines individuals' payoffs and priors

        Args:
            U (np.ndarray): Payoffs Matrix of shape J x N (#{Feasible Products} x #{States of the world})
            ppi (np.ndarray): Priors Vectors of length N (#{States of the world})
            llambda (float): Info Cost
        """
        # Ndarray
        self.U = U
        self.ppi = ppi  # define differently from Armenter et al. Since, we assume that products can be in different states of the world.
        self.b_logscales = -np.max(self.U, axis=0) if method == "SQP" else 0
        self.B = np.exp((self.U / llambda) + self.b_logscales)

        # Shapes
        self.J, self.N = U.shape

        # Dist
        self.IE: Callable[[np.ndarray], np.ndarray] = lambda p: llambda * (
            np.log(self.B.T @ p) - self.b_logscales
        )
        self.DIE: Callable[[np.ndarray, np.ndarray], float] = (
            lambda p, q: (self.IE(p) - self.IE(q)).T
            @ np.diag(self.P)
            @ (self.IE(p) - self.IE(q))
        )

        # Stoppping function

        if stop_fun == "DIE":
            self.stop_fun = self.DIE

        elif isinstance(stop_fun, float):
            self.stop_fun: Callable[[np.ndarray, np.ndarray], float] = (
                lambda p, q: np.linalg.norm(p - q, ord=stop_fun)
            )

        else:
            return KeyError

    def solve_BA(self, init_guess: np.ndarray, cvg_criterion: float, max_iter: int):
        q = init_guess

        with tqdm(total=max_iter, desc="Blahut–Arimoto Solver", unit="iter") as pbar:
            for _ in range(max_iter):
                denominator = 1 / np.einsum("j,jn->n", q, self.B)
                numerator = np.einsum("jn,jn->jn", self.B, self.P)
                factor = np.einsum("jn,n->j", numerator, denominator)
                q_new = factor * q
                error = self.stop_fun(q_new, q)

                if error < cvg_criterion:
                    break
                else:
                    q = q_new

                pbar.set_postfix({"Error": error})
                pbar.update(1)

        return q

    def solve_SQP(
        self,
        init_guess: np.ndarray = None,
        cvg_criterion: float = 1e-10,
        max_iter: int = 1000,
    ):
        if init_guess is None:
            FI_pjoint = (
                np.arange(self.J)[:, None] == np.argmax(self.U, axis=0)
            ).astype(int)
            init_guess = FI_pjoint @ self.P

        b = self.B @ init_guess

        with tqdm(total=max_iter, desc="SQP Solver", unit="iter") as pbar:

            for _ in range(max_iter):
                b_old = b
                scores = np.sum(self.ppi * (self.B / self.b), axis=0) - 1

                pbar.set_postfix({"Error": error})
                pbar.update(1)

    def verify_q(self, q):
        numerator = np.einsum("ijn,ij,ijn->ijn", self.B, q, self.P)
        denominator = 1 / np.einsum("ij,ijn->in", q, self.B)
        q_hat = np.einsum("ijn,in->ij", numerator, denominator)
        return q, q_hat
