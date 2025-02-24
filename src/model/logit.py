import numpy as np

from abc import ABC
from typing import Callable


class Logit(ABC):

    def __init__(
        self, U: np.ndarray, P: np.ndarray, llambda: float, method: str = "BA"
    ):
        """Initiate the matrix that defines individuals' payoffs and priors

        Args:
            U (np.ndarray): Payoffs Matrix of shape J x N (#{Individuals} x #{Feasible Products} x #{States of the world})
            P (np.ndarray): Priors Tensors of shape J x N (#{Different Beliefs Groups} x #{Feasible Products} x #{States of the world})
            llambda (float): Info Cost
        """
        # Ndarray
        self.U = U
        self.P = P
        self.B_logscales = -np.max(self.U, axis=0) if method == "SQP" else 0
        self.B = np.exp((self.U / llambda) + self.B_logscales)

        # Shapes
        self.J, self.N = U.shape

        # Dist
        self.IE: Callable[[np.ndarray], np.ndarray] = lambda p: llambda * (
            np.log(self.B.T @ p) - self.B_logscales
        )
        self.DIE: Callable[[np.ndarray, np.ndarray], np.ndarray] = (
            lambda p, q: (self.IE(p) - self.IE(q)).T
            @ self.P
            @ (self.IE(p) - self.IE(q))
        )

    def solve_BA(self, init_guess: np.ndarray, cvg_criterion: float, max_iter: int):
        q = init_guess
        for _ in range(max_iter):
            denominator = 1 / np.einsum("j,jn->n", q, self.B)
            numerator = np.einsum("jn,jn->jn", self.B, self.P)
            factor = np.einsum("jn,n->j", numerator, denominator)
            q_new = factor * q
            if np.max(np.abs(q_new - q)) < cvg_criterion:
                break
            else:
                q = q_new
        return q

    def solve_SQP(self, cvg_criterion: float, max_iter: int):
        pass

    def verify_q(self, q):
        numerator = np.einsum("ijn,ij,ijn->ijn", self.B, q, self.P)
        denominator = 1 / np.einsum("ij,ijn->in", q, self.B)
        q_hat = np.einsum("ijn,in->ij", numerator, denominator)
        return q, q_hat
