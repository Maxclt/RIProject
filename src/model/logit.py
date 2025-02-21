import numpy as np

from abc import ABC


class Logit(ABC):

    def __init__(self, U: np.ndarray, P: np.ndarray):
        """Initiate the matrix that defines individuals' payoffs and priors

        Args:
            U (np.ndarray): Payoffs Matrix of shape I x J x N (#{Individuals} x #{Feasible Products} x #{States of the world})
            P (np.ndarray): Priors Tensors of shape G x J x N (#{Different Beliefs Groups} x #{Feasible Products} x #{States of the world})
        """
        # Ndarray
        self.U = U
        self.P = P
        self.Z = np.exp(self.U)

        # Shapes
        if U.ndim == 3:
            self.I, self.N, self.J = U.shape
        else:
            self.N, self.J = U.shape
        self.G = P.shape[1] if P.ndim == 3 else 1

    def solve_q(self, cvg_criterion: float, max_iter: int):
        q = 0.5 * np.ones((self.I, self.J))
        for _ in range(max_iter):
            denominator = 1 / np.einsum("ij,ijn->in", q, self.Z)
            numerator = np.einsum("ijn,gjn->ijn", self.Z, self.P)
            factor = np.einsum("ijn,in->ij", numerator, denominator)
            q_new = factor * q
            if np.max(np.abs(q_new - q)) < cvg_criterion:
                break
            else:
                q = q_new
        return q

    def verify_q(self, q):
        numerator = np.einsum("ijn,ij,ijn->ijn", self.Z, q, self.P)
        denominator = 1 / np.einsum("ij,ijn->in", q, self.Z)
        q_hat = np.einsum("ijn,in->ij", numerator, denominator)
        return q, q_hat
