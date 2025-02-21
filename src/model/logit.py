import numpy as np

from abc import ABC


class Logit(ABC):

    def __init__(self, U: np.ndarray, P: np.ndarray):
        """Initiate the matrix that defines individuals' payoffs and priors

        Args:
            U (np.ndarray): Payoffs Matrix of shape I x J x N (#{Individuals} x #{Feasible Products} x #{States of the world})
            P (np.ndarray): Priors Tensors of shape G x N x J (#{Feasible Products} x #{States of the world} x #{Different Beliefs Groups})
        """
        # Ndarray
        self.U = U
        self.P = P

        # Shapes
        if U.ndim == 3:
            self.I, self.N, self.J = U.shape
        else:
            self.N, self.J = U.shape
        self.G = P.shape[1] if P.ndim == 3 else 1

    def get_q(self):
        return np.einsum("njg,")
