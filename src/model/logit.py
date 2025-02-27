import numpy as np
import cvxpy as cp

from abc import ABC
from typing import Callable, Union
from tqdm import tqdm
from scipy.optimize import root_scalar


class Logit(ABC):

    def __init__(
        self,
        U: np.ndarray,
        ppi: np.ndarray,
        llambda: float,
        method: str = "BA",
        stop_fun: Union[str, float] = "DIE",
        init_guess: np.ndarray = None,
    ):
        """Initiate the matrix that defines individuals' payoffs and priors

        Args:
            U (np.ndarray): Payoffs Matrix of shape J x N (#{Feasible Products} x #{States of the world})
            ppi (np.ndarray): Prior Vector of length N (#{States of the world})
            llambda (float): Info Cost
            method (str, optional): String to choose between the Blahut–Arimoto or the SQP Solver. Defaults to "BA".
            stop_fun (Union[str, float], optional): String to choose between the DIE or a norm p (float) as a stopping function. Defaults to "DIE".
            init_guess (np.ndarray, optional): Initial Guess of Unconditionnal Posterior as a Vector of length J. Defaults to None.

        Returns:
            _type_: _description_
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

        # Objective for SQP

        self.neg_w: Callable[[np.ndarray], float] = (
            lambda p: -llambda * ppi @ np.log(self.B.T @ p)
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

        # Initial Guess

        if init_guess is None:
            FI_pjoint = (
                np.arange(self.J)[:, None] == np.argmax(self.U, axis=0)
            ).astype(int)
            self.init_guess = FI_pjoint @ self.ppi

        else:
            self.init_guess = init_guess

    def slide(self, b_old: np.ndarray, b_new: np.ndarray) -> np.ndarray:
        if (self.ppi / b_new).T @ (b_old - b_new) <= 0:
            return b_new, 1
        elif (self.ppi / b_old).T @ (b_old - b_new) >= 0:
            return b_old, 0
        else:
            f = lambda t: (self.ppi / (t * b_new + (1 - t) * b_old)) @ (b_old - b_new)
            t = root_scalar(f, bracket=[0, 1], method="brentq").root
            return t * b_new + (1 - t) * b_old, t

    def solve_BA(self, init_guess: np.ndarray, cvg_criterion: float, max_iter: int):
        q = init_guess

        with tqdm(total=max_iter, desc="Blahut–Arimoto Solver", unit="iter") as pbar:
            for _ in range(max_iter):
                denominator = 1 / np.einsum("j,jn->n", q, self.B)
                numerator = np.einsum("jn,n->jn", self.B, self.ppi)
                factor = np.einsum("jn,n->j", numerator, denominator)
                q_new = factor * q
                error = self.stop_fun(q_new, q)

                if error < cvg_criterion:
                    break
                else:
                    q = q_new

                pbar.set_postfix({"Error": error})
                pbar.update(1)

        q = np.maximum(0, q)
        q /= q.sum()

        return q

    def solve_SQP(
        self,
        zerotol: float,
        init_guess: np.ndarray = None,
        cvg_criterion: float = 1e-10,
        max_iter: int = 1000,
    ):

        b = self.B @ init_guess

        with tqdm(total=max_iter, desc="SQP Solver", unit="iter") as pbar:

            for _ in range(max_iter):
                scores = np.sum(self.ppi * (self.B / b), axis=1) - 1
                zmax = np.max(scores)
                cand = scores >= min(-zmax * (1 - zerotol) / zerotol, -zerotol)
                j = np.sum(cand)

                D = np.diag(self.ppi / b**2)
                H = self.B[cand, :] @ D @ self.B[cand, :].T

                q_old, q_trimmed = q.copy(), q.copy()[cand]

                # Optimization variable
                Dmarg = cp.Variable(j)

                # Define the quadratic matrix H and ensure PSD
                H_psd = cp.psd_wrap(H)

                # Define the linear term
                f = (
                    -2 * (self.ppi / b).T @ self.B[cand, :].T + q_trimmed @ H_psd
                ).flatten()  # Ensure shape

                # Define the optimization variable
                Dmarg = cp.Variable(j)

                # Constraints
                constraints = [
                    cp.sum(Dmarg) == 0,  # Equality constraint (sum to 0)
                    Dmarg >= (np.zeros(j) - q_trimmed),  # Lower bound
                    Dmarg <= (np.ones(j) - q_trimmed),  # Upper bound
                ]

                # Objective function
                objective = cp.Minimize(0.5 * cp.quad_form(Dmarg, H_psd) + q @ Dmarg)

                # Solve the problem
                problem = cp.Problem(objective, constraints)
                problem.solve(
                    solver=cp.OSQP
                )  # Use a robust solver  # Alternative solvers: cp.SCS, cp.ECOS

                # Extract results
                found_quad = problem.status in ["optimal", "optimal_inaccurate"]
                Dmarg_value = Dmarg.value if found_quad else None

                if Dmarg_value.shape[0] < 1:
                    q = q_old
                    exitflag = 0

                q[cand] += Dmarg_value
                Dw = self.neg_w(q_old) - self.neg_w(q)

                slide_step = True
                if Dw < 0:
                    slide_step = False

                if slide_step:
                    b_old = b.copy()
                    b_new = self.B.T @ q
                    b, t = self.slide(b_old, b_new)

                error = self.DIE(q_old, q)

                q = np.maximum(0, q)
                q /= q.sum()

                pbar.set_postfix({"Error": error})
                pbar.update(1)

                if error < cvg_criterion:
                    break

        return q

    def verify_q(self, q):
        numerator = np.einsum("jn,j,jn->jn", self.B, q, self.ppi)
        denominator = 1 / np.einsum("j,jn->n", q, self.B)
        q_hat = np.einsum("jn,n->j", numerator, denominator)
        return q, q_hat
