import numpy as np
import cvxpy as cp
import itertools

from typing import Callable, Union
from tqdm import tqdm
from scipy.optimize import root_scalar, linprog

# TODO code the first function that returns the ppi from a N x J prior matrix or vector of independant probabilities and adapt u_mat to repeat with each states (n, j)
# TODO see and test the best


class RILogit:

    def __init__(
        self,
        u_mat: np.ndarray,
        ppi: np.ndarray,
        llambda: float,
        method: str = "BA",
        stop_fun: Union[str, float] = "DIE",
        **kwargs,
    ):
        """Initiate the matrix that defines individuals' payoffs and priors

        Args:
            u_mat (np.ndarray): Payoffs Matrix of shape N x J ( #{States of the world} x #{Feasible Products})
            ppi (np.ndarray): Prior Vector (Matrix) of length (Shape) N x J (#{States of the world})
            llambda (float): Info Cost
            method (str, optional): String to choose between the Blahut–Arimoto or the SQP Solver. Defaults to "BA".
            stop_fun (Union[str, float], optional): String to choose between the DIE or a norm p (float) as a stopping function. Defaults to "DIE".
        """
        # Shapes
        self.num_states, self.num_products = u_mat.shape

        # Args
        self.all_states = list(
            itertools.product(range(self.num_states), repeat=self.num_products)
        )
        self.u_mat = np.array(
            [
                [u_mat[n_j, j] for j, n_j in enumerate(combo)]
                for combo in self.all_states
            ]
        )
        self.ppi = np.array(
            [
                np.prod([ppi[n, j] for j, n in enumerate(combo)])
                for combo in self.all_states
            ]
        )
        self.N, self.J = self.u_mat.shape
        self.ppi = self.ppi.reshape(-1, 1) if method == "BA" else self.ppi
        self.llambda = llambda

        # Optional Args
        self.actionlabels = kwargs.get("actionlabels", np.arange(1, self.N + 1))
        self.maxit = kwargs.get("MaxIterations", 10000)
        self.maxlinit = kwargs.get("MaxLinIt", 10000)
        self.maxquadit = kwargs.get("MaxQuadIt", 200)
        self.method = method
        self.initial_p = kwargs.get("initial_p", None)
        self.stop_tol = kwargs.get("stop_tol", 1e-12)
        self.zero_tol = kwargs.get("zero_tol", 1e-9)

        # Attention Matrix
        self.b_logscales = -np.max(self.u_mat, axis=1)
        self.b_mat = (
            np.exp((self.u_mat / llambda) + self.b_logscales[:, None])
            if method == "SQP"
            else np.exp(self.u_mat / llambda)
        )

        # Dist
        self.IE: Callable[[np.ndarray], np.ndarray] = lambda p: llambda * (
            np.log(self.b_mat @ p) - self.b_logscales
            if method == "SQP"
            else np.log(self.b_mat @ p)
        )

        self.DIE: Callable[[np.ndarray, np.ndarray], float] = (
            lambda p, q: (self.IE(p) - self.IE(q))
            @ np.diag(self.ppi)
            @ (self.IE(p) - self.IE(q))
        )

        # Objective for SQP

        self.neg_w: Callable[[np.ndarray], float] = (
            lambda p: -llambda * self.ppi.T @ np.log(self.b_mat @ p)
        )

        # Stoppping function

        if stop_fun == "DIE":
            self.stop_fun = self.DIE

        elif isinstance(stop_fun, float):
            self.stop_fun: Callable[[np.ndarray, np.ndarray], float] = (
                lambda p, q: np.linalg.norm(p - q, ord=stop_fun)
            )

        else:
            self.stop_fun: Callable[[np.ndarray, np.ndarray], float] = (
                lambda p, q: np.abs(self.neg_w(p) - self.neg_w(q))
            )

        # Initial Guess

        if self.initial_p is None:
            FI_actions = np.argmax(self.u_mat, axis=1)
            FI_pjoint = np.zeros(self.u_mat.shape)
            FI_pjoint[np.arange(self.N), FI_actions] = 1
            self.initial_p = FI_pjoint.T @ self.ppi

    def slide(self, b_old: np.ndarray, b_new: np.ndarray) -> np.ndarray:
        if (self.ppi / b_new).T @ (b_old - b_new) <= 0:
            return b_new, 1
        elif (self.ppi / b_old).T @ (b_old - b_new) >= 0:
            return b_old, 0
        else:
            f = lambda t: (self.ppi / (t * b_new + (1 - t) * b_old)) @ (b_old - b_new)
            t = root_scalar(f, bracket=[0, 1], method="brentq").root
            return t * b_new + (1 - t) * b_old, t

    def solve_BA(self):

        exitflag = -1

        marg = self.initial_p

        with tqdm(total=self.maxit, desc="Blahut–Arimoto Solver", unit="iter") as pbar:
            for _ in range(self.maxit):

                # Store previous marginal
                marg_old = marg.copy()

                # Apply Blahut-Arimoto update
                temp_mat = marg.T * self.b_mat
                pcond_mat = temp_mat / temp_mat.sum(axis=1, keepdims=True)
                marg = pcond_mat.T @ self.ppi

                # Compute step size
                step_size = self.stop_fun(marg, marg_old)

                pbar.set_postfix({"Error": step_size})
                pbar.update(1)

                if step_size < self.stop_tol:
                    exitflag = 1
                    break

        # Normalize output marginal probabilities
        p_marg = np.maximum(marg, 0)
        p_marg /= p_marg.sum()

        return p_marg, exitflag

    def solve_SQP(self):

        marg = self.initial_p
        b = self.b_mat @ marg

        exitflag = -1

        with tqdm(total=self.maxit, desc="SQP Solver", unit="iter") as pbar:

            for _ in range(self.maxit):

                # Copy previous iterations
                b_old = b.copy()
                marg_old = marg.copy()

                # Remove unlikely actions
                scores = (
                    np.sum(self.ppi[:, None] * (self.b_mat / b[:, None]), axis=0) - 1
                )
                zmax = np.max(scores)
                cand = scores >= min(
                    -(1 - self.zero_tol) / self.zero_tol * zmax, -self.zero_tol
                )
                j = np.sum(cand)
                # Optimization variable
                Dmarg = cp.Variable(j)

                # Compute quadratic and linear parameters of the objective
                D = np.diag(self.ppi / (b * b))
                H = self.b_mat[:, cand].T @ D @ self.b_mat[:, cand]
                marg_trimmed = marg[cand]
                H_psd = cp.psd_wrap(H)  # Ensure PSD
                f = (
                    -2 * (self.ppi / b) @ self.b_mat[:, cand] + marg_trimmed @ H_psd
                ).flatten(
                    order="C"
                )  # Ensure shape

                # Objective
                objective = cp.Minimize(0.5 * cp.quad_form(Dmarg, H_psd) + f @ Dmarg)

                # Constraints
                constraints = [
                    cp.sum(Dmarg) == 0,  # Equality constraint (sum to 0)
                    Dmarg >= (np.zeros(j) - marg_trimmed),  # Lower bound
                    Dmarg <= (np.ones(j) - marg_trimmed),  # Upper bound
                ]

                # Minimize the objective
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.OSQP)
                found_quad = problem.status in ["optimal", "optimal_inaccurate"]
                Dmarg_value = Dmarg.value if found_quad else None

                if not found_quad:
                    exitflag = 0
                    break

                # Update marginal
                marg[cand] += Dmarg_value

                # Check if the objective is improved
                Dw = -self.neg_w(marg) + self.neg_w(marg_old)
                if Dw < 0:
                    exitflag = 0
                    break

                b, _ = self.slide(b_old, self.b_mat @ marg)

                stepsize = self.stop_fun(marg, marg_old)

                pbar.set_postfix({"Error": stepsize})
                pbar.update(1)

                if stepsize < self.stop_tol:
                    exitflag = 1
                    break

        # Scaling
        if self.maxlinit > 0:
            constr1 = np.hstack([-self.b_mat, b[:, None]])
            ff = np.hstack([np.zeros(self.J), -1])

            result = linprog(
                ff,
                A_ub=constr1,
                b_ub=np.zeros(self.N),
                A_eq=np.ones((1, self.J + 1)),
                b_eq=[1],
                bounds=[(0, 1)] * self.J + [(0, None)],
            )

            p_marg = result.x[:-1] if result.success else marg

        else:
            p_marg = marg

        # Normalize probability
        p_marg = np.clip(p_marg, 0, None)
        p_marg /= np.sum(p_marg)

        return p_marg.reshape(-1, 1), exitflag

    def get_marg(self):
        p_marg, exitflag = self.solve_BA() if self.method == "BA" else self.solve_SQP()
        if exitflag < 1:
            print("Warning algorithm did not converge")
        return p_marg

    def get_logit(self):
        p_marg = self.get_marg()
        temp_mat = p_marg.T * self.b_mat
        return temp_mat / temp_mat.sum(axis=1, keepdims=True)
