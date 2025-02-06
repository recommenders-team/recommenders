# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.
#
# Based on https://github.com/microsoft/UniRec/
#

import math
import torch
import numpy as np
import cvxpy as cp
from torch import Tensor
from ._min_norm_solver import MinNormSolver


class StaticWeightSolver(object):
    def __init__(self, num_tasks: int, weight: list = None):
        self.num_tasks = num_tasks
        self.weight = weight
        print(f"static weight: {self.weight}.")

    def solve(self, grads, value):
        if self.weight is None:
            return torch.zeros(self.num_tasks, device=value.device) + 1.0 / self.num_tasks
        else:
            return torch.tensor(self.weight, device=value.device)


class MGDASolver(StaticWeightSolver):

    def solve(self, grads, value):
        sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
        return torch.tensor(sol, device=value.device).float()


class ParetoMTLSolver(StaticWeightSolver):
    def __init__(self, num_tasks: int, pref_id: int, device: torch.device, init_steps: int) -> None:
        super(ParetoMTLSolver, self).__init__(num_tasks)
        self.device = device
        self.perf_vectors = self._generate_fixed_pref_vectors(num_tasks).to(device)
        print(f"There are {len(self.perf_vectors)} optional preference vectors.")
        self.pref_id = pref_id
        self._step = 0
        self._init_flag = False
        self.init_steps = init_steps

    def _generate_random_pref_vectors(self, dim: int, num_prefs: int, norm: bool = True) -> torch.Tensor:
        r"""Generate num_prefs perference vectors in n-dim space, where all the value should be in (0,1)."""
        vectors = torch.normal(
            mean=0.0,
            std=1.0,
            size=(num_prefs, dim),
        )
        vectors = torch.abs(vectors)
        if norm:
            vectors = vectors / vectors.sum(-1, keepdim=True)
        return vectors

    def _generate_pref_vectors_2_task(self, n_pref: int = 5) -> torch.Tensor:
        _delta = math.pi / (2 * (n_pref - 1))  # there are n_pref spaces
        vecs = []
        for i in range(n_pref):
            vecs.append(torch.tensor([math.cos(_delta * i), math.sin(_delta * i)]))
        vecs = torch.stack(vecs, dim=0)
        vecs = vecs.to(self.device)
        return vecs

    def _generate_fixed_pref_vectors(self, n_tasks: int) -> torch.Tensor:
        # How to define vectors?
        if n_tasks == 3:
            vecs = np.array(
                [
                    [0.8, 0.1, 0.1],
                    [0.6, 0.2, 0.2],
                    [0.4, 0.3, 0.3],
                    [0.3, 0.4, 0.3],
                    [0.3, 0.3, 0.4],
                    [0.2, 0.6, 0.2],
                    [0.2, 0.2, 0.6],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8],
                ],
                dtype=float,
            )
        elif n_tasks == 2:
            vecs = np.array(
                [[0.9, 0.1], [0.7, 0.3], [0.5, 0.5], [0.1, 0.9], [0.3, 0.7]],
                dtype=float,
            )
        else:
            raise NotImplementedError(f"Not support for {n_tasks} tasks.")
        return torch.from_numpy(vecs).type(dtype=torch.float32).to(self.device)

    def solve(self, grads: Tensor, value: Tensor) -> Tensor:
        if (not self._init_flag) and (self._step < self.init_steps):
            return self._init_step(grads, value)

        if value.dtype == torch.float64:
            value = value.type_as(grads)
        cur_pref = self.perf_vectors[self.pref_id]
        w = self.perf_vectors - cur_pref

        gx = torch.matmul(w, value / torch.norm(value))
        idx = gx > 0

        # calculate the descent direction
        if idx.sum() <= 0:
            sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
            return torch.tensor(sol, device=self.device).float()

        vec = torch.cat((grads, torch.matmul(w[idx], grads)))
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])

        sol = torch.from_numpy(sol).type_as(w).to(self.device)
        weight = sol[self.num_tasks :] @ w[idx] + sol[: self.num_tasks]

        # BUG: the weight is for gradient, which may have negative numbers. Maybe we could apply softmax here
        weight = weight / (torch.abs(weight).sum() + 1e-8)
        return weight

    def _init_step(self, grads: Tensor, value: Tensor) -> Tensor:
        if value.dtype == torch.float64:
            value = value.type_as(grads)

        cur_pref = self.perf_vectors[self.pref_id]
        w = self.perf_vectors - cur_pref

        gx = torch.matmul(w, value / torch.norm(value))
        idx = gx > 0

        self._init_flag = False
        # calculate the descent direction
        if idx.sum() <= 0:
            self._init_flag = True
            # sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
            return torch.zeros(self.num_tasks).to(self.device)
        if idx.sum() == 1:
            sol = np.ones((1,), dtype=float)
        else:
            vec = torch.matmul(w[idx], grads)
            sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])

        sol = torch.from_numpy(sol).type_as(w).to(self.device)
        weight = sol @ w[idx]
        self._step += 1
        return weight


class EPOSolver(StaticWeightSolver):
    def __init__(self, num_tasks: int, pref: np.ndarray, eps: float = 1e-4):
        super().__init__(num_tasks)
        assert len(pref) == num_tasks, "length of pref must equal to the number of tasks."
        self.pref = pref / pref.sum()
        self.eps = eps
        self.last_move = None
        self.a = cp.Parameter(num_tasks)  # Adjustments
        self.C = cp.Parameter((num_tasks, num_tasks))  # C: Gradient inner products, G^T G
        self.Ca = cp.Parameter(num_tasks)  # d_bal^TG
        self.rhs = cp.Parameter(num_tasks)  # RHS of constraints for balancing

        self.alpha = cp.Variable(num_tasks)  # Variable to optimize
        obj_bal = cp.Maximize(self.alpha @ self.Ca)  # objective for balance
        constraints_bal = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,  # Simplex
            self.C @ self.alpha >= self.rhs,
        ]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)  # LP balance

        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))  # obj for descent
        constraints_res = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,  # Restrict
            self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
            self.C @ self.alpha >= 0,
        ]
        constraints_rel = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,  # Relaxed
            self.C @ self.alpha >= 0,
        ]
        self.prob_dom = cp.Problem(obj_dom, constraints_res)  # LP dominance
        self.prob_rel = cp.Problem(obj_dom, constraints_rel)  # LP dominance

        self.gamma = 0  # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0  # Stores the latest non-uniformity

    def solve(
        self,
        grads: Tensor,
        value: Tensor,
        r: np.ndarray = None,
        C: bool = False,
        relax: bool = False,
    ):
        try:
            l = value.cpu().numpy()
            G = (grads @ grads.T).cpu().numpy()
            r = self.pref if r is None else r
            assert len(l) == len(G) == len(r) == self.num_tasks, "length != num_tasks"
            rl, self.mu_rl, self.a.value = adjustments(l, r)
            self.C.value = G if C else G @ G.T
            self.Ca.value = self.C.value @ self.a.value

            if self.mu_rl > self.eps:
                J = self.Ca.value > 0
                if len(np.where(J)[0]) > 0:
                    J_star_idx = np.where(rl == np.max(rl))[0]
                    self.rhs.value = self.Ca.value.copy()
                    self.rhs.value[J] = -np.inf  # Not efficient; but works.
                    self.rhs.value[J_star_idx] = 0
                else:
                    self.rhs.value = np.zeros_like(self.Ca.value)
                self.gamma = self.prob_bal.solve(solver=cp.GLPK, verbose=False)
                # self.gamma = self.prob_bal.solve(verbose=False)
                self.last_move = "bal"
            else:
                if relax:
                    self.gamma = self.prob_rel.solve(solver=cp.GLPK, verbose=False)
                else:
                    self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
                # self.gamma = self.prob_dom.solve(verbose=False)
                self.last_move = "dom"

            return torch.from_numpy(self.alpha.value).to(value.device) * self.num_tasks
        except Exception as e:
            alpha = self.pref / self.pref.sum()
            return torch.from_numpy(alpha).to(value.device) * self.num_tasks


def mu(rl, normed=False):
    if len(np.where(rl < 0)[0]):
        raise ValueError(f"rl<0 \n rl={rl}")
        return None
    m = len(rl)
    l_hat = rl if normed else rl / rl.sum()
    eps = np.finfo(rl.dtype).eps
    l_hat = l_hat[l_hat > eps]
    return np.sum(l_hat * np.log(l_hat * m))


def adjustments(l, r=1):
    m = len(l)
    rl = r * l
    l_hat = rl / rl.sum()
    mu_rl = mu(l_hat, normed=True)
    a = r * (np.log(l_hat * m) - mu_rl)
    return rl, mu_rl, a


class PIController(object):
    """
    Add a weight to accuracy loss, which would control the accuracy performance of the model.
    """

    def __init__(
        self,
        expect_loss: float,
        beta_min: int = 0.2,
        beta_max: float = 1,
        K_p: float = 0.01,
        K_i: float = 0.0001,
        max_iter: int = 1e6,
        metric_name: str = None,
        expect_metric: float = None,
        metric_mode: str = "max",
        drop_rate_thres: float = 0.05,
    ):
        """
        Args:
            expect_loss(float): the expected value of the loss to be monitored.
            beta_min(float): the minimum value of beta.
            beta_max(float): the maximum value of beta.
            K_p(float): the coef of P-part.
            K_i(float): the coef of I-part.
            max_iter(int): when the number of updates arrived the value, the beta keep unchanged.
            expect_metric(float): the expected value of metric, which is a more sparce signal compared to loss.
            metric_mode(str): judgement of the current metrics. If `max`, the bigger the better. If `min`, the smaller the better.
                To better control the performance, a metric is monitored additionally, which is a more sparse signal compared to loss.
            drop_rate_thres(float): when the monitored metric drops to the thres compared with the
        """
        self.t = 0  # time counter
        self.K_p = K_p
        self.K_i = K_i

        assert beta_min <= beta_min, "beta_min should be smaller than beta_max."
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta = 0

        # NOTE: a expected value is required here. Here we would use the final BPR loss in training set of the base model.
        if metric_name is not None and metric_name.lower() != "none":
            self.metric_name = metric_name
        else:
            self.metric_name = None
        self.expect_loss = expect_loss

        assert metric_mode in {
            "max",
            "min",
        }, "expected `metric_mode` to be `min` or `max`."
        self.metric_mode = metric_mode
        self.expect_metric = expect_metric

        assert drop_rate_thres >= 0.0 and drop_rate_thres <= 1.0, "expected `drop_rate_thres` to be in [0,1], which get {}.".format(
            drop_rate_thres
        )
        self.drop_rate_thres = drop_rate_thres
        self._beta_metric_term = 0.0

        self._integral_error = 0.0
        self.__max_iter = max_iter

    @torch.no_grad()
    def control(self, loss: torch.Tensor):
        """
        Optimize the beta as time steps.

        Args:
            loss(torch.Tensor): the value of controlled loss. (KL is used in ControlVAE). We would use the loss of accuracy here.
        """
        if self.t < self.__max_iter:
            e_t = self.expect_loss - loss
            P_t = self.K_p / (1 + torch.exp(e_t))
            I_t = self._integral_error
            if (self.beta >= self.beta_min) and (self.beta <= self.beta_max):
                I_t -= self.K_i * e_t
            else:
                pass
            beta = P_t + I_t + self.beta_min
            if beta > self.beta_max:
                beta = self.beta_max
            elif beta < self.beta_min:
                beta = self.beta_min

            # save the state
            self.beta = beta
            self._integral_error = I_t
            self.t += 1
        else:  # after max_iter, fix beta
            beta = self.beta
        return min(beta + self._beta_metric_term, self.beta_max)

    def __repr__(self) -> str:
        arg_info = f"expect_loss={self.expect_loss}, beta_min={self.beta_min}, " f"beta_max={self.beta_max}, K_p={self.K_p}, K_i={self.K_i}"
        if self.metric_name is not None:
            extra_arg_info = (
                f"metric_name={self.metric_name}, expect_metric={self.expect_metric}, "
                f"metric_mode={self.metric_mode}, drop_rate_thres={self.drop_rate_thres}"
            )
            arg_info = arg_info + ", " + extra_arg_info
        info = f"{self.__class__.__name__}({arg_info})"
        return info


class PIXController(PIController):

    def __init__(
        self,
        expect_loss: float,
        beta_min: int = 0.2,
        beta_max: float = 1,
        K_p: float = 0.01,
        K_i: float = 0.0001,
        max_iter: int = 1000000,
        metric_name: str = None,
        expect_metric: float = None,
        metric_mode: str = "max",
        drop_rate_thres: float = 0.05,
        pareto_solver=None,
    ):
        super().__init__(
            expect_loss,
            beta_min,
            beta_max,
            K_p,
            K_i,
            max_iter,
            metric_name,
            expect_metric,
            metric_mode,
            drop_rate_thres,
        )
        self.pareto_solver = pareto_solver

    @torch.no_grad()
    def pareto_solve(self, grads, values):
        weights = self.pareto_solver.solve(grads, values)
        return weights


__all__ = [
    "PIController",
    "PIXController",
    "StaticWeightSolver",
    "MGDASolver",
    "ParetoMTLSolver",
    "EPOSolver",
]


if __name__ == "__main__":
    controller = PIController(expect_value=0.0)
    beta = controller.control(torch.tensor(1.0))
    beta = controller.control(torch.tensor(0.5))
    print(f"Test pass. beta={beta}.")
