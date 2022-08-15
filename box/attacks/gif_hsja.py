import logging
from typing import Union, Any, Optional, Callable, List

import torch
from typing_extensions import Literal

import math

import eagerpy as ep
import numpy as np

from box.attacks import LinearSearchBlendedUniformNoiseAttack
from box.tensorboard import TensorBoard
from ..models import Model

from ..criteria import Criterion

from ..distances import l1

from ..devutils import atleast_kd, flatten

from .base import MinimizationAttack, get_is_adversarial
from .base import get_criterion
from .base import T
from .base import raise_if_kwargs
from ..distances import l2, linf
from typing import Callable, Union, Optional, Tuple, List, Any, Dict
import cv2


class GIF_HSJA(MinimizationAttack):
    """A powerful adversarial attack that requires neither gradients
    nor probabilities [#Chen19].

    Args:
        init_attack : Attack to use to find a starting points. Defaults to
            LinearSearchBlendedUniformNoiseAttack. Only used if starting_points is None.
        steps : Number of optimization steps within each binary search step.
        initial_gradient_eval_steps: Initial number of evaluations for gradient estimation.
            Larger initial_num_evals increases time efficiency, but
            may decrease query efficiency.
        max_gradient_eval_steps : Maximum number of evaluations for gradient estimation.
        stepsize_search : How to search for stepsize; choices are 'geometric_progression',
            'grid_search'. 'geometric progression' initializes the stepsize
            by ||x_t - x||_p / sqrt(iteration), and keep decreasing by half
            until reaching the target side of the boundary. 'grid_search'
            chooses the optimal epsilon over a grid, in the scale of
            ||x_t - x||_p.
        gamma : The binary search threshold theta is gamma / d^1.5 for
                   l2 attack and gamma / d^2 for linf attack.
        tensorboard : The log directory for TensorBoard summaries. If False, TensorBoard
            summaries will be disabled (default). If None, the logdir will be
            runs/CURRENT_DATETIME_HOSTNAME.
        constraint : Norm to minimize, either "l2" or "linf"

    References:
        .. [#Chen19] Jianbo Chen, Michael I. Jordan, Martin J. Wainwright,
        "HopSkipJumpAttack: A Query-Efficient Decision-Based Attack",
        https://arxiv.org/abs/1904.02144
    """

    distance = l1

    def __init__(
        self,
        args,
        init_attack: Optional[MinimizationAttack] = None,
        steps: int = 64,
        initial_gradient_eval_steps: int = 100,
        max_gradient_eval_steps: int = 10000,
        stepsize_search: Union[
            Literal["geometric_progression"], Literal["grid_search"]
        ] = "geometric_progression",
        gamma: float = 1.0,
        tensorboard: Union[Literal[False], None, str] = False,
        constraint: Union[Literal["linf"], Literal["l2"]] = "l2",
        max_queries: int = 1000,
    ):
        if init_attack is not None and not isinstance(init_attack, MinimizationAttack):
            raise NotImplementedError
        self.args = args
        self.init_attack = init_attack
        self.steps = steps
        self.initial_num_evals = initial_gradient_eval_steps
        self.max_num_evals = max_gradient_eval_steps
        self.stepsize_search = stepsize_search
        self.gamma = gamma
        self.tensorboard = tensorboard
        self.constraint = constraint
        self._nqueries: Dict[int, int] = {}
        self.max_queries = max_queries
        assert constraint in ("l2", "linf")
        if constraint == "l2":
            self.distance = l2
        else:
            self.distance = linf

    def get_nqueries(self) -> Dict:
        return self._nqueries

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        starting_points: Optional[T] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        originals, restore_type = ep.astensor_(inputs)
        del inputs, kwargs
        self._nqueries = {i: 0 for i in range(len(originals))}
        criterion = get_criterion(criterion)
        self._criterion_is_adversarial = get_is_adversarial(criterion, model)

        if starting_points is None:
            init_attack: MinimizationAttack
            if self.init_attack is None:
                init_attack = LinearSearchBlendedUniformNoiseAttack(steps=50)
                logging.info(
                    f"Neither starting_points nor init_attack given. Falling"
                    f" back to {init_attack!r} for initialization."
                )
            else:
                init_attack = self.init_attack
            # TODO: use call and support all types of attacks (once early_stop is
            # possible in __call__)
            x_advs = init_attack.run(model, originals, criterion, early_stop=early_stop)
        else:
            x_advs = ep.astensor(starting_points)

        is_adv = self._is_adversarial(x_advs)
        if not is_adv.all():
            failed = is_adv.logical_not().float32().sum()
            if starting_points is None:
                raise ValueError(
                    f"init_attack failed for {failed} of {len(is_adv)} inputs"
                )
            else:
                raise ValueError(
                    f"{failed} of {len(is_adv)} starting_points are not adversarial"
                )
        del starting_points

        tb = TensorBoard(logdir=self.tensorboard)

        # Project the initialization to the boundary.
        x_advs = self._binary_search(self._is_adversarial, originals, x_advs)

        assert ep.all(self._is_adversarial(x_advs))

        distances = self.distance(originals, x_advs)

        for step in range(self.steps):
            delta = self.select_delta(originals, distances, step)

            # Choose number of gradient estimation steps.
            num_gradient_estimation_steps = int(
                min([self.initial_num_evals * math.sqrt(step + 1), self.max_num_evals])
            )

            gradients = self.approximate_gradients(
                self._is_adversarial, x_advs, num_gradient_estimation_steps, delta, originals
            )

            if self.constraint == "linf":
                update = ep.sign(gradients)
            else:
                update = gradients

            if self.stepsize_search == "geometric_progression":
                # find step size.
                epsilons = distances / math.sqrt(step + 1)

                while True:
                    x_advs_proposals = ep.clip(
                        x_advs + atleast_kd(epsilons, x_advs.ndim) * update, 0, 1
                    )
                    success = self._is_adversarial(x_advs_proposals)
                    epsilons = ep.where(success, epsilons, epsilons / 2.0)

                    if ep.all(success):
                        break

                # Update the sample.
                x_advs = ep.clip(
                    x_advs + atleast_kd(epsilons, update.ndim) * update, 0, 1
                )

                assert ep.all(self._is_adversarial(x_advs))

                # Binary search to return to the boundary.
                x_advs = self._binary_search(self._is_adversarial, originals, x_advs)

                assert ep.all(self._is_adversarial(x_advs))

            elif self.stepsize_search == "grid_search":
                # Grid search for stepsize.
                epsilons_grid = ep.expand_dims(
                    ep.from_numpy(
                        distances,
                        np.logspace(-4, 0, num=20, endpoint=True, dtype=np.float32),
                    ),
                    1,
                ) * ep.expand_dims(distances, 0)

                proposals_list = []

                for epsilons in epsilons_grid:
                    x_advs_proposals = (
                        x_advs + atleast_kd(epsilons, update.ndim) * update
                    )
                    x_advs_proposals = ep.clip(x_advs_proposals, 0, 1)

                    mask = self._is_adversarial(x_advs_proposals)

                    x_advs_proposals = self._binary_search(
                        self._is_adversarial, originals, x_advs_proposals
                    )

                    # only use new values where initial guess was already adversarial
                    x_advs_proposals = ep.where(
                        atleast_kd(mask, x_advs.ndim), x_advs_proposals, x_advs
                    )

                    proposals_list.append(x_advs_proposals)

                proposals = ep.stack(proposals_list, 0)
                proposals_distances = self.distance(
                    ep.expand_dims(originals, 0), proposals
                )
                minimal_idx = ep.argmin(proposals_distances, 0)

                x_advs = proposals[minimal_idx]

            distances = self.distance(originals, x_advs)

            # log stats
            tb.histogram("norms", distances, step)
            if all(v > self.max_queries for v in self._nqueries.values()):
                print("Max queries attained for all the images.")
                break

        return restore_type(x_advs)

    def approximate_gradients(
        self,
        is_adversarial: Callable[[ep.Tensor], ep.Tensor],
        x_advs: ep.Tensor,
        steps: int,
        delta: ep.Tensor,
        originals,
    ) -> ep.Tensor:
        # (steps, bs, ...)
        noise_shape = tuple([steps] + list(x_advs.shape))
        if self.constraint == "l2":
            #rv = ep.normal(x_advs, noise_shape)
            #rv = self.gnn_generator(originals, x_advs, noise_shape)
            rv = self.gif_generator(originals, x_advs, noise_shape)
        elif self.constraint == "linf":
            rv = ep.uniform(x_advs, low=-1, high=1, shape=noise_shape)
        rv /= atleast_kd(ep.norms.l2(flatten(rv, keep=1), -1), rv.ndim) + 1e-12

        scaled_rv = atleast_kd(ep.expand_dims(delta, 0), rv.ndim) * rv

        perturbed = ep.expand_dims(x_advs, 0) + scaled_rv
        perturbed = ep.clip(perturbed, 0, 1)

        rv = (perturbed - x_advs) / 2

        multipliers_list: List[ep.Tensor] = []
        for step in range(steps):
            decision = is_adversarial(perturbed[step])
            multipliers_list.append(
                ep.where(
                    decision,
                    ep.ones(x_advs, (len(x_advs,))),
                    -ep.ones(x_advs, (len(decision,))),
                )
            )
        # (steps, bs, ...)
        multipliers = ep.stack(multipliers_list, 0)

        vals = ep.where(
            ep.abs(ep.mean(multipliers, axis=0, keepdims=True)) == 1,
            multipliers,
            multipliers - ep.mean(multipliers, axis=0, keepdims=True),
        )
        grad = ep.mean(atleast_kd(vals, rv.ndim) * rv, axis=0)

        grad /= ep.norms.l2(atleast_kd(flatten(grad), grad.ndim)) + 1e-12

        return grad

    def _project(
        self, originals: ep.Tensor, perturbed: ep.Tensor, epsilons: ep.Tensor
    ) -> ep.Tensor:
        """Clips the perturbations to epsilon and returns the new perturbed

        Args:
            originals: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.
            epsilons: A batch of norm values to project to.
        Returns:
            A tensor like perturbed but with the perturbation clipped to epsilon.
        """
        epsilons = atleast_kd(epsilons, originals.ndim)
        if self.constraint == "linf":
            perturbation = perturbed - originals

            # ep.clip does not support tensors as min/max
            clipped_perturbed = ep.where(
                perturbation > epsilons, originals + epsilons, perturbed
            )
            clipped_perturbed = ep.where(
                perturbation < -epsilons, originals - epsilons, clipped_perturbed
            )
            return clipped_perturbed
        else:
            return (1.0 - epsilons) * originals + epsilons * perturbed

    def _binary_search(
        self,
        is_adversarial: Callable[[ep.Tensor], ep.Tensor],
        originals: ep.Tensor,
        perturbed: ep.Tensor,
    ) -> ep.Tensor:
        # Choose upper thresholds in binary search based on constraint.
        d = np.prod(perturbed.shape[1:])
        if self.constraint == "linf":
            highs = linf(originals, perturbed)

            # TODO: Check if the threshold is correct
            #  empirically this seems to be too low
            thresholds = highs * self.gamma / (d * d)
        else:
            highs = ep.ones(perturbed, len(perturbed))
            thresholds = self.gamma / (d * math.sqrt(d))

        lows = ep.zeros_like(highs)

        # use this variable to check when mids stays constant and the BS has converged
        old_mids = highs

        while ep.any(highs - lows > thresholds):
            mids = (lows + highs) / 2
            mids_perturbed = self._project(originals, perturbed, mids)
            is_adversarial_ = is_adversarial(mids_perturbed)

            highs = ep.where(is_adversarial_, mids, highs)
            lows = ep.where(is_adversarial_, lows, mids)

            # check of there is no more progress due to numerical imprecision
            reached_numerical_precision = (old_mids == mids).all()
            old_mids = mids

            if reached_numerical_precision:
                # TODO: warn user
                break

        res = self._project(originals, perturbed, highs)

        return res

    def select_delta(
        self, originals: ep.Tensor, distances: ep.Tensor, step: int
    ) -> ep.Tensor:
        result: ep.Tensor
        if step == 0:
            result = 0.1 * ep.ones_like(distances)
        else:
            d = np.prod(originals.shape[1:])

            if self.constraint == "linf":
                theta = self.gamma / (d * d)
                result = d * theta * distances
            else:
                theta = self.gamma / (d * np.sqrt(d))
                result = np.sqrt(d) * theta * distances

        return result

    def _is_adversarial(self, perturbed: ep.Tensor) -> ep.Tensor:
        # Count the queries made for each image
        # Count if the vector is different from the null vector
        for i, p in enumerate(perturbed):
            if not (p == 0).all():
                self._nqueries[i] += 1
        is_advs = self._criterion_is_adversarial(perturbed)
        return is_advs

    '''def gnn_generator(self, originals, x_advs, noise_shape):
        print("noise")
        print(noise_shape)
        while True:
            x=1
        N = originals.shape[0]
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        for i in range(N):
            features = torch.cat((originals[i].raw.reshape(-1, 1), x_advs[i].raw.reshape(-1, 1)), -1)
            graph = self.image_graph[0]
            self.image_graph.reprocess(features=features, k=self.args.k)

            for j in range(noise_shape[0]):
                self.image_graph.set_labels()
                model = GCN(in_feats=graph.ndata["feat"].shape[1],
                            n_hidden=self.args.hidden,
                            n_classes=1,
                            n_layers=3,
                            dropout=self.args.dropout)
                p = train(model, graph, self.args).reshape(3, n, n)
                noise[j, i, :, :, :] = p.detach()
        return ep.astensor(noise)

    def graph_filter_generator(self, originals, x_advs, noise_shape):
        # noise_shape: (steps, num_images, 3, 224, 224)
        N = originals.shape[0]
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        for i in range(N):
            #features = torch.cat((originals[i].raw.reshape(-1, 1), x_advs[i].raw.reshape(-1, 1)), -1)
            graph = self.image_graph[0]
            #self.image_graph.reprocess(features=features)
            for j in range(noise_shape[0]):
                features = torch.from_numpy(np.random.normal(0., 1., (n * n * 3, 1))).to(torch.float32).cuda()
                conv = GraphConv(n*n*3, n*n*3, norm='both', weight=False, bias=False)
                alpha = self.args.alpha
                layers = self.args.layers
                p = (1-alpha) * conv(graph, features) + alpha * features
                for k in range(layers-1):
                    p = (1-alpha) * conv(graph, p) + alpha * features
                noise[j, i, :, :, :] = p.reshape(3, n, n)
        return ep.astensor(noise)

    def ep_graph_filter_generator(self, originals, x_advs, noise_shape):
        # noise_shape: (steps, num_images, 3, 224, 224)
        N = originals.shape[0]
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        for i in range(N):
            #features = torch.cat((originals[i].raw.reshape(-1, 1), x_advs[i].raw.reshape(-1, 1)), -1)
            self.image_graph.reprocess(originals[i].raw)
            graph = self.image_graph[0]
            for j in range(noise_shape[0]):
                features = torch.from_numpy(np.random.normal(0., 1., (n * n * 3, 1))).to(torch.float32).cuda()
                conv = GraphConv(n*n*3, n*n*3, norm='none', weight=False, bias=False)
                alpha = self.args.alpha
                layers = self.args.layers
                p = (1-alpha) * conv(graph, features, edge_weight=self.image_graph.graph.edata['h']) + alpha * features
                for k in range(layers-1):
                    p = (1-alpha) * conv(graph, p, edge_weight=self.image_graph.graph.edata['h']) + alpha * features
                #p = p * self.image_graph.mask
                noise[j, i, :, :, :] = p.reshape(3, n, n)
        return ep.astensor(noise)'''

    def gif_generator(self, originals, x_advs, noise_shape):
        # noise_shape: (steps, num_images, 3, 224, 224)
        N = originals.shape[0]
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        for i in range(N):
            # features = torch.cat((originals[i].raw.reshape(-1, 1), x_advs[i].raw.reshape(-1, 1)), -1)
            for j in range(noise_shape[0]):
                # features = torch.from_numpy(np.random.normal(0., 1., (n * n * 3, 1))).to(torch.float32).cuda()
                features = np.random.normal(0., 1., (n, n, 3)).astype(np.float32)
                layers = self.args.layers
                p = cv2.ximgproc.guidedFilter(np.moveaxis(originals[i].raw.cpu().numpy(), 0, 2), features, layers,
                                              math.sqrt(layers*4) / 255)
                #p = cv2.GaussianBlur(features, (layers, layers), 0)
                noise[j, i, :, :, :] = torch.from_numpy(np.moveaxis(p, 2, 0)).cuda()
        return ep.astensor(noise)
