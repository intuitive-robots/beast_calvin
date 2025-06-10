from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

def _get_value(x):
    if torch.is_tensor(x):
        return x.item()
    return x

class ADOPT(Optimizer):
    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.9999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        decoupled: bool = False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decoupled=decoupled,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("fused", None)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("ADOPT does not support sparse gradients")
                
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])

            self._single_tensor_adopt(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                group,
                loss
            )

        return loss

    def _single_tensor_adopt(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        group: dict,
        loss: Optional[Tensor] = None,
    ):
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        eps = group["eps"]
        maximize = group["maximize"]
        decoupled = group["decoupled"]

        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step_t = state_steps[i]

            # Update step
            step_t += 1

            if weight_decay != 0:
                if decoupled:
                    param.add_(param, alpha=-lr * weight_decay)
                else:
                    grad = grad.add(param, alpha=weight_decay)

            step = _get_value(step_t)
            if step == 1:
                exp_avg_sq.addcmul_(grad, grad.conj())
                continue

            denom = torch.clamp(exp_avg_sq.sqrt(), eps)
            if step == 2:
                exp_avg.addcdiv_(grad, denom)
            else:
                exp_avg.mul_(beta1).addcdiv_(grad, denom, value=1 - beta1)

            param.add_(exp_avg, alpha=-lr)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)