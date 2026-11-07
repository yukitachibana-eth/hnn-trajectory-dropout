"""
integration.py
Leapfrog integration of Hamiltonian dynamics: learned HNNs (single mask index, or batched Monte Carlo
rollouts over a set of dropout masks) and the ground-truth Hamiltonians used for data generation.
"""

import torch
import warnings
import torch.nn as nn
from typing import Optional, Callable, Tuple, Union
from hnnuq.masked_linear import get_model_active_mask_num, MaskedLinear, FiniteMaskMode


# =============================================================================
# Numerical Stability Utilities
# =============================================================================


def check_nan_inf(tensor: torch.Tensor) -> bool:
    """True if the tensor contains NaN or Inf (one device sync)."""
    return not torch.isfinite(tensor).all().item()


def safe_clamp(
    tensor: torch.Tensor, min_val: float = -1e6, max_val: float = 1e6
) -> torch.Tensor:
    """Clamp tensor values to prevent numerical overflow."""
    return torch.clamp(tensor, min=min_val, max=max_val)


# =============================================================================
# Core Integration Functions
# =============================================================================


def compute_hamiltonian_gradients(
    p: torch.Tensor,
    q: torch.Tensor,
    hamiltonian_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Hamiltonian gradients dH/dp and dH/dq.

    Args:
        p: Momentum tensor (must have requires_grad=True for gradient computation)
        q: Position tensor (must have requires_grad=True for gradient computation)
        hamiltonian_fn: Function computing H(p, q)
        create_graph: If True, create graph for higher-order derivatives (training)

    Returns:
        grad_p: dH/dp
        grad_q: dH/dq
    """
    with torch.enable_grad():
        if not p.requires_grad:
            p = p.requires_grad_(True)
        if not q.requires_grad:
            q = q.requires_grad_(True)

        H = hamiltonian_fn(p, q)

        grad_p = torch.autograd.grad(
            H.sum(),
            p,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad_p is None:
            grad_p = torch.zeros_like(p)

        grad_q = torch.autograd.grad(
            H.sum(),
            q,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad_q is None:
            grad_q = torch.zeros_like(q)

        grad_p = safe_clamp(grad_p, -50.0, 50.0)
        grad_q = safe_clamp(grad_q, -50.0, 50.0)

    return grad_p, grad_q


def leapfrog_step(
    p: torch.Tensor,
    q: torch.Tensor,
    hamiltonian_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dt: float,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform one leapfrog (Störmer-Verlet) integration step.

    For create_graph=False (ground truth / inference):
        - Computes gradients locally without building backprop graph
        - Returns detached tensors with requires_grad=True for next step

    For create_graph=True (training):
        - Builds full computation graph for backpropagation
    """
    # Step 1: p_half = p - (dt/2) * dH/dq(p, q)
    _, grad_q = compute_hamiltonian_gradients(
        p, q, hamiltonian_fn, create_graph=create_graph
    )
    p_half = p - 0.5 * dt * grad_q

    # Step 2: q_new = q + dt * dH/dp(p_half, q)

    grad_p, _ = compute_hamiltonian_gradients(
        p_half, q, hamiltonian_fn, create_graph=create_graph
    )
    q_new = q + dt * grad_p

    # Step 3: p_new = p_half - (dt/2) * dH/dq(p_half, q_new)
    _, grad_q_new = compute_hamiltonian_gradients(
        p_half, q_new, hamiltonian_fn, create_graph=create_graph
    )
    p_new = p_half - 0.5 * dt * grad_q_new

    # Inference: detach and re-enable grad for the next step (training keeps the graph).
    if not create_graph:
        p_new = p_new.detach().requires_grad_(True)
        q_new = q_new.detach().requires_grad_(True)

    return p_new, q_new


def integrate_trajectory_generic(
    p_0: torch.Tensor,
    q_0: torch.Tensor,
    hamiltonian_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    T: int,
    dt: float,
    use_leapfrog: bool = True,
    requires_grad: bool = True,
) -> torch.Tensor:
    """Integrate Hamiltonian dynamics with the leapfrog scheme and record T states.

    Args:
        p_0, q_0: initial momentum and position, (batch_size, dim)
        hamiltonian_fn: function computing H(p, q)
        T: number of recorded states (the initial state is state 0)
        dt: time step size
        use_leapfrog: must be True (the only implemented integrator)
        requires_grad: True keeps the computation graph (training); False detaches every step

    A step that becomes non-finite stops the integration: training repeats the last finite
    state for the remaining steps, inference leaves them at zero.

    Returns:
        trajectory: (T, batch_size, 2*dim) with [p, q] concatenated
    """
    if not use_leapfrog:
        raise NotImplementedError("only the leapfrog integrator is implemented")

    device = p_0.device
    batch_size, dim = p_0.shape

    if requires_grad:
        # Training: keep the computation graph connected to the inputs.
        p = p_0 if p_0.requires_grad else p_0.requires_grad_(True)
        q = q_0 if q_0.requires_grad else q_0.requires_grad_(True)
        trajectory_p = [p]
        trajectory_q = [q]

        for _ in range(1, T):
            p, q = leapfrog_step(p, q, hamiltonian_fn, dt, create_graph=True)

            if check_nan_inf(p) or check_nan_inf(q):
                # Divergence: repeat the last finite state for the remaining steps.
                last_p = trajectory_p[-1].detach()
                last_q = trajectory_q[-1].detach()
                while len(trajectory_p) < T:
                    trajectory_p.append(last_p)
                    trajectory_q.append(last_q)
                break

            trajectory_p.append(p)
            trajectory_q.append(q)

        return torch.cat([torch.stack(trajectory_p, dim=0), torch.stack(trajectory_q, dim=0)], dim=2)

    # Inference / ground truth: no computation graph.
    trajectory = torch.zeros(T, batch_size, dim * 2, device=device)
    trajectory[0, :, :dim] = p_0.detach()
    trajectory[0, :, dim:] = q_0.detach()
    p = p_0.detach().clone().requires_grad_(True)
    q = q_0.detach().clone().requires_grad_(True)

    for step in range(1, T):
        p, q = leapfrog_step(p, q, hamiltonian_fn, dt, create_graph=False)

        if check_nan_inf(p) or check_nan_inf(q):
            break

        trajectory[step, :, :dim] = p.detach()
        trajectory[step, :, dim:] = q.detach()

    return trajectory


# =============================================================================
# HNN Model Integration (with dropout support)
# =============================================================================


def numerically_integrate(
    p_0: torch.Tensor,
    q_0: torch.Tensor,
    model: nn.Module,
    T: int,
    dt: float,
    k_matrix: Optional[torch.Tensor] = None,
    mask_index: Optional[int] = None,
    use_leapfrog: bool = True,
    requires_grad: bool = False,
    mask_index_fn: Optional[Callable[[int], int]] = None,
) -> torch.Tensor:
    """Integrate Hamiltonian dynamics with a learned HNN model; returns (T, batch, 2*dim) in [p, q] order.

    Args:
        mask_index: bank mask used at every Hamiltonian evaluation (None for models without a bank).
        mask_index_fn: optional callable taking a running Hamiltonian-evaluation counter (0, 1, 2, ..., one
            per kick / drift / kick sub-step of the leapfrog scheme) and returning the bank index for that
            evaluation; overrides ``mask_index``. All MaskedLinear layers of one evaluation share the index.
    """
    eval_counter = [0]

    def hamiltonian_fn(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if mask_index_fn is not None:
            idx = mask_index_fn(eval_counter[0])
            eval_counter[0] += 1
        else:
            idx = mask_index
        if k_matrix is not None:
            return model(p, q, k_matrix, index=idx)
        else:
            return model(p, q, index=idx)

    return integrate_trajectory_generic(
        p_0=p_0,
        q_0=q_0,
        hamiltonian_fn=hamiltonian_fn,
        T=T,
        dt=dt,
        use_leapfrog=use_leapfrog,
        requires_grad=requires_grad,
    )


_FRESH_MASK_SEMANTICS_VALUES = ("weight_group", "activation")


def model_uses_no_finite_mask(model: nn.Module) -> bool:
    """True iff ``model`` has MaskedLinear layers and none of them uses a finite mask bank (a Fresh-mask arm).

    Deterministic models (no MaskedLinear) and models with a mask bank return False, so
    ``fresh_mask_semantics`` does not affect them.
    """
    found_masked_linear = False
    for module in model.modules():
        if isinstance(module, MaskedLinear):
            found_masked_linear = True
            if module.finite_mask_mode != FiniteMaskMode.NONE:
                return False
    return found_masked_linear


def integrate_mc_samples(
    p_0: torch.Tensor,
    q_0: torch.Tensor,
    model: nn.Module,
    T: int,
    dt: float,
    n_mc_samples: int,
    k_matrix: Optional[torch.Tensor] = None,
    trajectory_consistent: bool = True,
    use_leapfrog: bool = True,
    return_mask_idxs: bool = False,
    mask_subset: Optional[torch.Tensor] = None,
    fresh_mask_semantics: str = "weight_group",
    generator: Optional[torch.Generator] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Monte Carlo rollouts of one batch of initial states under ``n_mc_samples`` dropout masks.

    Mask selection: ``mask_subset[:n_mc_samples]`` if given; else, with ``trajectory_consistent``, a random
    permutation of the bank truncated to ``n_mc_samples`` distinct masks (with a warning if the bank is
    smaller); else ``n_mc_samples`` independent uniform draws. Every rollout keeps its mask at all leapfrog
    sub-steps. All rollouts run as one batch (group i of ``n_mc * n_ics`` rows uses mask i).

    Args:
        return_mask_idxs: also return the bank index of each rollout, (n_mc,) int64, for scoring a rollout
            with the Hamiltonian of the mask that generated it.
        fresh_mask_semantics: how Fresh-mask models (no bank) draw masks. "weight_group": one fresh
            weight-space Bernoulli mask per MC group via ``model.batch_hamiltonian``, shared by every initial
            state of the group. "activation": per-mask sequential forward, i.e. per-element activation dropout
            as in training. Models with a bank or without MaskedLinear layers ignore this flag.
        generator: generator for the mask-index draws (its device is used for the draw); None uses the
            global RNG on the device of ``p_0``.

    Returns:
        trajectories (n_mc, T, n_ics, 2*dim), plus mask indices (n_mc,) if ``return_mask_idxs``.
    """
    if fresh_mask_semantics not in _FRESH_MASK_SEMANTICS_VALUES:
        raise ValueError(
            f"Unknown fresh_mask_semantics: {fresh_mask_semantics!r}. "
            f"Expected one of {_FRESH_MASK_SEMANTICS_VALUES}."
        )
    force_activation_path = fresh_mask_semantics == "activation" and (
        model_uses_no_finite_mask(model)
    )
    effective_mask_num = get_model_active_mask_num(model)
    if mask_subset is not None:
        mask_idxs = mask_subset[:n_mc_samples]
    elif trajectory_consistent:
        effective_n = min(n_mc_samples, effective_mask_num)
        if n_mc_samples > effective_mask_num:
            warnings.warn(
                f"integrate_mc_samples: n_mc_samples ({n_mc_samples}) > effective_mask_num "
                f"({effective_mask_num}); truncating to {effective_mask_num} trajectories.",
                UserWarning,
                stacklevel=2,
            )
        # Draw on the generator's device (CPU), then move to p_0's device as needed
        gen_device = generator.device if generator is not None else p_0.device
        perm = torch.randperm(effective_mask_num, device=gen_device, generator=generator)
        mask_idxs = perm[:effective_n]
        if gen_device != p_0.device:
            mask_idxs = mask_idxs.to(p_0.device)
    else:
        gen_device = generator.device if generator is not None else p_0.device
        mask_idxs = torch.randint(
            0, effective_mask_num, (n_mc_samples,), device=gen_device, generator=generator
        )
        if gen_device != p_0.device:
            mask_idxs = mask_idxs.to(p_0.device)

    n_mc = len(mask_idxs)
    n_ics = p_0.shape[0]
    dim = p_0.shape[-1]  # physical half-dim (p or q only)

    # Layout: p_0_exp[i*n_ics : (i+1)*n_ics] holds the initial states of mask i. The Hamiltonian
    # applies mask i to group i, so the gradients stay block-diagonal across groups.
    p_0_exp = (
        p_0.unsqueeze(0).expand(n_mc, -1, -1).reshape(-1, dim)
    )  # (n_mc*n_ics, dim)
    q_0_exp = q_0.unsqueeze(0).expand(n_mc, -1, -1).reshape(-1, dim)

    mask_idxs_int = [
        int(m.item()) if isinstance(m, torch.Tensor) else int(m) for m in mask_idxs
    ]

    def _batched_hamiltonian(
        p_batched: torch.Tensor, q_batched: torch.Tensor
    ) -> torch.Tensor:
        """H of every row, with mask i applied to rows [i*n_ics : (i+1)*n_ics].

        Uses ``model.batch_hamiltonian`` (one bmm over the groups) when the model has it, unless
        the activation-dropout path is forced for a Fresh-mask model; otherwise each group goes
        through the model's ordinary forward, which draws per-element dropout masks.
        """
        if not force_activation_path and hasattr(model, "batch_hamiltonian"):
            return model.batch_hamiltonian(
                p_batched, q_batched, k_matrix, mask_idxs_int
            )
        chunks = []
        for i, midx in enumerate(mask_idxs_int):
            p_i = p_batched[i * n_ics : (i + 1) * n_ics]
            q_i = q_batched[i * n_ics : (i + 1) * n_ics]
            if k_matrix is not None:
                chunks.append(model(p_i, q_i, k_matrix, index=midx))
            else:
                chunks.append(model(p_i, q_i, index=midx))
        return torch.cat(chunks, dim=0)  # (n_mc * n_ics,)

    traj_flat = integrate_trajectory_generic(
        p_0=p_0_exp,
        q_0=q_0_exp,
        hamiltonian_fn=_batched_hamiltonian,
        T=T,
        dt=dt,
        use_leapfrog=use_leapfrog,
        requires_grad=False,
    )  # (T, n_mc * n_ics, dim*2)

    trajectories = (
        traj_flat.reshape(T, n_mc, n_ics, dim * 2).permute(1, 0, 2, 3).contiguous()
    )  # (n_mc, T, n_ics, dim*2)
    if return_mask_idxs:
        return trajectories, mask_idxs.to(torch.long)
    return trajectories


# =============================================================================
# Ground Truth Hamiltonian Integration (for data generation)
# =============================================================================


def integrate_ground_truth(
    p_0: torch.Tensor,
    q_0: torch.Tensor,
    hamiltonian: nn.Module,
    T: int,
    dt: float,
    use_leapfrog: bool = True,
) -> torch.Tensor:
    """Integrate a ground-truth Hamiltonian (module H(p, q)) for dataset generation; returns (T, batch, 2*dim)."""

    def hamiltonian_fn(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return hamiltonian(p, q)

    return integrate_trajectory_generic(
        p_0=p_0,
        q_0=q_0,
        hamiltonian_fn=hamiltonian_fn,
        T=T,
        dt=dt,
        use_leapfrog=use_leapfrog,
        requires_grad=False,
    )
