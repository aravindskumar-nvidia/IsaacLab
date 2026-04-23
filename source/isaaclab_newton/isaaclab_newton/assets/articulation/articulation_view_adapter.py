# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapter that extends Newton's :class:`ArticulationView` with a PhysX-compatible
:meth:`get_jacobians` method so that IsaacLab action terms (e.g.
:class:`DifferentialInverseKinematicsAction`) work unchanged with the Newton backend."""

from __future__ import annotations

import torch
import warp as wp
from newton.selection import ArticulationView as _NewtonArticulationView

from isaaclab_newton.physics import NewtonManager as SimulationManager


class ArticulationView(_NewtonArticulationView):
    """Newton :class:`ArticulationView` extended with :meth:`get_jacobians`.

    The method mirrors the shape convention of the PhysX ``ArticulationView.get_jacobians()``
    so that downstream code (IK, OSC action terms) can index the result identically.

    PhysX convention:

    * **Fixed-base** — shape ``(count, numLinks - 1, 6, max_dofs)``.
      The root body row is excluded and the DOF columns cover only actuated joints.
    * **Floating-base** — shape ``(count, numLinks, 6, max_dofs + 6)``.
      All body rows are kept and the first 6 DOF columns correspond to the root
      free-joint.

    Newton's :meth:`eval_jacobian` returns the *spatial* (Plücker) Jacobian where
    the linear rows encode the velocity of the world-origin point on the body,
    whereas PhysX returns the *geometric* Jacobian where the linear rows encode
    the velocity of the body origin.  The adapter converts between the two via:

    .. math:: v_{body} = v_{origin} + \\omega \\times p_{body}
    """

    def get_jacobians(self) -> wp.array:
        """Compute and return the geometric Jacobian in PhysX-compatible layout.

        Returns:
            Warp array of shape ``(count, num_bodies, 6, num_dofs)``.
        """
        state = SimulationManager.get_state_0()

        # eval_jacobian returns shape (model.articulation_count, max_links * 6, max_dofs).
        J_full = self.eval_jacobian(state)

        max_links = self.model.max_joints_per_articulation
        max_dofs = self.model.max_dofs_per_articulation

        J_torch = wp.to_torch(J_full)

        # Extract rows belonging to this view.
        arti_ids_torch = wp.to_torch(self.articulation_ids).reshape(-1).long()
        J_view = J_torch[arti_ids_torch]  # (view_count, max_links * 6, max_dofs)

        view_count = J_view.shape[0]
        J_view = J_view.reshape(view_count, max_links, 6, max_dofs)

        # Convert Plücker → geometric: v_body = v_origin + ω × p_body
        body_pos = self._gather_body_positions(state, view_count, max_links)
        ang = J_view[:, :, 3:, :]  # (view_count, max_links, 3, max_dofs)
        p = body_pos.unsqueeze(-1).expand_as(ang)
        J_view = J_view.clone()
        J_view[:, :, :3, :] += torch.cross(ang, p, dim=2)

        # Strip root body row for fixed-base (PhysX excludes the immovable root).
        if self.is_fixed_base:
            J_view = J_view[:, 1:, :, :]

        return wp.from_torch(J_view)

    def _gather_body_positions(self, state: object, view_count: int, max_links: int) -> torch.Tensor:
        """Return world-frame body positions in Jacobian row order.

        Returns:
            Tensor of shape ``(view_count, max_links, 3)``.
        """
        if not hasattr(self, "_body_index_map"):
            self._body_index_map = self._build_body_index_map(max_links)

        body_q = wp.to_torch(state.body_q)  # (total_bodies, 7)
        flat_idx = self._body_index_map.reshape(-1)
        return body_q[flat_idx, :3].reshape(view_count, max_links, 3)

    def _build_body_index_map(self, max_links: int) -> torch.Tensor:
        """Build a ``(view_count, max_links)`` index tensor mapping each Jacobian
        row to its body index in ``state.body_q``.

        The Jacobian kernel iterates joints ``joint_start .. joint_end-1`` for
        each articulation, putting joint *i*'s contribution into row *i*. The
        body for row *i* is ``joint_child[joint_start + i]``.
        """
        art_start = wp.to_torch(self.model.articulation_start).long()
        joint_child = wp.to_torch(self.model.joint_child).long()
        arti_ids = wp.to_torch(self.articulation_ids).reshape(-1).long()

        joint_starts = art_start[arti_ids]  # (view_count,)
        offsets = torch.arange(max_links, device=joint_starts.device)
        joint_indices = joint_starts.unsqueeze(1) + offsets.unsqueeze(0)
        return joint_child[joint_indices]  # (view_count, max_links)
