# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .distillation import Distillation
from .ppo import PPO
from .multi_critic_ppo import MultiCriticPPO

__all__ = ["PPO", "MultiCriticPPO", "Distillation"]
