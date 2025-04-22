import torch
import torch.nn as nn

from torch.distributions import Normal
from torch.func import vmap, functional_call, stack_module_state

from rsl_rl.utils import resolve_nn_activation


from typing import Sequence


class ActorMultiCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_critics: int,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # === Actor ===
        self.actor = self._build_mlp(num_actor_obs, actor_hidden_dims, num_actions, activation)

        # === Multi-Critic ===
        self.num_critics = num_critics

        critics = [
            self._build_mlp(num_critic_obs, critic_hidden_dims, 1, activation)
            for _ in range(num_critics)
        ]

        self.critic_params_raw, self.critic_buffers = stack_module_state(critics)

        # 注册参数：替换 key 中的 '.' 为 '__'
        encoded_params = {
            encode_param_name(name): nn.Parameter(tensor)
            for name, tensor in self.critic_params_raw.items()
        }
        self.critic_param_dict = nn.ParameterDict(encoded_params)

        # buffers 不注册，保留原样（functional_call 会用到）
        self.critic_buffers_dict = self.critic_buffers

        def wrapper(params, buffers, x):
            # decode names back to original (with .) for functional_call
            decoded_params = {
                decode_param_name(k): v for k, v in params.items()
            }
            return functional_call(critics[0], (decoded_params, buffers), x)

        self._critic_func = vmap(wrapper, in_dims=(0, 0, None))

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def _build_mlp(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, activation: nn.Module):
        layers = [nn.Linear(input_dim, hidden_dims[0]), activation]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), activation]
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        return nn.Sequential(*layers)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        """

        Args:
            critic_observations (torch.Tensor): (num_envs, critic_obs_dim)
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: (num_envs, num_critics)
        """
        params = {k: v for k, v in self.critic_param_dict.items()}
        buffers = self.critic_buffers_dict
        batched_out = self._critic_func(params, buffers, critic_observations)
        return batched_out.squeeze(-1).transpose(0, 1)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)

        return True


def encode_param_name(name: str) -> str:
    return name.replace('.', '__')


def decode_param_name(name: str) -> str:
    return name.replace('__', '.')
