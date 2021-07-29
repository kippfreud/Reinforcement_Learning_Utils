"""
Diversity Is All You Need (DIAYN).
"""

from .sac import SacAgent # DIAYN inherits from SAC.
from ._default_hyperparameters import default_hyperparameters
from ..common.networks import SequentialNetwork
from ..common.utils import col_concat, one_hot

import numpy as np
import torch
import torch.nn.functional as F


class DiaynAgent(SacAgent):
    def __init__(self, env, hyperparameters):
        # Overwrite default hyperparameters for SAC.
        P = default_hyperparameters["sac"]
        for k, v in default_hyperparameters["diayn"]["sac_parameters"].items(): P[k] = v
        for k, v in hyperparameters["sac_parameters"].items(): P[k] = v
        if len(env.observation_space.shape) > 1: raise NotImplementedError()
        else: 
            self.state_dim = env.observation_space.shape[0]       
            # Set up augmented observation shape for SAC.
            P["aug_obs_shape"] = (self.state_dim + hyperparameters["num_skills"],)
        SacAgent.__init__(self, env, P)
        # Add parameters specific to DIAYN.
        for k, v in hyperparameters.items(): 
            if k != "sac_parameters": self.P[k] = v
        # Create skill discriminator network, optionally accepting action dimensions as inputs alongside state dimensions.
        disc_input = self.state_dim + (self.env.action_space.shape[0] if self.P["include_actions"] else 0)
        self.discriminator = SequentialNetwork(code=self.P["net_disc"], input_shape=disc_input, output_size=self.P["num_skills"], lr=self.P["lr_disc"]).to(self.device)
        # Skill distribution.
        self.p_z = np.full(self.P["num_skills"], 1.0 / self.P["num_skills"]) # NOTE: Uniform.
        # Tracking variables.
        self.skill = self._sample_skill() # Initialise skill.
        self.ep_losses_discriminator = []
        self.ep_pseudo_reward_sum = 0.

    def act(self, state, skill=None, explore=True, do_extra=False):
        """Augment state with one-hot skill vector, then use SAC action selection."""
        if skill is None: skill = self.skill
        state_aug = col_concat(state, one_hot(skill, self.P["num_skills"], self.device))
        action, extra = SacAgent.act(self, state_aug, explore, do_extra)
        if do_extra: extra["skill"] = skill
        return action, extra

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the discriminator, pi and Q network parameters."""
        if len(self.memory) < self.P["batch_size"]: return
        # Sample a batch and transpose it (see https://stackoverflow.com/a/19343/3343043).
        batch = self.memory.element(*zip(*self.memory.sample(self.P["batch_size"])))
        features, zs = torch.split(torch.cat(batch.state), [self.state_dim, self.P["num_skills"]], dim=1)
        if self.P["include_actions"]: features = col_concat(features, torch.cat(batch.action))
        # Update discriminator to minimise cross-entropy loss against skills.
        loss = F.cross_entropy(self.discriminator(features), zs.argmax(1))   
        self.discriminator.optimise(loss)
        self.ep_losses_discriminator.append(loss.item()) # Keeping separate prevents confusion of SAC methods.
        # Send same batch to SAC update function and return losses.
        return SacAgent.update_on_batch(self, batch)

    def per_timestep(self, state, action, _, next_state, done, skill=None):
        """Operations to perform on each timestep during training."""
        # Augment state and next state with one-hot skill vector and compute diversity reward.
        if skill is None: skill = self.skill
        z = one_hot(skill, self.P["num_skills"], self.device)
        pseudo_reward = self._pseudo_reward(
            col_concat(state, torch.tensor([action], device=self.device, dtype=torch.float)) 
            if self.P["include_actions"] else state, skill)
        self.ep_pseudo_reward_sum += pseudo_reward
        SacAgent.per_timestep(self, col_concat(state, z), action, pseudo_reward, col_concat(next_state, z), done)

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        out = SacAgent.per_episode(self)
        out["logs"]["discriminator_loss"] = np.mean(self.ep_losses_discriminator) if self.ep_losses_discriminator else 0.
        out["logs"]["pseudo_reward_sum"] = self.ep_pseudo_reward_sum
        del self.ep_losses_discriminator[:]; self.ep_pseudo_reward_sum = 0.
        self.per_episode_deploy()
        return out

    def per_episode_deploy(self):
        self.skill = self._sample_skill() # Resample skill for the next episode.

    def _sample_skill(self):
        """Sample skill according to probabilities in self.p_z.""" 
        return np.random.choice(self.P["num_skills"], p=self.p_z)

    def _pseudo_reward(self, features, skill):
        """Construct diversity-promoting pseudo-reward using skill discriminator network."""
        # Log conditional probability of skill according to discriminator.
        reward = - F.cross_entropy(self.discriminator(features), torch.tensor([skill])).item()
        if self.P["log_p_z_in_reward"]:
            # Log prior probability of skill. Subtracting this means rewards are non-negative as long as the discriminator does better than chance.
            reward -= np.log(self.p_z[skill])
        return reward