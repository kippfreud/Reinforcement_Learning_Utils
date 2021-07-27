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
        # Overwrite default hyperparameters for DDPG.
        P = default_hyperparameters["sac"]
        for k, v in default_hyperparameters["diayn"]["sac_parameters"].items(): P[k] = v
        for k, v in hyperparameters["sac_parameters"].items(): P[k] = v
        SacAgent.__init__(self, env, P)
        # Add parameters specific to DIAYN.
        for k, v in hyperparameters.items(): 
            if k != "sac_parameters": self.P[k] = v
        if len(self.env.observation_space.shape) > 1: raise NotImplementedError()
        else: self.state_dim = self.env.observation_space.shape[0]        
        # Create pi network. NOTE: This overwrites the network previously created on SacAgent.__init__, which is a little inelegant.  
        self.pi = SequentialNetwork(code=self.P["net_pi"], input_shape=self.state_dim+self.P["num_skills"], output_size=2*self.env.action_space.shape[0], lr=self.P["lr_pi"]).to(self.device)
        # Create discriminator network.
        self.discriminator = SequentialNetwork(code=self.P["net_disc"], input_shape=self.state_dim, output_size=self.P["num_skills"], lr=self.P["lr_disc"]).to(self.device)
        # Tracking variables.
        self.skill = "x"
        self.ep_losses_discriminator = []

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
        states_aug = torch.cat(batch.state)

        """
        (obs, z_one_hot) = self._split_obs()
        if self._include_actions:
            logits = self._discriminator.get_output_for(obs, self._action_pl,
                                                        reuse=True)
        else:
            logits = self._discriminator.get_output_for(obs, reuse=True) 

        self._discriminator_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=z_one_hot, # NOTE: Cross-entropy loss between discriminator output and one-hot skill encoding 
                                                    logits=logits)
        )
        optimizer = tf.train.AdamOptimizer(self._discriminator_lr)
        discriminator_train_op = optimizer.minimize(
            loss=self._discriminator_loss,
            var_list=self._discriminator.get_params_internal()
        )
        self._training_ops.append(discriminator_train_op)
        """
        
        self.discriminator.optimise(loss)
        self.ep_losses_discriminator.append(loss.item()) # Keeping separate prevents confusion of SAC methods.
        # Send same batch to SAC update function and return losses.
        return SacAgent.update_on_batch(self, batch)

    def per_timestep(self, state, action, _, next_state, done):
        """Operations to perform on each timestep during training."""
        # 
        #
        #
        #
        SacAgent.per_timestep(self, state_aug, action, reward, next_state_aug, done, suppress_update=self.random_mode)

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        out = SacAgent.per_episode(self)
        out["logs"]["discriminator_loss"] = np.mean(self.ep_losses_discriminator) if self.ep_losses_discriminator else 0.
        del self.ep_losses_discriminator[:]
        return out