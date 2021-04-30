from ..agents._default_hyperparameters import default_hyperparameters

from torch import load as torch_load


def make(agent, env, hyperparameters=dict()):
    """
    Make an instance of an agent class to train/deploy in env, overwriting default hyperparameters with those provided.
    """
    agent = agent.lower()
    # Special treatment for TD3 (a variant of DDPG).
    if agent == "td3": hyperparameters["td3"] = True; agent = "ddpg"
    assert agent in default_hyperparameters
    # Overwrite default hyperparameters where applicable.
    P = default_hyperparameters[agent]
    for k, v in hyperparameters.items(): P[k] = v
    # Load agent class.
    if agent == "actor_critic":         from ..agents.actor_critic import ActorCriticAgent as agent_class
    elif agent == "ddpg":               from ..agents.ddpg import DdpgAgent as agent_class
    elif agent == "dqn":                from ..agents.dqn import DqnAgent as agent_class
    elif agent == "off_policy_mc":      from ..agents.off_policy_mc import OffPolicyMCAgent as agent_class
    elif agent == "random":             from ..agents.random import RandomAgent as agent_class
    elif agent == "reinforce":          from ..agents.reinforce import ReinforceAgent as agent_class
    elif agent == "sac":                from ..agents.sac import SacAgent as agent_class
    elif agent == "simple_model_based": from ..agents.simple_model_based import SimpleModelBasedAgent as agent_class
    elif agent == "stable_baselines":   from ..agents.stable_baselines import StableBaselinesAgent as agent_class
    return agent_class(env, P)

def load(path): return torch_load(path)