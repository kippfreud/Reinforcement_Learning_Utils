import rlutils
from rlutils.specific.Pendulum import reward_function

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

"""

True phase plane plot for inverted pendulum with and without control: https://bit.ly/3vpoOVc

"""

agent = rlutils.load("runs/crisp-gorge-50.agent")

if False: # On-policy distribution

    obs = rlutils.Observer(state_dims=3, action_dims=1, do_extra=True)
    rlutils.deploy(agent, {"num_episodes": 2, "do_extra": True, "observe_freq": 1}, observer=obs)
    df = obs.dataframe()

    theta = np.arccos(df["s_0"].values) * np.sign(df["s_1"].values) # Convert from cos(theta) and sin(theta) to theta.
    theta_dot = df["s_2"].values
    action = df["a"].values

    d_theta = (np.arccos(df["next_state_pred_0"].values) * np.sign(df["next_state_pred_1"].values)) - theta
    d_theta_dot = df["next_state_pred_2"].values - theta_dot

    # Scatter states.
    plt.figure(); plt.scatter(theta, theta_dot, s=10)
    plt.xlabel("$\\theta$"); plt.ylabel("$\\frac{d\\theta}{dt}$")

    # Colour arrows by action.
    # # mn = action.min(); rng = action.max() - mn
    # # c = [mpl.cm.coolwarm((a - mn)/rng) for a in action]
    c = "k"

    # Quiver plot model predictions.
    fig = plt.figure(); ax = fig.gca(projection="3d")
    plt.quiver(theta, theta_dot, action, d_theta, d_theta_dot, np.zeros_like(action), angles="xy", color=c)
    ax.set_xlabel("$\\theta$"); ax.set_ylabel("$\\frac{d\\theta}{dt}$"); ax.set_zlabel("Action")

if True: # Regular grid.

    theta_dot, theta = np.mgrid[-4:4:60j, -0.5:0.5:60j] # Good for revealing errors near upright.
    # theta_dot, theta = np.mgrid[-4:4:30j, -2:2:30j] 
    # theta_dot, theta = np.mgrid[-8:8:30j, -np.pi:np.pi:30j] 

    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    for n, action in enumerate([0,"$\pi$"]):
        
        fig = plt.figure(); # ax = fig.gca(projection='3d')
       
        next_state_pred = np.zeros((theta.shape[0], theta.shape[1], agent.P["num_models"], 3))
        reward = np.zeros_like(theta)
        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                state = torch.Tensor([[cos_theta[i,j], sin_theta[i,j], theta_dot[i,j]]])
                if action == "$\pi$": a = agent.act(state, explore=False)[0].item()
                else: a = action
                next_state_pred[i,j] = agent.predict(state, [a], mode="all").numpy()
                reward[i,j] = reward_function(state[0], a)
        
        var = next_state_pred.var(axis=2)
        var_norm_sum = (var / var.max(axis=(0,1))).sum(axis=2)

        d_theta = (np.arccos(np.clip(next_state_pred[:,:,:,0].mean(axis=2), -1, 1)) * np.sign(next_state_pred[:,:,:,1].mean(axis=2))) - theta
        d_theta_dot = d_theta_dot = next_state_pred[:,:,:,2].mean(axis=2) - theta_dot

        # print(d_theta)

        plt.imshow(
                    # reward, 
                    var_norm_sum,
            extent=(theta.min(), theta.max(), theta_dot.min(), theta_dot.max()), aspect="auto", cmap="coolwarm")
        plt.colorbar()

        plt.quiver(theta, theta_dot, d_theta, d_theta_dot, angles="xy", width=1e-3)
        
        # plt.scatter(theta, theta_dot, s=5, c="gray")

        # plt.streamplot(theta, theta_dot, d_theta, d_theta_dot, color="k")#, color=abs(d_theta), cmap="inferno")

        plt.title(f"Action = {action}"); plt.xlabel("$\\theta$"); plt.ylabel("$\\frac{d\\theta}{dt}$")
        
                   
# plt.tight_layout()
plt.show()