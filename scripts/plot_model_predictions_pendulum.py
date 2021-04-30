import rlutils
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

"""
NOTE: From cos(theta) alone, cannot loslessly reconstruct theta because could be +/-.
However, because the problem is symmetrical can still study everything we need to.
"""

agent = rlutils.load("saved_runs/2021-04-29_17-14-46.agent")

if False: # On-policy distribution

    obs = rlutils.Observer(["cos_theta","sin_theta","theta_dot"], ["a"], extra=True)
    rlutils.deploy(agent, {"num_episodes": 1, "do_extra": True, "observe_freq": 1}, observer=obs)
    df = obs.dataframe()

    # Don't need both cos and sin.
    state = df[["cos_theta","theta_dot"]].values
    action = df["a"].values
    next_state_pred = df[["next_state_pred_0","next_state_pred_2"]].values
    ds = next_state_pred - state

    # Scatter states.
    plt.figure(); plt.scatter(state[:,0], state[:,1], s=10)
    plt.xlabel("$\cos(\\theta)$"); plt.ylabel("$\\frac{d\\theta}{dt}$")

    # Colour arrows by action.
    # mn = action.min(); rng = action.max() - mn
    # c = [mpl.cm.coolwarm((a - mn)/rng) for a in action]
    c = "k"

    # Quiver plot model predictions.
    fig = plt.figure(); ax = fig.gca(projection='3d')
    plt.quiver(state[:,0], state[:,1], action, ds[:,0], ds[:,1], np.zeros_like(action), color=c)
    ax.set_xlabel("$\cos(\\theta)$"); ax.set_ylabel("$\\frac{d\\theta}{dt}$"); ax.set_zlabel("Action")

if True: # Regular grid.

    import numpy as np
    import matplotlib.pyplot as plt

    # fig, axes = plt.subplots(1, 3)
    for n, action in enumerate([1]):

        theta_dot, cos_theta = np.mgrid[-4:4:10j, -1:1:10j] # Passing in complex number gives number of steps.
        # sin_theta = np.sqrt(1 - cos_theta**2) # NOTE: Will always be in range [0,1].
        
        fig = plt.figure(); # ax = fig.gca(projection='3d')
        for direction in [-1, 1]:
        
            sin_theta = direction * np.sqrt(1 - cos_theta**2)

            next_state_pred = np.zeros((10, 10, 3))
            for i in range(10):
                for j in range(10):
                    next_state_pred[i,j] = agent.predict(torch.Tensor([[cos_theta[i,j], sin_theta[i,j], theta_dot[i,j]]]), [action]).numpy()

            d_cos_theta = next_state_pred[:,:,0] - cos_theta 
            d_sin_theta = next_state_pred[:,:,1] - sin_theta 
            d_theta_dot = d_theta_dot = next_state_pred[:,:,2] - theta_dot

            theta = direction * np.arccos(cos_theta)
            d_theta = (direction * np.arccos(next_state_pred[:,:,0])) - theta

            """

            TODO: PLOT THETA NOT SIN AND COS

            """

            plt.quiver(theta, theta_dot, d_theta, d_theta_dot, width=1e-3)

            # axes[n].set_title(f"Action = {action}")
            # axes[n].streamplot(cos_theta, theta_dot, d_cos_theta, d_theta_dot, color=abs(d_theta_dot), cmap="inferno")
            # axes[n].set_xlabel("$\cos(\\theta)$"); axes[n].set_ylabel("$\\frac{d\\theta}{dt}$")

            
            # plt.quiver(cos_theta.flatten(), sin_theta.flatten(), theta_dot.flatten(), 
                    # d_cos_theta.flatten(), d_sin_theta.flatten(), d_theta_dot.flatten(), arrow_length_ratio=0.1)
                   
        break

plt.tight_layout()
plt.show()