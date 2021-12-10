import hyperrectangles as hr

import os
import numpy as np
np.set_printoptions(precision=3, suppress=True, edgeitems=30, linewidth=100000)   
from scipy.stats import norm
import networkx as nx
import cv2
from tqdm import tqdm
from joblib import dump
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class PbrlObserver:
    def __init__(self, P, dim_names, run_names=[], episodes=[], interface=None):
        # self.env = env
        self.P = P # Dictionary containing hyperparameters.
        self.dim_names = dim_names
        self.run_names = run_names # Can be multiple; order crucial to match with episodes.
        self.load_episodes(episodes)
        self.interface = interface
        if "feedback_freq" in P and P["feedback_freq"] > 0:
            # Compute batch size for online.
            b = P["num_episodes_before_freeze"] / P["feedback_freq"]
            assert b % 1 == 0
            self.num_batches = int(b)
        # Initialise empty tree.
        space = hr.Space(dim_names=["ep", "reward"] + dim_names)
        root = hr.node.Node(space, sorted_indices=space.all_sorted_indices) 
        self.tree = hr.tree.Tree(
            name="reward_model", 
            root=root, 
            split_dims=space.idxify(dim_names), 
            eval_dims=space.idxify(["reward"])
            )
        # Mean and variance of reward components.  
        self.r, self.var = np.zeros(self.m), np.zeros(self.m) 
        # Overwrite env reward function with the method of this class.
        # self.env.reward = self.reward
        # History of tree modifications.
        self.history = {}

    @property
    def feedback_count(self): return np.triu(np.invert(np.isnan(self.Pr))).sum()
    @property
    def m(self): return len(self.tree.leaves)

# ==============================================================================
# PREDICTION FUNCTIONS

    def phi(self, transition):
        """
        Map a transition to a component index.
        """
        return self.tree.leaves.index(next(iter(self.tree.propagate([None,None]+list(transition), mode="max"))))

    def n(self, trajectory):
        """
        Map a trajectory to a component counts vector.
        """
        n = np.zeros(self.m, dtype=int)
        for x in trajectory: n[self.phi(x[:len(self.dim_names)])] += 1
        return n

    def reward(self, state, action, next_state, reward_original=0, done=False, info={}, meanvar=False, stochastic=False):
        """
        Reward function, defined over individual transitions. Directly usable in OpenAI Gym.
        """
        x = self.phi(list(state) + list(action) + list(next_state))
        if meanvar: reward = (self.r[x], self.var[x])
        elif stochastic: reward = np.random.normal(loc=self.r[x], scale=np.sqrt(self.var[x]))   
        else: reward = self.r[x]
        return reward, done, {"reward_components": [r if xx==x else 0 for xx, r in enumerate(self.r)], 
                              "reward_original": reward_original}

    def F(self, trajectory_i, trajectory_j=None):
        """
        Fitness function, defined over trajectories. Returns mean and variance.
        """
        n = (self.n(trajectory_i) if trajectory_j is None else self.n(trajectory_i)-self.n(trajectory_j))
        return [np.matmul(n, self.r), np.matmul(n, np.matmul(np.diag(self.var), n.T))]

    def F_ucb_for_pairs(self, episodes):
        """
        Compute UCB fitness for a list of episodes and sum for all pairs to create a matrix.
        """
        mu, var = np.array([self.F(ep) for ep in episodes]).T
        F_ucb = mu + self.P["ucb_num_std"] * np.sqrt(var)
        return np.add(F_ucb.reshape(-1,1), F_ucb.reshape(1,-1))

    def Pr_pred(self, trajectory_i, trajectory_j): 
        """
        Predicted probability of trajectory i being preferred to trajectory j.
        """
        F_diff, F_var = self.F(trajectory_i, trajectory_j)
        with np.errstate(divide="ignore"): return norm.cdf(F_diff / np.sqrt(F_var))

# ==============================================================================
# FUNCTIONS FOR EXECUTING THE LEARNING PROCESS

    def load_episodes(self, episodes):
        """
        Load a dataset of episodes and initialise data structures.
        """
        self.episodes = episodes
        self.Pr = np.full((len(episodes), len(episodes)), np.nan)
        self.n_on_prev_feedback = 0
        self.current_batch_num = 1

    def per_timestep(self, _, t, state, action, next_state, __, ___, ____, _____):     
        """
        Store transition for current timestep.
        """
        self.episodes[-1].append(list(state.cpu().numpy().flatten()) + list(action) + list(next_state))
            
    def per_episode(self, _): 
        """
        NOTE: To ensure video saving, this is completed *after* env.reset() is called for the next episode.
        """            
        n = len(self.episodes)
        if n > 0:
            self.episodes[-1] = np.array(self.episodes[-1]) # Convert to NumPy after appending finished.
            if "feedback_freq" in self.P and self.P["feedback_freq"] > 0 and \
                n <= self.P["num_episodes_before_freeze"] and n % self.P["feedback_freq"] == 0:    
                # Gather a batch of feedback.
                K = self.P["feedback_budget"] # Total feedback budget.
                B = self.num_batches # Number of feedback batches.
                f = self.P["feedback_freq"] # Number of episodes between feedback batches.
                c = self.P["scheduling_coef"] # How strongly to apply scheduling.
                b = self.current_batch_num # Current batch number.
                k_max = (K / B * (1 - c)) + (K * (f * (2*b - 1) - 1) / (B * (B*f - 1)) * c)
                self.get_feedback(k_max=round(k_max))
                # Update reward function.
                self.update(history_key=(n-1))
                self.n_on_prev_feedback = n
                self.current_batch_num += 1 
        # Expand data structures.
        self.episodes.append([])
        Pr_old = self.Pr; self.Pr = np.full((n+1, n+1), np.nan); self.Pr[:-1,:-1] = Pr_old
        return {"feedback_count": self.feedback_count}

    def get_feedback(self, k_max=1, ij=None): 
        """
        xxx
        """
        # TODO: Make sampler class in similar way to interface class
        if "ucb" in self.P["sampling_mode"]: w = self.F_ucb_for_pairs(self.episodes) # Only need to compute once per batch.
        self.interface.open()
        for k in range(k_max):
            if ij is None:
                if self.P["sampling_mode"] == "ucb_constrained":
                    found, i, j, _ = self.select_i_j_constrained(w, ij_min=self.n_on_prev_feedback)
                else: raise NotImplementedError()
                if not found: print("=== All rated ==="); break
            else: assert k_max == 1; i, j = ij # Force specified i, j
            y_ij = self.interface(i, j)
            if y_ij == "esc": print("=== Feedback exited ==="); break
            assert 0 <= y_ij <= 1
            self.Pr[i, j] = y_ij
            self.Pr[j, i] = 1 - y_ij
            print(f"{k+1} / {k_max}: P({i} > {j}) = {y_ij}")
        self.interface.close()

    def select_i_j_constrained(self, w, ij_min=0):
        """
        Sample a trajectory pair from a weighting matrix subject to constraints.
        """
        # Enforce non-repeat constraint...
        n = self.Pr.shape[0]; assert w.shape == (n, n)
        not_rated = np.isnan(self.Pr)
        if not_rated.sum() <= n: return False, None, None, None # If have all possible ratings, no more are possible.
        p = w.copy()
        rated = np.invert(not_rated)
        p[rated] = np.nan
        # ...enforce non-identity constraint...
        np.fill_diagonal(p, np.nan)
        # ...enforce connectedness constraint...    
        unconnected = np.argwhere(rated.sum(axis=1) == 0).flatten()
        if len(unconnected) < n: p[unconnected] = np.nan # (ignore connectedness if first ever rating)
        # ...enforce recency constraint...
        p[:ij_min, :ij_min] = np.nan
        nans = np.isnan(p)
        # ... and rescale into a probability distribution.
        p -= np.nanmin(p)
        sm = np.nansum(p)
        if sm == 0: p[np.invert(nans)] = 1; p /= np.nansum(p)
        else: p /= sm
        p[nans] = 0
        # Then sample a pair.
        i, j = np.unravel_index(np.random.choice(p.size, p=p.ravel()), p.shape)
        # Sense check.
        if len(unconnected) < n: assert np.invert(not_rated[i]).sum() > 0 
        assert i >= ij_min or j >= ij_min 
        return True, i, j, p

    def update(self, history_key, reset_tree=False):
        """
        Update the reward function to reflect the current feedback dataset.
        If reset_tree=True, tree is first pruned back to its root (i.e. start from scratch).
        """
        n = len(self.episodes)
        Pr_train, Pr_val = train_val_split(self.Pr)
        # Compute episode-level fitness estimates and scale by episode lengths to get reward targets.
        A, d = construct_A_and_d(Pr_train, self.P["p_clip"])
        ep_fitness_cv = fitness_case_v(A, d)
        ep_length = np.array([len(ep) for ep in self.episodes])
        reward_target = ep_fitness_cv * ep_length.mean() / ep_length
        # Populate tree with episodes that are connected to the training set comparison graph.
        include = np.invert(np.isnan(Pr_train)).sum(axis=1) > 0 
        print(f"Including {include.sum()} / {n} episodes")
        self.tree.space.data = np.hstack((
            np.array([[[i, reward_target[i]]] * ep_length[i] for i in range(n) if include[i]]).reshape(-1,2), # Episode number and reward target.
            np.vstack([self.episodes[i][:,:len(self.dim_names)] for i in range(n) if include[i]])             # Transition vector.
            ))
        if reset_tree: self.tree.prune_to(self.tree.root) # NOTE:
        self.tree.populate()
        # Perform best-first splitting until m_max is reached.
        history_split = []        
        with tqdm(total=self.P["m_max"], initial=self.m, desc="Splitting") as pbar:
            while self.m < self.P["m_max"] and len(self.tree.split_queue) > 0:
                result = self.tree.split_next_best(min_samples_leaf=self.P["min_samples_leaf"]) 
                if result is not None:
                    pbar.update(1)
                    node, dim, threshold = result
                    history_split.append([self.m, node, dim, threshold, None, sum([q[1] for q in self.tree.split_queue])])        
        # Perform minimal cost complexity pruning until labelling loss is minimised.
        N = np.array([self.n(ep) for ep in self.episodes])
        tree_before_merge = self.tree.clone()                
        history_merge, parent_num, pruned_nums = [], None, None
        with tqdm(total=self.P["m_max"], initial=self.m, desc="Merging") as pbar:
            while True: 
                # Measure loss.
                r, var = np.array(self.tree.gather(("mean","reward"))), np.array(self.tree.gather(("var","reward")))
                history_merge.append([self.m, parent_num, pruned_nums,
                    labelling_loss(A, d, N, r, var, self.P["p_clip"]), # True labelling loss.
                    sum([q[1] for q in self.tree.split_queue]) # Proxy loss: variance.
                    ])
                if self.m <= self.P["m_stop_merge"]: break
                # Perform prune.
                parent_num, pruned_nums = self.tree.prune_mccp()
                pbar.update(-1)
                # Efficiently update N array.
                assert pruned_nums[-1] == pruned_nums[0] + len(pruned_nums)-1
                N[:,pruned_nums[0]] = N[:,pruned_nums].sum(axis=1)
                N = np.delete(N, pruned_nums[1:], axis=1)
                if False: assert (N == np.array([self.n(ep) for ep in self.episodes])).all() # Sense check.     
        # Now prune to minimum-loss size.
        optimum = np.argmin([l + (self.P["alpha"] * m) for m,_,_,l,_ in history_merge]) # NOTE: Apply size regularisation here.
        self.tree = tree_before_merge # Reset to pre-merging stage.
        for _, parent_num, pruned_nums_prev, _, _ in history_merge[:optimum+1]: 
            if parent_num is None: continue # First entry of history_merge will have this.
            pruned_nums = self.tree.prune_to(self.tree._get_nodes()[parent_num])
            assert set(pruned_nums_prev) == set(pruned_nums)
        self.r, self.var = np.array(self.tree.gather(("mean","reward"))), np.array(self.tree.gather(("var","reward")))   
        # Store history.
        # history_split, history_merge = split_merge_cancel(history_split, history_merge)
        self.history[history_key] = {"split": history_split, "merge": history_merge}
        # Save out some visualisations.
        print(self.tree.space)
        print(self.tree)
        path = f"run_logs/{self.run_names[-1]}"
        if not os.path.exists(path): os.makedirs(path)
        if False and n % self.P["plot_freq"] == 0:
            if True:
                plt.figure()
                m_range = [m for m,_,_,_,_ in history_merge]
                plt.plot(m_range, [l for _,_,_,l,_ in history_merge], c="k")
                plt.plot([m_range[0], m_range[-1]], [self.P["alpha"] * m_range[0], self.P["alpha"] * m_range[-1]], c="r")
                plt.scatter(self.m, history_merge[optimum][3], c="g")
                plt.gca().set_ylim(bottom=0)
                plt.savefig(f"{path}/loss_{n}.png")
            if True: 
                self.show_comparison_matrix()
                plt.savefig(f"{path}/matrix_{n}.png")
            if False: 
                self.show_alignment()
                plt.savefig(f"{path}/alignment_{n}.png")
            if True: 
                self.show_fitness_pdfs()
                plt.savefig(f"{path}/pdfs_{n}.png")
            if True: 
                for vis_dims, vis_lims in [([2, 3], [[0,10],[0,10]])]:
                    self.show_rectangles(vis_dims, vis_lims)
                    plt.savefig(f"{path}/{vis_dims}_{n}.png")
            if True:
                _, _, _, p = self.select_i_j_constrained(self.F_ucb_for_pairs(self.episodes), ij_min=self.n_on_prev_feedback)
                plt.figure()
                plt.imshow(p, interpolation="none")
                plt.savefig(f"{path}/psi_matrix_{n}.png")
            if True:
                hr.diagram(self.tree, pred_dims=["reward"], verbose=True, out_name=f"{path}/diagram_{n}", png=True)
                
    def save(self): 
        self.episodes[-1] = np.array(self.episodes[-1])
        path = f"run_logs/{self.run_names[-1]}"
        if not os.path.exists(path): os.makedirs(path)
        dump(self, f"{path}/checkpoint_{len(self.episodes)}.joblib")

# ==============================================================================
# VISUALISATION

    def show_comparison_matrix(self):
        """Binary matrix showing which comparisons have been made."""
        plt.figure()
        plt.imshow(np.invert(np.isnan(self.Pr)), norm=Normalize(0, 1), interpolation="none")

    def show_alignment(self, vs="ground_truth", ax=None):
        """Decomposed fitness (+/- 1 std) vs a baseline, either:
            - Case V fitness, or
            - Ground truth fitness if an oracle is available    
        """
        if vs == "case_v":
            baseline = fitness_case_v(*construct_A_and_d(self.Pr, self.P["p_clip"]))
            xlabel = "Case V Fitness"
        elif vs == "ground_truth":
            assert self.oracle is not None
            if type(self.oracle) == list: baseline = self.oracle
            else: baseline = [self.oracle(ep, []) for ep in self.episodes]
            xlabel = "Ground Truth Fitness"
        ranking = np.argsort(baseline); baseline.sort()
        mu, var = np.array([self.F(self.episodes[i]) for i in ranking]).T
        std = np.sqrt(var)
        if ax is None: _, ax = plt.subplots()
        ax.plot(baseline, mu, c="k")
        ax.fill_between(baseline, mu-std, mu+std, color=[.8,.8,.8], zorder=-1, lw=0)
        ax.set_xlabel(xlabel); ax.set_ylabel("Decomposed Fitness")

    def show_fitness_pdfs(self):
        """PDFs of fitness predictions."""
        mu, var = np.array([self.F(ep) for ep in self.episodes]).T
        mn, mx = np.min(mu - 3*var**.5), np.max(mu + 3*var**.5)
        rng = np.arange(mn, mx, (mx-mn)/1000)
        P = np.array([norm.pdf(rng, m, v**.5) for m, v in zip(mu, var)])
        P /= P.max(axis=1).reshape(-1, 1)
        plt.figure(figsize=(5, 15))
        # for p in P: plt.plot(rng, p)
        plt.imshow(P, 
        aspect="auto", extent=[mn, mx, len(self.episodes)-0.5, -0.5], interpolation="None")
        plt.yticks(range(len(self.episodes)), fontsize=6)

    def show_rectangles(self, vis_dims, vis_lims):
        """Projected hyperrectangles."""
        ax = hr.show_rectangles(self.tree, vis_dims, attribute=("mean", "reward"), 
            vis_lims=vis_lims, 
            cmap_lims=[-1, 1], 
            maximise=False
            )
        hr.show_leaf_numbers(self.tree, vis_dims, ax=ax)
        if False: # Overlay episodes.
            for i, ep in enumerate(self.episodes): ax.plot(ep[:,vis_dims[0]-2], ep[:,vis_dims[1]-2], alpha=1, label=i)
            ax.legend()

    def show_comparison_graph(self):
        # Graph creation.
        self.graph = nx.DiGraph()
        n = len(self.episodes)
        self.graph.add_nodes_from(range(n), fitness=0)
        for i in range(n): self.graph.nodes[i]["fitness"] = self.F(self.episodes[i])[0]
        self.graph.add_weighted_edges_from([(j, i, self.Pr[i,j]) for i in range(n) for j in range(n) if not np.isnan(self.Pr[i,j])])
        # Graph plotting.
        plt.figure(figsize=(12, 12))
        pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog="neato")
        nx.draw_networkx_nodes(self.graph, pos=pos, linewidths=0,
            node_color=list(nx.get_node_attributes(self.graph, "fitness").values()),
            cmap="coolwarm_r"
        )
        edge_collection = nx.draw_networkx_edges(self.graph, pos=pos, 
            connectionstyle="arc3,rad=0.1",
        )
        weights = list(nx.get_edge_attributes(self.graph, "weight").values())
        for i, e in enumerate(edge_collection): e.set_alpha(weights[i])
        nx.draw_networkx_labels(self.graph, pos=pos)
        # nx.draw_networkx_edge_labels(self.graph, pos=pos, label_pos=0.4, font_size=6,
        #     edge_labels={(i, j): f"{d['weight']:.2f}" for i, j, d in self.graph.edges(data=True)}
        #     )

# ==============================================================================
# UTILITIES

def construct_A_and_d(Pr, p_clip):
    n = Pr.shape[0]
    pairs, y = [], []
    for j in range(n):
        for i in range(j): 
            if not np.isnan(Pr[i, j]): pairs.append([i, j]); y.append(Pr[i, j])
    # Comparison matrix.
    A = np.zeros((len(pairs), n), dtype=int)
    for l, ij in enumerate(pairs): A[l, ij] = [1, -1] 
    # Target vector.
    d = norm.ppf(np.clip(y, p_clip, 1 - p_clip)) 
    return A, d

def train_val_split(Pr):
    """
    Split rating matrix into training and validation sets, while keeping comparison graph connected for training set.
    """
    # pairs = [(i, j) for i, j in np.argwhere(np.triu(np.invert(np.isnan(Pr))))]
    return Pr, None

def fitness_case_v(A, d):
    """
    Construct ranking under Thurstone's Case V model. 
    Uses Morrissey-Gulliksen least squares for incomplete comparison matrix.
    """
    return np.matmul(np.matmul(np.linalg.pinv(np.matmul(A.T, A)), A.T), d)

def labelling_loss(A, d, N, r, var, p_clip):
    """
    Loss function l that this algorithm is ultimately trying to minimise.
    """
    AN = np.matmul(A, N)
    F_diff = np.matmul(AN, r)
    F_std = np.sqrt(np.matmul(AN**2, var)) # Faster than actual matrix multiplication N A^T diag(var) A N^T.
    F_std[np.logical_and(F_diff == 0, F_std == 0)] = 1 # Catch 0 / 0 error.
    with np.errstate(divide="ignore"): 
        d_pred = norm.ppf(np.clip(norm.cdf(F_diff / F_std), p_clip, 1-p_clip)) # Clip to prevent infinite values.
        assert not np.isnan(d_pred).any()
    return ((d_pred - d)**2).mean()

def split_merge_cancel(split, merge):
    raise NotImplementedError("Still doesn't work")
    split.reverse()
    for m, (_, siblings, _) in enumerate(merge):
        split_cancel = set()
        subtractions_to_undo = 0
        for s, (_, parent_num, _, _, _) in enumerate(split): 
            if len(siblings) == 1: break
            if parent_num < siblings[-1]:
                siblings = siblings[:-1]
                if parent_num < siblings[0]: 
                    siblings = [siblings[0]-1] + siblings; subtractions_to_undo += 1
                else: 
                    split_cancel.add(s)
                    subtractions_to_undo = 0
                    for ss, (_, later_parent, _, _, _) in enumerate(split[:s]): 
                        if later_parent > siblings[0]: split[ss][1] -= 1 # ; split[ss][0] -= 1
        siblings = [sb+subtractions_to_undo for sb in siblings]
        split = [split[s] for s in range(len(split)) if s not in split_cancel]
        merge[m][1] = siblings
    split.reverse()
    merge = [m for m in merge if len(m[1]) > 1]

    print("====")
    print(split)
    print(merge)

    return split, merge
       
# split = [0,2,5,4,0,3,5,8,10,7]
# merge = [[3,4,5,6,7,8,9,10]]
# [[0, s, None, None, None] for s in split]
# [[None, m, None] for m in merge]

# split_merge_cancel(split, merge)

# ==============================================================================
# INTERFACES

class Interface:
    def __init__(self, pbrl): self.pbrl = pbrl
    def open(self): pass
    def close(self): pass

class VideoInterface(Interface):
    def __init__(self, pbrl): 
        Interface.__init__(self, pbrl)
        self.mapping = {81: 1., 83: 0., 32: 0.5, 27: "esc"}

    def open(self):
        self.videos = []
        for rn in self.pbrl.run_names:
            run_videos = sorted([f"video/{rn}/{f}" for f in os.listdir(f"video/{rn}") if ".mp4" in f])
            assert [int(v[-10:-4]) for v in run_videos] == list(range(len(run_videos)))
            self.videos += run_videos
        if len(self.videos) != len(self.pbrl.episodes): 
            assert len(self.videos) == len(self.pbrl.episodes) + 1
            print("Partial video found; ignoring.")                
        cv2.startWindowThread()
        cv2.namedWindow("Trajectory Pairs", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Trajectory Pairs", 1000, 500)

    def close(self): 
        cv2.destroyAllWindows()

    def __call__(self, i, j):
        vid_i = cv2.VideoCapture(self.videos[i])
        vid_j = cv2.VideoCapture(self.videos[j])
        while True:
            ret, frame1 = vid_i.read()
            if not ret: vid_i.set(cv2.CAP_PROP_POS_FRAMES, 0); _, frame1 = vid_i.read() # Will get ret = False at the end of the video, so reset.
            ret, frame2 = vid_j.read()
            if not ret: vid_j.set(cv2.CAP_PROP_POS_FRAMES, 0); _, frame2 = vid_j.read()
            if frame1 is None or frame2 is None: raise Exception("Video saving not finished!") 
            cv2.imshow("Trajectory Pairs", np.concatenate((frame1, frame2), axis=1))
            cv2.setWindowProperty("Trajectory Pairs", cv2.WND_PROP_TOPMOST, 1)
            key = cv2.waitKey(10) & 0xFF # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1.                        
            if key in self.mapping: break
        vid_i.release(); vid_j.release()
        return self.mapping[key]

class OracleInterface(Interface):
    def __init__(self, pbrl, oracle): 
        Interface.__init__(self, pbrl)
        self.oracle = oracle

    def __call__(self, i, j): 
        diff = (self.oracle[i] - self.oracle[j] if type(self.oracle) == list # Lookup 
                else self.oracle(self.pbrl.episodes[i], self.pbrl.episodes[j])) # Function
        if diff > 0: return 1. # Positive diff means i preferred.
        if diff < 1: return 0.
        return 0.5