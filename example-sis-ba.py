#!/usr/bin/env python
# coding: utf-8

# # Example of dynamics learning of SIS on BA networks.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In this notebook, we show an example of dynamics learning of SIS dynamics on Barabasi-Albert networks. We first start by gathering the configuration of the experiment.

# In[ ]:


from dynalearn.config import ExperimentConfig

config = ExperimentConfig.default(
    "example-sis-cave3-net", 
    "sis", 
    "mod_net", 
    path_to_data="./examples/data", 
    path_to_summary="./examples/summaries", 
    path_to_best="./examples/best", 
    seed=0
)
config.metrics.names = ("TrueLTPMetrics","GNNLTPMetrics","MLELTPMetrics")
config.train_details.num_samples = 1000
config.train_details.epochs = 10
config.dataset.use_groundtruth = True
config.dynamics.is_modular = True


# Then, we define the experiment from this configuration.
# 

# In[3]:




from dynalearn.experiments import Experiment
exp = Experiment(config, verbose=2)


# Finally, we can run the experiment. We must perform certain tasks before the experiment is completed: 1) we generate the training and validation datasets, 2) we train the model using these datasets and finally 3) we compute the transition probabilities computed by the trained model.

# In[4]:


exp.run(["generate_data", "partition_val_dataset", "train_model", "compute_metrics"])


# In[ ]:


import sys

from dynalearn.experiments import LTPMetrics
from dynalearn.util.display import *
transitions = [(0, 1), (1, 0)]
colors = {
    "true": [color_pale["blue"], color_pale["red"]],
    "gnn": [color_dark["blue"], color_dark["red"]],
    "mle": [color_dark["blue"], color_dark["red"]],
}
linestyles = {
    "true": ["-", "-"],
    "gnn": ["--", "--"],
    "mle": ["None", "None"],
}
markers = {
    "true": ["None", "None"],
    "gnn": ["None", "None"],
    "mle": ["o", "^"],
}

def plot_ltp(experiment, ax):
    summary = experiment.metrics["TrueLTPMetrics"].data["summaries"]
    true_ltp = experiment.metrics["TrueLTPMetrics"].data["ltp"]
    gnn_ltp = experiment.metrics["GNNLTPMetrics"].data["ltp"]
    mle_ltp = experiment.metrics["MLELTPMetrics"].data["ltp"]
    agg = lambda ltp, in_s, out_s: LTPMetrics.aggregate(
            ltp, summary, 
            in_state=in_s, 
            out_state=out_s,
            axis=1, 
            reduce="mean", 
            err_reduce="percentile"
        )
    x_min, x_max = -np.inf, np.inf
    for i, (in_s, out_s) in enumerate(transitions):
        x, y, yl, yh = agg(true_ltp, in_s, out_s)
        ax.plot(
            x, y, color=colors["true"][i], linestyle=linestyles["true"][i],marker=markers["true"][i],linewidth=3
        )
        ax.fill_between(x, yl, yh, color=colors["true"][i], alpha=0.3)
        
        x, y, yl, yh = agg(gnn_ltp, in_s, out_s)
        ax.plot(
            x, y, color=colors["gnn"][i], linestyle=linestyles["gnn"][i],marker=markers["gnn"][i],linewidth=3
        )
        ax.fill_between(x, yl, yh, color=colors["gnn"][i], alpha=0.3)
        
        x, y, yl, yh = agg(mle_ltp, in_s, out_s)
        yerr = np.concatenate([np.expand_dims(y-yl,0), np.expand_dims(yh-y,0)], axis=0)
        ax.errorbar(
            x, 
            y, 
            yerr=yerr,
            color=colors["mle"][i], 
            linestyle=linestyles["mle"][i], 
            marker=markers["mle"][i], 
            alpha=0.3
        )
#         ax.plot(
#             x, y, color=colors["mle"][i], linestyle=linestyles["mle"][i], marker=markers["mle"][i], alpha=0.5
#         )
#         ax.fill_between(x, yl, yh, color=colors["mle"][i], alpha=0.3)
        
        if x.min() > x_min:
            x_min = x.min()
        if x.max() < x_max:
            x_max = x.max()
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([0, 1.1])
    return ax

fig, ax = plt.subplots(1, 1, figsize=(12, 12))

plot_ltp(exp, ax)
ax.set_xlabel("Number of infected neighbors", fontsize=18)
ax.set_ylabel("Transition probability", fontsize=18)
handles = []

handles.append(Line2D([-1], [-1], linestyle="-", marker="None", linewidth=3,
                     color=color_pale["grey"], 
                     label=r"True")
             )
handles.append(Line2D([-1], [-1], linestyle="--", marker="None", linewidth=3,
                     color=color_dark["grey"], 
                     label=r"GNN")
             )
handles.append((Line2D([-1], [-1], linestyle="None", marker="o", markersize=5, markeredgewidth=1,
                      markeredgecolor='k', color=color_dark["grey"], alpha=0.3),
                Line2D([-1], [-1], linestyle="None", marker="^", markersize=5, markeredgewidth=1,
                      markeredgecolor='k', color=color_dark["grey"], alpha=0.3))
             )
handles.append(Line2D([-1], [-1], linestyle="None", marker="s", markersize=12,
                     color=color_pale["blue"])
             )
handles.append(Line2D([-1], [-1], linestyle="None", marker="s", markersize=12,
                     color=color_pale["red"])
             )
ax.legend(handles=handles, 
             labels=[r"GT", r"GNN", r"MLE", "Infection", "Recovery"],
             handler_map={tuple: HandlerTuple(ndivide=None)},
             loc="upper right", fancybox=True, fontsize=14, framealpha=0.75, ncol=1
)
plt.show()


# In[ ]:




