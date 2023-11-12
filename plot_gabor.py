from __future__ import unicode_literals, print_function, division
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from utils import computeRunningAverage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_order = [0, 1, 2, 3]
results = torch.load(f'results/task-aware-gabor-continual-learning-dataset-v2-task-order-{task_order}_score_24.pt', map_location=device)
n = results['train_acc']['SCORE'].size(1)

# task_names = ['Frequency\nDiscrimination', 'Orientation\nDiscrimination', 'Conjunction', 'Information\nIntegration']
task_names = ['Frequency\nDiscrimination', 'Orientation\nDiscrimination', 'Color\nDiscrimination', 'Conjunction']
num_tasks = len(task_names)

fig=plt.figure()
fig.set_size_inches(14, 3)
ax1=fig.add_subplot(1, 1, 1)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plot_colors = {
    'SCORE':colors[3],
    'DynaMoE':colors[0],
    'EWC':colors[2],
    'Fine-Tune':colors[1],
    'Scratch':colors[4]
}
linewidth = 3

font = {'family' : 'sans-serif',
        'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
matplotlib.rc('legend', fontsize=20)

x_axis_scale = 'linear'
y_axis_scale = 'linear'

window_size = 20

plt.axvline(x=n/4, linestyle='--', color='gray', label='_nolegend_')
plt.axvline(x=n/4*2, linestyle='--', color='gray', label='_nolegend_')
plt.axvline(x=n/4*3, linestyle='--', color='gray', label='_nolegend_')

def add_curve(train_accs, alg, color):
    plot_acc = train_accs[alg].cpu()*100
    n_trials = plot_acc.shape[0]
    n = plot_acc.shape[1]
    task_boundaries = [int(n * task_id / 4) for task_id in range(5)]
    y_ave = torch.zeros((n_trials, n))
    for trial in range(n_trials):
        # y_ave[trial] = torch.tensor(ndi.gaussian_filter1d(plot_acc[trial], sigma=window_size))
        for task_id in range(len(task_boundaries)-1):
            task_start = task_boundaries[task_id]
            task_end = task_boundaries[task_id+1]
            y_ave[trial, task_start:task_end] = computeRunningAverage(plot_acc[trial, task_start:task_end], window_size=window_size)
    y = y_ave.mean(dim=0)
    x = [i for i in range(len(y))]
    sem = (y_ave.std(dim=0)/np.sqrt(n_trials))
    ax1.plot(x, y, color, linewidth=linewidth, label=alg)
    ax1.fill_between(x, (y-sem), (y+sem), color=color, alpha=.2, label='_nolegend_')

    for i, task_idx in enumerate(task_order):
        task_name = task_names[task_idx]
        ax1.annotate(task_name, (int(n/8*(2*i+1)), 101), ha='center', va='bottom')

    return

algs = ['SCORE', 'DynaMoE', 'EWC', 'Fine-Tune']
for alg in list(reversed(algs)):
    add_curve(results['train_acc'], alg, plot_colors[alg])

ax1.legend(algs, loc='lower left', bbox_to_anchor=(1.01, -0.1), frameon=False)
handles, labels = ax1.get_legend_handles_labels()
order = list(reversed(range(len(algs))))
ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower left', bbox_to_anchor=(1.01, -0.1), frameon=False)

ax1.set_xlabel('Trial')
ax1.set_ylabel(f'Sliding\nAccuracy (%)')
# ax1.set_title('Task 1')
ax1.set_xscale(x_axis_scale)
ax1.set_yscale(y_axis_scale)
ax1.set_xlim((0, n))
ax1.set_xticks([i*n/8 for i in range(9)])
ax1.set_yticks([0, 25, 50, 75, 100])

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.tight_layout()

plt.savefig(f'results/task-aware-gabor-continual-learning-dataset-v2-results-task-order-{task_order}_score_24.pdf')



fig, ax = plt.subplots()

fig.set_size_inches(11, 5.5)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

font = {'family' : 'sans-serif',
        'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 
matplotlib.rc('legend', fontsize=18)
matplotlib.rc('axes', titlesize=24)

linewidth = 4

x_axis_scale = 'linear'
y_axis_scale = 'linear'

algs = ['SCORE', 'DynaMoE', 'EWC', 'Fine-Tune']
algs = list(reversed(algs))
for alg in algs:
    n_trials = results['val_acc'][alg].size(0)
    y = results['val_acc'][alg].mean(dim=0)
    y_err = results['val_acc'][alg].std(dim=0)/np.sqrt(n_trials)
    x = [i+1 for i in range(len(y))]
    if alg == 'DynaMoE':
        label = alg + ' + Task Oracle'
    else:
        label = alg
    ax.plot(x, y, color=plot_colors[alg], linewidth=linewidth, label=label)

for alg in algs:
    n_trials = results['val_acc'][alg].size(0)
    y = results['val_acc'][alg].mean(dim=0)
    y_err = results['val_acc'][alg].std(dim=0)/np.sqrt(n_trials)
    x = [i+1 for i in range(len(y))]
    ax.errorbar(x, y, yerr=y_err, color=plot_colors[alg], linewidth=linewidth, label='_no_legend_')

ax.set_ylim([0.35, 1.02])
algs = [alg + ' + Task Oracle' if alg == 'DynaMoE' else alg for alg in algs]
ax.legend(algs, frameon=False, fontsize=18, loc='center left', bbox_to_anchor=[1.01, 0.5])
handles, labels = ax.get_legend_handles_labels()
order = list(reversed(range(len(algs))))
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], frameon=False, fontsize=16, loc='center left', bbox_to_anchor=[1.01, 0.5])

# ax.legend(algs, frameon=False, fontsize=18)
ax.set_xlabel('Task Name', fontsize=16)
ax.set_xticks(list(range(1, 5)))
ax.set_xticklabels(task_names, fontsize=16)
ax.set_ylabel('Accuracy', fontsize=16)
plt.title('Test Phase', fontsize=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.xaxis.set_tick_params(width=3)
ax.yaxis.set_tick_params(width=3)

plt.tight_layout()

save_path = f'results/task-aware-gabor-continual-learning-dataset-v2-test-phase-task-order-{task_order}_score_24'
plt.savefig(save_path + '.pdf')
# plt.savefig(save_path + '.png')