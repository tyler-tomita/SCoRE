import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import copy
import cv2
import pandas as pd
import seaborn as sb
import sys

my_cmap = copy.copy(plt.cm.get_cmap('gray')) # get a copy of the gray color map
my_cmap.set_bad(alpha=0) # set how the colormap handles 'bad' values

def gabor_patch(theta, lambd, ksize, badvals=np.nan):
    # theta is in radians
#     fig, ax = plt.subplots()
#     fig.set_size_inches(2, 2)
#     plt.gray()
    sigma = ksize  # Larger Values produce more edges
    gamma = 1.0
    psi = 0  # Offset value - lower generates cleaner results
    gk = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
    r = int((ksize-1)/2)
    for x in range(ksize):
        for y in range(ksize):
            rel_dist = (x - r)**2 + (y - r)**2
            if rel_dist > (r**2):
                gk[x, y] = badvals

    return gk


def generate_stimuli(n_per_comp, root_dir=None):

    ##### distribution of stimuli #####
    # 2 dimensions:
    #
    # x1 is the frequency of gabors \in [0.05, 0.15], normalized to [0, 1]
    #
    # x2 is the weight for a weighted average of gabors such that
    # stimulus = (1-x2)*gabor(0, x1) + x2*gabor(pi/2, x1),
    # where gabor(theta, f) is the gabor patch with orientation theta and frequency f
    #
    # there are four components, with components located closest to one of the four quadrants

    n_comps = 4
    comp_labels = np.array([comp for comp in range(n_comps) for i in range(n_per_comp)], dtype=np.short)
    n = comp_labels.shape[0]

    n_tasks = 2
    category_labels = np.zeros((n, n_tasks), dtype=np.short)
    X = np.zeros((n, 2))

    n1 = int(n_per_comp/2)
    n2 = n_per_comp - n1

    fmin = 0.03/2
    fmax = 0.10/2
    frange = fmax - fmin

    # centers of each component +/- the offset for each dimension
    c1, c2 = 0.25, 0.75
    gaussian_offset = 0.05

    sg1 = 0.06
    sg2 = 0.06
    rho = 0.9

    # component 1 - lower left quadrant of feature space
    sgma = np.array([
        [sg1**2, rho*sg1*sg2],
        [rho*sg1*sg2, sg2**2]
                    ])
    mu1 = np.array([c1+gaussian_offset, c1-gaussian_offset])
    mu2 = np.array([c1-gaussian_offset, c1+gaussian_offset])

    x1 =np.random.multivariate_normal(mu1, sgma, size=n1)
    x2 = np.random.multivariate_normal(mu2, sgma, size=n2)
    x = np.concatenate((x1, x2), axis=0)
    x[x < 0] = 0.0
    x[x > 0.5] = 0.5
    X[:n_per_comp] = x

    # component 2 - lower right quadrant of stim space
    sgma = np.array([
        [sg1**2, -rho*sg1*sg2],
        [-rho*sg1*sg2, sg2**2]
                    ])
    mu1 = np.array([c2-gaussian_offset, c1-gaussian_offset])
    mu2 = np.array([c2+gaussian_offset, c1+gaussian_offset])

    x1 =np.random.multivariate_normal(mu1, sgma, size=n1)
    x2 = np.random.multivariate_normal(mu2, sgma, size=n2)
    x = np.concatenate((x1, x2), axis=0)
    x[x[:, 0] < 0.5, 0] = 0.5
    x[x[:, 0] > 1.0, 0] = 1.0
    x[x[:, 1] < 0, 1] = 0.0
    x[x[:, 1] > 0.5, 1] = 0.5
    X[n_per_comp:(2*n_per_comp)] = x

    # component 3 - upper left quadrant of stim space
    sgma = np.array([
        [sg1**2, -rho*sg1*sg2],
        [-rho*sg1*sg2, sg2**2]
                    ])
    mu1 = np.array([c1-gaussian_offset, c2-gaussian_offset])
    mu2 = np.array([c1+gaussian_offset, c2+gaussian_offset])

    x1 =np.random.multivariate_normal(mu1, sgma, size=n1)
    x2 = np.random.multivariate_normal(mu2, sgma, size=n2)
    x = np.concatenate((x1, x2), axis=0)
    x[x[:, 0] < 0, 0] = 0.0
    x[x[:, 0] > 0.5, 0] = 0.5
    x[x[:, 1] < 0.5, 1] = 0.5
    x[x[:, 1] > 1, 1] = 1.0
    X[(2*n_per_comp):(3*n_per_comp)] = x

    # component 4 - upper right quadrant of stim space
    sgma = np.array([
        [sg1**2, rho*sg1*sg2],
        [rho*sg1*sg2, sg2**2]
                    ])
    mu1 = np.array([c2+gaussian_offset, c2-gaussian_offset])
    mu2 = np.array([c2-gaussian_offset, c2+gaussian_offset])

    x1 =np.random.multivariate_normal(mu1, sgma, size=n1)
    x2 = np.random.multivariate_normal(mu2, sgma, size=n2)
    x = np.concatenate((x1, x2), axis=0)
    x[x[:, 0] < 0.5, 0] = 0.5
    x[x[:, 0] > 1, 0] = 1.0
    x[x[:, 1] < 0.5, 1] = 0.5
    x[x[:, 1] > 1, 1] = 1.0
    X[(3*n_per_comp):(4*n_per_comp)] = x

    shuffle_inds = np.random.choice(np.arange(n), n, replace=False)
    X = X[shuffle_inds]
    comp_labels = comp_labels[shuffle_inds]


    # initialize array of gabor images
    ksize = 201
    gabor_kernels = np.zeros((n, ksize, ksize))
    for i in range(n):
        f = X[i, 0]*frange + fmin
        gabor1 = gabor_patch(0, 1/f, ksize, badvals=np.nan)
        gabor2 = gabor_patch(np.pi/2, 1/f, ksize, badvals=np.nan)
        gabor_kernels[i] = (1 - X[i, 1])*gabor1 + X[i, 1]*gabor2

        Xpad_horizontal = np.empty((ksize, int(ksize/4)))
        Xpad_horizontal[:] = np.nan
        Xpad_vertical = np.empty((int(ksize/4), ksize+2*int(ksize/4)))
        Xpad_vertical[:] = np.nan
        Xcat = np.concatenate(
            (
                Xpad_vertical,
                np.concatenate((Xpad_horizontal, gabor_kernels[i], Xpad_horizontal), axis=1),
                Xpad_vertical
            ),
            axis=0
        )

        if root_dir:
            fig, ax = plt.subplots()
            fig.set_size_inches(3.5, 3.5)

            ax.imshow(Xcat, cmap=my_cmap)
            ax.set_alpha(0.)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()
            plt.margins(0.)

            fig.set_facecolor((0.85, 0.85, 0.85))

            plt.savefig(f'{root_dir}/stim_' + str(i) + '.png', facecolor=(0.85, 0.85, 0.85))
            plt.close()

    np.savetxt(f'{root_dir}/features.csv', X, delimiter=',', fmt ='%.5f')

    #### generate binary category labels ####
    # Task 1: classify along x1
    category_labels[np.isin(comp_labels, [1, 3]), 0] = 1

    # Task 2: classify along x2
    category_labels[np.isin(comp_labels, [2, 3]), 1] = 1

    np.savetxt(f'{root_dir}/targets.csv', category_labels, delimiter=',', fmt ='%.0f')

    return

if __name__ == '__main__':
    num_stims = int(sys.argv[1])
    root_dir = sys.argv[2]
    generate_stimuli(int(num_stims/4), root_dir=root_dir)
