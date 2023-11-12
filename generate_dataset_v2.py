from __future__ import unicode_literals, print_function, division
import torch
import cv2
from io import open
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
from torchvision.transforms import Resize, ToTensor
from PIL import Image

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

def generate_stimuli(n_per_comp, save_images=False):

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

    n_tasks = 4
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
    X[:] = X[shuffle_inds]
    comp_labels = comp_labels[shuffle_inds]


    # initialize array of gabor images
    ksize = 201
    gabor_kernels = np.zeros((n, ksize, ksize))
    for i in range(n):
        f = X[i, 0]*frange + fmin
        gabor1 = gabor_patch(0, 1/f, ksize, badvals=np.nan)
        gabor2 = gabor_patch(np.pi/2, 1/f, ksize, badvals=np.nan)
        gabor_kernels[i] = (1 - X[i, 1])*gabor1 + X[i, 1]*gabor2

    gabor_kernels = np.stack((gabor_kernels, ) * 3, axis=-1)
    background_pixels = np.isnan(gabor_kernels)
    gabor_kernels[background_pixels] = np.nan
    pixel_min = gabor_kernels[~background_pixels].min()
    pixel_max = gabor_kernels[~background_pixels].max()
    pixel_mean = gabor_kernels[~background_pixels].mean()

    # fill background with mean pixel value
    gabor_kernels[np.isnan(gabor_kernels)] = pixel_mean

    # normalize to integers in [0, 255]
    gabor_kernels = (gabor_kernels - pixel_min)/(pixel_max - pixel_min)
    gabor_kernels = (gabor_kernels*255).astype(np.uint8)

    # add colored border in two clusters (for color classification)
    # border_color = np.array([255, 0, 0])
    border_width = 10
    border_color = np.zeros((n, 3))
    npos = int(n/2)
    nneg = n - npos
    border_color[:npos, 1] = np.random.randint(low=128, high=256, size=npos)
    border_color[npos:, 2] = np.random.randint(low=128, high=256, size=nneg)
    shuffle_inds = np.random.choice(np.arange(n), n, replace=False)
    border_color[:] = border_color[shuffle_inds]
    border_color_expanded = border_color.reshape((n, 1, 1, 3)).repeat(gabor_kernels.shape[1]+2*border_width, axis=1).repeat(gabor_kernels.shape[2]+2*border_width, axis=2)
    gabor_kernels_plus_border = np.zeros((gabor_kernels.shape[0], gabor_kernels.shape[1]+2*border_width, gabor_kernels.shape[2]+2*border_width, gabor_kernels.shape[3]), dtype=np.uint8)
    for channel in range(3):
        gabor_kernels_plus_border[:, :, :, channel] = border_color_expanded[:, :, :, channel]
    gabor_kernels_plus_border[:, border_width:-border_width, border_width:-border_width, :] = gabor_kernels

    #### generate binary category labels ####
    # Task 1: classify along x1
    category_labels[np.isin(comp_labels, [1, 3]), 0] = 1

    # Task 2: classify along x2
    category_labels[np.isin(comp_labels, [2, 3]), 1] = 1

    # Task 3: classify along color border of image
    category_labels[border_color[:, 1] > 0., 2] = 1

    # Task 4: classify along conjunction of x1 and x2
    category_labels[np.isin(comp_labels, [3]), 3] = 1

    category_labels = torch.from_numpy(category_labels).type(torch.long)

    X = np.concatenate((X, border_color), axis=1)

    return gabor_kernels_plus_border, X, category_labels


def generate_dataset_v2(root_dir):

    my_cmap = copy.copy(plt.cm.get_cmap('gray')) # get a copy of the gray color map
    my_cmap.set_bad(alpha=0) # set how the colormap handles 'bad' values

    dataset_size = {'train':int(512), 'val':int(512)}


    for phase in ['train', 'val']:
        all_images = []
        all_features = []
        all_targets = []
        all_task_ids = []

        images, stim_features, targets = generate_stimuli(int(dataset_size[phase]/4))
        
        for task_id in range(4):
            all_images.append(images)
            all_features.append(stim_features)
            all_targets.append(targets[:, task_id, np.newaxis])
            all_task_ids.append(np.zeros(dataset_size[phase], dtype=np.short)+task_id)

        all_images = np.concatenate(all_images, axis=0)         
        all_features = np.concatenate(all_features, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_task_ids = np.concatenate(all_task_ids, axis=0)

        n = all_images.shape[0]
        image_path = f'{root_dir}/{phase}/images.pkl'

        with open (image_path, "wb") as f:
            for i in range(n):
                pickle.dump(all_images[i], f)
                # fig, ax = plt.subplots()
                # fig.set_size_inches(3.5, 3.5)

                # ax.imshow(all_images[i], cmap=my_cmap)
                # ax.set_alpha(0.)
                # ax.set_xticks([])
                # ax.set_yticks([])
                # ax.set_axis_off()
                # plt.margins(0.)

                # fig.set_facecolor((0.85, 0.85, 0.85))


                # plt.savefig(image_path, facecolor=(0.85, 0.85, 0.85))
                # plt.close()

        f.close()

        np.savetxt(f'{root_dir}/{phase}/features.csv', all_features, delimiter=',', fmt ='%.5f')
        np.savetxt(f'{root_dir}/{phase}/targets.csv', all_targets, delimiter=',', fmt ='%.0f')
        np.savetxt(f'{root_dir}/{phase}/task_ids.csv', all_task_ids, delimiter=',', fmt ='%.0f')