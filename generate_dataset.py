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

        # if save_images:
        #     fig, ax = plt.subplots()
        #     fig.set_size_inches(3.5, 3.5)

        #     ax.imshow(gabor_kernel, cmap=my_cmap)
        #     ax.set_alpha(0.)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_axis_off()
        #     plt.margins(0.)

        #     fig.set_facecolor((0.85, 0.85, 0.85))

        #     plt.savefig('gabor/gabor_' + str(i) + '.png', facecolor=(0.85, 0.85, 0.85))
        #     plt.close()

    #### generate binary category labels ####
    # Task 1: classify along x1
    category_labels[np.isin(comp_labels, [1, 3]), 0] = 1

    # Task 2: classify along x2
    category_labels[np.isin(comp_labels, [2, 3]), 1] = 1

    category_labels = torch.from_numpy(category_labels).type(torch.long)

    return gabor_kernels, X, category_labels

def generate_stimuli_task_3(n, save_images=False):

    n_comps = 4

    # oversample by a factor of 4
    n_per_comp = int(n*4/n_comps)
    comp_labels = np.array([comp for comp in range(n_comps) for i in range(n_per_comp)], dtype=np.short)
    n_oversample = comp_labels.shape[0]

    category_labels = np.zeros((n, 1), dtype=np.short)

    X = np.zeros((n_oversample, 2))

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

    # keep n/2 trials from first component (the positive class) and n/2 from the remaining components
    # to ensure chance accuracy of 50%
    n_pos = int(n/2)
    n_neg = n - n_pos
    inds_pos = np.random.choice(np.arange(n_per_comp), n_pos, replace=False)
    inds_neg = np.random.choice(np.arange(n_per_comp, n_oversample), n_neg, replace=False)
    inds_subsample = np.concatenate((inds_pos, inds_neg))
    shuffle_inds = np.random.choice(inds_subsample, n, replace=False)
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
        
    #### generate binary category labels ####
    # Task 3: conjunction task
    # category_labels[np.isin(comp_labels, [1, 2, 3]), 0] = 1
    category_labels[np.isin(comp_labels, [0]), 0] = 1

    return gabor_kernels, X, category_labels

def generate_stimuli_task_4(n, save_images=False):
    n_comp_2 = int(n/2)
    n_comp_3 = n - n_comp_2

    category_labels = np.zeros((n, 1), dtype=np.short)
    X = np.zeros((n ,2))


    fmin = 0.03/2
    fmax = 0.10/2
    frange = fmax - fmin

    # centers of each component +/- the offset for each dimension
    c1, c2 = 0.25, 0.75
    gaussian_offset = 0.05

    sg1 = 0.06
    sg2 = 0.06
    rho = 0.9

    # component 2 - lower right quadrant of stim space
    sgma = np.array([
        [sg1**2, -rho*sg1*sg2],
        [-rho*sg1*sg2, sg2**2]
                    ])
    mu1 = np.array([c2-gaussian_offset, c1-gaussian_offset])
    mu2 = np.array([c2+gaussian_offset, c1+gaussian_offset])

    n1 = int(n_comp_2/2)
    n2 = n_comp_2 - n1

    x1 =np.random.multivariate_normal(mu1, sgma, size=n1)
    x2 = np.random.multivariate_normal(mu2, sgma, size=n2)
    x = np.concatenate((x1, x2), axis=0)
    x[x[:, 0] < 0.5, 0] = 0.5
    x[x[:, 0] > 1.0, 0] = 1.0
    x[x[:, 1] < 0, 1] = 0.0
    x[x[:, 1] > 0.5, 1] = 0.5
    X[:n_comp_2] = x

    # component 3 - upper left quadrant of stim space
    sgma = np.array([
        [sg1**2, -rho*sg1*sg2],
        [-rho*sg1*sg2, sg2**2]
                    ])
    mu1 = np.array([c1-gaussian_offset, c2-gaussian_offset])
    mu2 = np.array([c1+gaussian_offset, c2+gaussian_offset])

    n1 = int(n_comp_3/2)
    n2 = n_comp_3 - n1

    x1 =np.random.multivariate_normal(mu1, sgma, size=n1)
    x2 = np.random.multivariate_normal(mu2, sgma, size=n2)
    x = np.concatenate((x1, x2), axis=0)
    x[x[:, 0] < 0, 0] = 0.0
    x[x[:, 0] > 0.5, 0] = 0.5
    x[x[:, 1] < 0.5, 1] = 0.5
    x[x[:, 1] > 1, 1] = 1.0
    X[n_comp_2:] = x

    shuffle_inds = np.random.choice(np.arange(n), n, replace=False)
    X = X[shuffle_inds]

    # initialize array of gabor images
    ksize = 201
    gabor_kernels = np.zeros((n, ksize, ksize))
    for i in range(n):
        f = X[i, 0]*frange + fmin
        gabor1 = gabor_patch(0, 1/f, ksize, badvals=np.nan)
        gabor2 = gabor_patch(np.pi/2, 1/f, ksize, badvals=np.nan)
        gabor_kernels[i] = (1 - X[i, 1])*gabor1 + X[i, 1]*gabor2

    # Task 4: information integration combination (samples from component 1 only)
    # compute discrimination direction
    proj_1 = 1
    proj_2 = 1
    v_proj = np.array([proj_1, proj_2])
    v_proj = v_proj[:, np.newaxis]/np.linalg.norm(v_proj) # normalize to unit length
    x_proj = np.matmul(X, v_proj)
    x_proj_mid = np.matmul(np.array([0.5, 0.5]), v_proj)

    # category 1 is greater than the midpoint along the discrimination direction
    # otherwise, it is category 0
    category_labels[x_proj.squeeze() > x_proj_mid, 0] = 1

    return gabor_kernels, X, category_labels

def generate_dataset(root_dir):

    my_cmap = copy.copy(plt.cm.get_cmap('gray')) # get a copy of the gray color map
    my_cmap.set_bad(alpha=0) # set how the colormap handles 'bad' values

    dataset_size = {'train':int(512), 'val':int(512)}

    border_width = 10

    for phase in ['train', 'val']:
        all_images = []
        all_features = []
        all_targets = []
        all_task_ids = []

        # Task 1
        images, stim_features, targets = generate_stimuli(int(dataset_size[phase]/4))
        images = np.stack((images, ) * 3, axis=-1)
        background_pixels = np.isnan(images)
        images[background_pixels] = np.nan
        pixel_min = images[~background_pixels].min()
        pixel_max = images[~background_pixels].max()
        pixel_mean = images[~background_pixels].mean()

        # fill background with mean pixel value
        images[np.isnan(images)] = pixel_mean

        # normalize to integers in [0, 255]
        images = (images - pixel_min)/(pixel_max - pixel_min)
        images = (images*255).astype(np.uint8)

        # add colored border
        border_color = np.array([255, 0, 0])
        images_plus_border = np.zeros((images.shape[0], images.shape[1]+2*border_width, images.shape[2]+2*border_width, images.shape[3]), dtype=np.uint8)
        for channel in range(3):
            images_plus_border[:, :, :, channel] = border_color[channel]
        images_plus_border[:, border_width:-border_width, border_width:-border_width, :] = images
        all_images.append(images_plus_border)
        all_features.append(stim_features)
        all_targets.append(targets[:, 0, np.newaxis])
        all_task_ids.append(np.zeros(dataset_size[phase], dtype=np.short))

        # Task 2
        images, stim_features, targets = generate_stimuli(int(dataset_size[phase]/4))
        images = np.stack((images, ) * 3, axis=-1)
        background_pixels = np.isnan(images)
        images[background_pixels] = np.nan
        pixel_min = images[~background_pixels].min()
        pixel_max = images[~background_pixels].max()
        pixel_mean = images[~background_pixels].mean()

        # fill background with mean pixel value
        images[np.isnan(images)] = pixel_mean

        # normalize to integers in [0, 255]
        images = (images - pixel_min)/(pixel_max - pixel_min)
        images = (images*255).astype(np.uint8)

        # add colored border
        border_color = np.array([0, 255, 0])
        images_plus_border = np.zeros((images.shape[0], images.shape[1]+2*border_width, images.shape[2]+2*border_width, images.shape[3]), dtype=np.uint8)
        for channel in range(3):
            images_plus_border[:, :, :, channel] = border_color[channel]
        images_plus_border[:, border_width:-border_width, border_width:-border_width, :] = images
        all_images.append(images_plus_border)
        all_features.append(stim_features)
        all_targets.append(targets[:, 1, np.newaxis])
        all_task_ids.append(np.zeros(dataset_size[phase], dtype=np.short)+1)

        # Task 3
        images, stim_features, targets = generate_stimuli_task_3(int(dataset_size[phase]))
        images = np.stack((images, ) * 3, axis=-1)
        background_pixels = np.isnan(images)
        images[background_pixels] = np.nan
        pixel_min = images[~background_pixels].min()
        pixel_max = images[~background_pixels].max()
        pixel_mean = images[~background_pixels].mean()

        # fill background with mean pixel value
        images[np.isnan(images)] = pixel_mean

        # normalize to integers in [0, 255]
        images = (images - pixel_min)/(pixel_max - pixel_min)
        images = (images*255).astype(np.uint8)

        # add colored border
        border_color = np.array([0, 0, 255])
        images_plus_border = np.zeros((images.shape[0], images.shape[1]+2*border_width, images.shape[2]+2*border_width, images.shape[3]), dtype=np.uint8)
        for channel in range(3):
            images_plus_border[:, :, :, channel] = border_color[channel]
        images_plus_border[:, border_width:-border_width, border_width:-border_width, :] = images
        all_images.append(images_plus_border)
        all_features.append(stim_features)
        all_targets.append(targets)
        all_task_ids.append(np.zeros(dataset_size[phase], dtype=np.short)+2)

        # Task 4
        images, stim_features, targets = generate_stimuli_task_4(int(dataset_size[phase]))
        images = np.stack((images, ) * 3, axis=-1)
        background_pixels = np.isnan(images)
        images[background_pixels] = np.nan
        pixel_min = images[~background_pixels].min()
        pixel_max = images[~background_pixels].max()
        pixel_mean = images[~background_pixels].mean()

        # fill background with mean pixel value
        images[np.isnan(images)] = pixel_mean

        # normalize to integers in [0, 255]
        images = (images - pixel_min)/(pixel_max - pixel_min)
        images = (images*255).astype(np.uint8)

        # add colored border
        border_color = np.array([255, 255, 0])
        images_plus_border = np.zeros((images.shape[0], images.shape[1]+2*border_width, images.shape[2]+2*border_width, images.shape[3]), dtype=np.uint8)
        for channel in range(3):
            images_plus_border[:, :, :, channel] = border_color[channel]
        images_plus_border[:, border_width:-border_width, border_width:-border_width, :] = images
        all_images.append(images_plus_border)
        all_features.append(stim_features)
        all_targets.append(targets)
        all_task_ids.append(np.zeros(dataset_size[phase], dtype=np.short)+3)

        all_images = np.concatenate(all_images, axis=0)

        # not_nan = ~np.isnan(all_images)
        # pixel_min = all_images[not_nan].min()
        # pixel_max = all_images[not_nan].max()
        # pixel_mean = all_images[not_nan].mean()

        # # fill background with mean pixel value
        # all_images[np.isnan(all_images)] = pixel_mean

        # # normalize to integers in [0, 255]
        # all_images = (all_images - pixel_min)/(pixel_max - pixel_min)
        # all_images = (all_images*255).astype(np.uint8)

        # expand dims to (n, h, w, 3)
        # all_images = np.tile(all_images[:, :, :, np.newaxis], (1, 1, 1, 3))

        # resize from (201, 201) to (32, 32)
        # mytransform = Resize(32)
        # resized_images = np.zeros((n, 32, 32, 3))
        # for i in range(n):
        #     resized_images[i] = np.asarray(mytransform(Image.fromarray(all_images[i], 'RGB')))

        all_features = np.concatenate(all_features, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_task_ids = np.concatenate(all_task_ids, axis=0)

        # all_images = torch.from_numpy(all_images[:, np.newaxis, :, :]).expand(-1, 3, -1, -1).type(torch.float)
        # all_features = torch.from_numpy(all_features).type(torch.float)
        # all_targets = torch.from_numpy(all_targets).type(torch.long)
        # all_task_ids = torch.from_numpy(all_task_ids).type(torch.long)
        
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