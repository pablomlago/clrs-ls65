# Copyright 2021 V. V. Mirjanic. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Generates Trajectory-wise and Step-wise plots from captured trajectories.
These plots are used to make Figure 1 from the paper.
"""

import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from typing import List


PIXEL_S = (72. / 300) ** 2
common_scatter_args = lambda s: dict(marker='o', s=s, edgecolor='none')
common_plot_args = dict(color='maroon', alpha=0.6, lw=0.1)


def get_pca_evr(pca, d):
  return 100 * pca.explained_variance_ratio_[:d].sum()


def plot_heatmap_trajwise_old(data, name):
  samples, mp_steps, dim = data.shape

  trajwise_pca = PCA(n_components=10)
  trajwise = trajwise_pca.fit_transform(data.reshape((samples, mp_steps * dim)))
  assert trajwise.shape == (samples, 10)

  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)

  THRESHOLD = np.mean(np.max(np.abs(trajwise_pca.components_), axis=0))  # 0.05
  relevant_components = trajwise_pca.components_[:, np.max(np.abs(trajwise_pca.components_), axis=0) > THRESHOLD]
  ax.imshow(np.abs(relevant_components), cmap='inferno')

  fig.suptitle(f"Trajectory-wise PCA components heatmap")
  fig.tight_layout()
  fig.savefig(f"{name}_heatmap.png", dpi=300)
  plt.close()


def plot_heatmap_trajwise(data, path="./"):
  samples, mp_steps, dim = data.shape

  # standard_scaler = StandardScaler()
  data = data.reshape(samples, mp_steps * dim)
  # data = standard_scaler.fit_transform(data)

  trajwise_pca = PCA()
  trajwise_pca.fit(data)

  fig = plt.figure(figsize=(12, 6))
  ax = fig.add_subplot(111)

  contributed_variance = trajwise_pca.explained_variance_[...,np.newaxis] * np.abs(trajwise_pca.components_)
  contributed_variance = np.sum(contributed_variance, axis=0)
  contributed_variance = np.reshape(contributed_variance, (mp_steps, dim))
  contributed_variance = np.sum(contributed_variance, axis=0)
  contributed_variance = contributed_variance / np.max(contributed_variance)
  ax.bar(np.arange(dim), contributed_variance)

  fig.suptitle(f"Trajectory-wise PCA components heatmap")
  fig.tight_layout()
  fig.savefig(f"{path}/trajwise_heatmap.png", dpi=300)
  plt.close()

def plot_trajwise(data, score, path ="./", prefix = 'default'):
  samples, mp_steps, dim = data.shape

  standard_scaler = StandardScaler()
  data = data.reshape(samples, mp_steps * dim)
  data = standard_scaler.fit_transform(data)

  trajwise_pca = PCA()
  trajwise = trajwise_pca.fit_transform(data)
  assert trajwise.shape == (samples, min(samples, mp_steps * dim))

  # Plot trajectories
  fig = plt.figure(figsize=(12, 6))
  ax = [fig.add_subplot(121), fig.add_subplot(122, projection='3d')]

  ax[0].scatter(trajwise[:, 0], trajwise[:, 1], c=score, vmin=0, vmax=1, cmap='viridis_r',
                **common_scatter_args(16 * PIXEL_S))
  ax[1].scatter3D(trajwise[:, 0], trajwise[:, 1], trajwise[:, 2], c=score, vmin=0, vmax=1, cmap='viridis_r',
                  **common_scatter_args(16 * PIXEL_S))

  fig.suptitle(f"Trajectory-wise PCA (shape = samples, mp_steps*dim)\n "
               f"{get_pca_evr(trajwise_pca, 3):.2f}% explained")
  fig.tight_layout()
  fig.savefig(f"{path}/{prefix}_trajwise.png", dpi=300)
  plt.close()


def plot_stepwise_global(data: np.ndarray, paths_drawn: int, sample_len: np.ndarray, path="./", prefix = 'default'):

  samples, mp_steps, dim = data.shape

  standard_scaler = StandardScaler()
  data = data.reshape((samples * mp_steps, dim))
  data = standard_scaler.fit_transform(data)

  pca = PCA()
  pca.fit(data)
  stepwise_global = pca.transform(data)
  stepwise_global: np.ndarray
  stepwise_global = stepwise_global.reshape((samples, mp_steps, -1))
  assert stepwise_global.shape == (samples, mp_steps, min(samples * mp_steps, dim))

  # Plot all hidden dims as independent datapoints
  fig = plt.figure(figsize=(12, 6))
  ax = [fig.add_subplot(121), fig.add_subplot(122, projection='3d')]

  for step in range(mp_steps):
    ax[0].scatter(stepwise_global[sample_len >= step, step, 0],
                  stepwise_global[sample_len >= step, step, 1],
                  label=f'mp step {step + 1}', **common_scatter_args(PIXEL_S))
    ax[1].scatter3D(stepwise_global[sample_len >= step, step, 0],
                    stepwise_global[sample_len >= step, step, 1],
                    stepwise_global[sample_len >= step, step, 2],
                    label=f'mp step {step + 1}', **common_scatter_args(PIXEL_S))

  if paths_drawn > 0:
    for sample in np.arange(0, samples, step=int(samples / paths_drawn)):
      ax[0].plot(stepwise_global[sample, :, 0],
                 stepwise_global[sample, :, 1],
                 **common_plot_args)
      ax[1].plot3D(stepwise_global[sample, :, 0],
                   stepwise_global[sample, :, 1],
                   stepwise_global[sample, :, 2],
                   **common_plot_args)
  zero = pca.transform(np.zeros((1, dim))).reshape(-1)
  ax[0].plot(zero[0], zero[1], 'rx')

  fig.suptitle(f"Pointwise PCA (shape = samples*mp_steps, dim)\n "
               f"{get_pca_evr(pca, 3):.2f}% explained")

  fig.tight_layout()
  fig.savefig(f"{path}/{prefix}_stepwise_global.png", dpi=300)
  plt.close()


def plot_stepwise_local(data, paths_drawn, sample_len, path="./", prefix='default'):

  samples, mp_steps, dim = data.shape

  standard_scalers = [StandardScaler().fit(data[sample_len >= step, step, :]) for step in range(mp_steps)]
  data_steps = [standard_scalers[step].transform(data[:, step, :]) for step in range(mp_steps)]
  # data = data.reshape(-1, mp_steps * dim)
  # standard_scaler = StandardScaler()
  # data = standard_scaler.fit_transform(data)
  # data = data.reshape(samples, mp_steps, dim)

  pcas = [PCA(n_components=3).fit(data_steps[step][sample_len >= step, :]) for step in range(mp_steps)]
  stepwise_local = np.array([pcas[step].transform(data_steps[step]) for step in range(mp_steps)])
  assert stepwise_local.shape == (mp_steps, samples, 3)

  SCALE_FACTOR = 2 + mp_steps
  trajs_PCA_scale_factor = 1 / np.max(np.abs(stepwise_local[:, :, 2]), axis=-1) / SCALE_FACTOR

  fig, ax = plt.subplots(figsize=(15, 6), subplot_kw=dict(projection=f'3d'))
  ax.view_init(azim=-75, elev=15)
  ax.set_box_aspect(aspect=(2.5, 1, 1))
  for i in range(mp_steps):
    ax.scatter(i + stepwise_local[i, sample_len >= i, 2] * trajs_PCA_scale_factor[i],
               stepwise_local[i, sample_len >= i, 0], stepwise_local[i, sample_len >= i, 1], **common_scatter_args(PIXEL_S))
  if paths_drawn > 0:
    for i in np.arange(0, samples, step=int(samples / paths_drawn)):
      ax.plot(np.arange(sample_len[i]) + stepwise_local[0:sample_len[i], i, 2] * trajs_PCA_scale_factor[0:sample_len[i]],
              stepwise_local[0:sample_len[i], i, 0], stepwise_local[0:sample_len[i], i, 1],
              **common_plot_args)

  fig.tight_layout()
  fig.savefig(f"{path}/{prefix}_stepwise_local.png", dpi=300)
  plt.close()

def plot_stepwise_local_asynchrony(data, paths_drawn, sample_len, path="./", prefix='default'):

  samples, mp_steps, dim = data[0].shape

  data_1, data_2 = data

  # data = data.reshape(-1, mp_steps * dim)
  # standard_scaler = StandardScaler()
  # data = standard_scaler.fit_transform(data)
  # data = data.reshape(samples, mp_steps, dim)

  pcas = [PCA(n_components=3).fit(np.concatenate([data_1[sample_len >= step, step, :], data_2[sample_len >= step, step, :]], axis=0)) for step in range(mp_steps)]

  data = data_1

  stepwise_local = np.array([pcas[step].transform(data[:, step, :]) for step in range(mp_steps)])
  assert stepwise_local.shape == (mp_steps, samples, 3)

  SCALE_FACTOR = 2 + mp_steps
  trajs_PCA_scale_factor = 1 / np.max(np.abs(stepwise_local[:, :, 2]), axis=-1) / SCALE_FACTOR

  fig, ax = plt.subplots(figsize=(15, 6), subplot_kw=dict(projection=f'3d'))
  ax.view_init(azim=-75, elev=15)
  ax.set_box_aspect(aspect=(2.5, 1, 1))
  for i in range(mp_steps):
    ax.scatter(i + stepwise_local[i, sample_len >= i, 2] * trajs_PCA_scale_factor[i],
               stepwise_local[i, sample_len >= i, 0], stepwise_local[i, sample_len >= i, 1], c='red', marker='o', s=PIXEL_S, edgecolor='none')
  if paths_drawn > 0:
    for i in np.arange(0, samples, step=int(samples / paths_drawn)):
      ax.plot(np.arange(sample_len[i]) + stepwise_local[0:sample_len[i], i, 2] * trajs_PCA_scale_factor[0:sample_len[i]],
              stepwise_local[0:sample_len[i], i, 0], stepwise_local[0:sample_len[i], i, 1],
              color='orange', alpha=0.6, lw=0.1)
      
  data = data_2

  stepwise_local = np.array([pcas[step].transform(data[:, step, :]) for step in range(mp_steps)])
  assert stepwise_local.shape == (mp_steps, samples, 3)

  SCALE_FACTOR = 2 + mp_steps
  trajs_PCA_scale_factor = 1 / np.max(np.abs(stepwise_local[:, :, 2]), axis=-1) / SCALE_FACTOR

  for i in range(mp_steps):
    ax.scatter(i + stepwise_local[i, sample_len >= i, 2] * trajs_PCA_scale_factor[i],
               stepwise_local[i, sample_len >= i, 0], stepwise_local[i, sample_len >= i, 1], c='blue', marker='o', s=PIXEL_S, edgecolor='none')
  if paths_drawn > 0:
    for i in np.arange(0, samples, step=int(samples / paths_drawn)):
      ax.plot(np.arange(sample_len[i]) + stepwise_local[0:sample_len[i], i, 2] * trajs_PCA_scale_factor[0:sample_len[i]],
              stepwise_local[0:sample_len[i], i, 0], stepwise_local[0:sample_len[i], i, 1],
              color='green', alpha=0.6, lw=0.1)

  fig.tight_layout()
  plt.locator_params(axis='y', nbins=5)
  fig.savefig(f"{path}/{prefix}_stepwise_local_embeddings.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
  plt.close()

def run_experiment(path: str, paths_drawn=100):

  data_dump = np.load(path + 'trajs.npz')
  data = data_dump['trajs']
  score = data_dump['score']
  l2_node_updates_partial = data_dump['l2_node_updates_partial']
  l2_node_updates_aggregated = data_dump['l2_node_updates_aggregated']
  true_lengths = data_dump['lengths']

  l3_cocycle_args_update_aggregated=data_dump['l3_cocycle_args_update_aggregated']
  l3_cocycle_args_update_aggregated_partial=data_dump['l3_cocycle_args_update_aggregated_partial']
  l3_multimorphism_msgs_aggregated=data_dump['l3_multimorphism_msgs_aggregated']
  l3_multimorphism_msgs_partial=data_dump['l3_multimorphism_msgs_partial']

  try:
    os.mkdir(f'{path}/pca_plots')
  except OSError:
    pass

  path = f'{path}/pca_plots'

  max_length = 0
  max_length_i = 0
  for i in range(1,20):
    length_i = np.sum(true_lengths == i)
    if length_i > max_length:
      max_length = length_i
      max_length_i = i
  # print([np.sum(true_lengths == i) for i in range(16)])

  data = data[true_lengths == max_length_i, :max_length_i, :]
  score = score[true_lengths == max_length_i]
  l2_node_updates_aggregated = l2_node_updates_aggregated[true_lengths == max_length_i, :max_length_i, :]
  l2_node_updates_partial = l2_node_updates_partial[true_lengths == max_length_i, :max_length_i, :]

  l3_cocycle_args_update_aggregated=l3_cocycle_args_update_aggregated[true_lengths == max_length_i, :max_length_i, :]
  l3_cocycle_args_update_aggregated_partial=l3_cocycle_args_update_aggregated_partial[true_lengths == max_length_i, :max_length_i, :]
  l3_multimorphism_msgs_aggregated=l3_multimorphism_msgs_aggregated[true_lengths == max_length_i, :max_length_i, :]
  l3_multimorphism_msgs_partial=l3_multimorphism_msgs_partial[true_lengths == max_length_i, :max_length_i, :]

  true_lengths = true_lengths[true_lengths == max_length_i]  

  # means = np.mean(data, axis=0)
  # mean_adjusted_data = data - means[np.newaxis, ...]
  # plot_stepwise_global(mean_adjusted_data, paths_drawn, true_lengths, name, 'mean')
  # plot_stepwise_local(mean_adjusted_data, paths_drawn, true_lengths, name, 'mean')
  #
  plot_trajwise(data, score, path)
  alt_score = np.linspace(0, 1, data.shape[0])
  plot_trajwise(data, alt_score, path, 'alt')
  plot_heatmap_trajwise(data, path)

  # Plot differences instead of points
  diffs = data[:,1:,:] - data[:,:-1,:]
  plot_stepwise_global(diffs, paths_drawn, true_lengths - 1, path, 'diffs')
  plot_stepwise_local(diffs, paths_drawn, true_lengths - 1, path, 'diffs')

  plot_stepwise_global(data, paths_drawn, true_lengths, path)
  plot_stepwise_local(data, paths_drawn, true_lengths, path)
  plot_stepwise_local_asynchrony([l2_node_updates_aggregated, l2_node_updates_partial], paths_drawn, true_lengths, path, 'l2')
  plot_stepwise_local_asynchrony([l3_cocycle_args_update_aggregated, l3_cocycle_args_update_aggregated_partial], paths_drawn, true_lengths, path, 'l3_cocycle')
  plot_stepwise_local_asynchrony([l3_multimorphism_msgs_aggregated, l3_multimorphism_msgs_partial], paths_drawn, true_lengths, path, 'l3_multimorphism')

  return
  a = 4
  b = 10
  plot_stepwise_global(data[true_lengths == b, a:b, :], 20, true_lengths[true_lengths == b] - a, name, 'reduced')
  plot_stepwise_local(data[true_lengths == b, a:b, :], 20, true_lengths[true_lengths == b] - a, name, 'reduced')



def traj_step(data: np.ndarray,
              score: np.ndarray,
              paths_drawn: int,
              sample_len: np.ndarray,
              name: str):

  acc = np.mean(score)

  if len(data.shape) == 4:
    data = np.max(data, axis=1)

  samples, mp_steps, dim = data.shape

  trajwise_pca = PCA()
  trajwise = trajwise_pca.fit_transform(data.reshape((samples, mp_steps * dim)))
  assert trajwise.shape == (samples, min(samples, mp_steps * dim))

  # Plot trajectories
  fig = plt.figure(figsize=(8, 4))
  ax = [fig.add_subplot(121), fig.add_subplot(122)]

  fig3d = plt.figure(figsize=(8, 4))
  ax3d = [fig3d.add_subplot(121, projection='3d'), fig3d.add_subplot(122, projection='3d')]

  ax[0].scatter(trajwise[:, 0], trajwise[:, 1],
                c=score, vmin=0, vmax=1, cmap='viridis_r',
                  **common_scatter_args(7*PIXEL_S))

  ax3d[0].scatter(trajwise[:, 0], trajwise[:, 1], trajwise[:, 2],
                c=score, vmin=0, vmax=1, cmap='viridis_r',
                  **common_scatter_args(7*PIXEL_S))

  ax[0].set_title(f"Trajectory-wise PCA, evr={get_pca_evr(trajwise_pca, 3):.2f}%")
  ax3d[0].set_title(f"Trajectory-wise PCA, evr={get_pca_evr(trajwise_pca, 3):.2f}%")

  pca = PCA()
  pca.fit(np.unique(data.reshape((samples * mp_steps, dim)), axis=0))
  stepwise_global = pca.transform(data.reshape((samples * mp_steps, dim)))
  stepwise_global: np.ndarray
  stepwise_global = stepwise_global.reshape((samples, mp_steps, -1))
  assert stepwise_global.shape == (samples, mp_steps, min(samples * mp_steps, dim))


  for step in range(mp_steps):
    ax[1].scatter(stepwise_global[sample_len >= step, step, 0],
                  stepwise_global[sample_len >= step, step, 1],
                  label=f'mp step {step + 1}',
                  **common_scatter_args(7*PIXEL_S))

    ax3d[1].scatter(stepwise_global[sample_len >= step, step, 0],
                  stepwise_global[sample_len >= step, step, 1],
                  stepwise_global[sample_len >= step, step, 2],
                  label=f'mp step {step + 1}',
                  **common_scatter_args(7*PIXEL_S))

  if paths_drawn > 0:
    for sample in np.arange(0, samples, step=int(samples / paths_drawn)):
      ax[1].plot(stepwise_global[sample, :, 0],
                 stepwise_global[sample, :, 1],
                 **common_plot_args)
      ax3d[1].plot(stepwise_global[sample, :, 0],
                 stepwise_global[sample, :, 1],
                 stepwise_global[sample, :, 2],
                 **common_plot_args)

  ax[1].set_title(f"Step-wise PCA, evr={get_pca_evr(pca, 3):.2f}%")
  # fig.suptitle(f"{name}, accuracy={acc*100:.2f}%")
  fig.tight_layout()
  fig.savefig(f"{name}_ts_2d.png", dpi=300)
  plt.close()

  ax3d[1].set_title(f"Step-wise PCA, evr={get_pca_evr(pca, 3):.2f}%")
  # fig3d.suptitle(f"{name}, accuracy={acc*100:.2f}%")
  fig3d.tight_layout()
  fig3d.savefig(f"plots_new/{name}_ts_3d.png", dpi=300)

def run_experiment_final(name: str, path: str, paths_drawn=100):

  data_dump = np.load(path + '.npz', allow_pickle=True)
  data = data_dump['trajs']
  score = data_dump['score']
  true_lengths = data_dump['lengths']

  try:
    os.mkdir(f'plots_final')
  except OSError:
    pass

  data = data[true_lengths == 9, 0:9, :]
  score = score[true_lengths == 9]
  true_lengths = true_lengths[true_lengths == 9]

  # alt_score = np.linspace(0, 1, data.shape[0])


  # plot_stepwise_local(data, paths_drawn, true_lengths, name, 'diffs')

  traj_step(data, score, paths_drawn, true_lengths, name)

if __name__ == '__main__':
    run_experiment_final(name='Random TGMPNN Graphs',
                         path='value_gen_graphs/trajs_good_dists')
                         # path='../trajs')
