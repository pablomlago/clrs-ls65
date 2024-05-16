import os
import numpy as np
from absl import logging
import jax
import jax.numpy as jnp
from clrs._src import probing
import haiku as hk

class LatentsConfig:
    def __init__(self):
        self.save_latents = None
        self.filepath = None
        self.use_shared_latent_space = False
        self.shared_encoder = {}
        self.shared_decoder = {}

    def set_latents_filepath(
            self, 
            latents_path, 
            checkpoint_path, 
            algo_index, 
            algorithms, 
            seed, 
            processor_type,
            hint_mode):
        
        if len(algorithms) > 1 or "multi_task" in checkpoint_path:
            task_type = "multi_task"
        else:
            task_type = "single_task"

        if hint_mode == "none":
            hint_str = "no_hint_"
        else:
            hint_str = ""

        latents_dir = os.path.join(latents_path, task_type, f'{processor_type}_{hint_str}{seed}')
        os.makedirs(latents_dir, exist_ok=True)
        latents_filepath = os.path.join(latents_dir, f'{algorithms[algo_index]}.npz')
        self.filepath = latents_filepath
    
    def save_latents_to_file(self, latents):
        latent_arrays = {key: np.array(latents[key]) for key in latents.keys()}
        np.savez(self.filepath, **latent_arrays)
        logging.info('Latents saved to %s', self.filepath)
    
    
latents_config = LatentsConfig()

class RegularisationConfig:
    def __init__(self):
        self.use_hint_reversal = True
        self.use_causal_augmentation = False
        self.use_hint_relic = False

    def reverse_pointers(self, hints):

        reversed_hints = [
            probing.DataPoint(
                name=dp.name + '_reversed',
                location='edge',
                type_='pointer',
                data=jnp.matrix_transpose(
                    hk.one_hot(dp.data, dp.data.shape[-1]) if dp.location == 'node' else dp.data
                ) 
                # interesting behaviour with zeros, worth investigating
                # data=jnp.zeros((hk.one_hot(dp.data, dp.data.shape[-1]).shape)) if dp.location == 'node' else jnp.zeros((dp.data.shape))
            )
            for dp in hints if dp.type_ == 'pointer'
        ]

        return reversed_hints

regularisation_config = RegularisationConfig()