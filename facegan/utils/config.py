# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""Global configuration."""

# ----------------------------------------------------------------------------
# Paths.
from facegan import ROOT_PATH

result_dir = 'results'
data_dir = 'datasets'
cache_dir = f'{ROOT_PATH}/data/cache'
run_dir_ignore = ['results', 'datasets', 'cache']

# experimental - replace Dense layers with TreeConnect
use_treeconnect = False
treeconnect_threshold = 1024

# ----------------------------------------------------------------------------

vgg16 = 'vgg16_zhang_perceptual.pkl'
model = 'stylegan2-ffhq-config-f.pkl'

networks_urls = {
    'european': [
        'https://drive.google.com/uc?id=1--kh2Em5U1qh-H7Lin9FzppkZCQ18c4W',
        'generator_model-stylegan2-config-f.pkl'
    ],
    'asian': [
        'https://drive.google.com/uc?id=1-3XU6KzIVywFoKXx2zG1hW8mH4OYpyO9',
        'generator_yellow-stylegan2-config-f.pkl'
    ],
    'asian beauty': [
        'https://drive.google.com/uc?id=1-04v78_pI59M0IvhcKxsm3YhK2-plnbj',
        'generator_star-stylegan2-config-f.pkl'
    ],
    'baby': [
        'https://drive.google.com/uc?id=1--684mANXSgC3aDhLc7lPM7OBHWuVRXa',
        'generator_baby-stylegan2-config-f.pkl'
    ],
}
