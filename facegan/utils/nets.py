# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html
"""List of pre-trained StyleGAN2 networks located on Google Drive."""

import os
import glob
import gdown
import pickle
from facegan import ROOT_PATH
import facegan.dnnlib as dnnlib
import facegan.dnnlib.tflib as tflib

# ----------------------------------------------------------------------------
# StyleGAN2 Google Drive root: https://drive.google.com/open?id=1QHc-yF5C3DChRwSdZKcx1w6K8JvSxQi7

gdrive_urls = {
    'gdrive:networks/stylegan2-car-config-a.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-a.pkl',
    'gdrive:networks/stylegan2-car-config-b.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-b.pkl',
    'gdrive:networks/stylegan2-car-config-c.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-c.pkl',
    'gdrive:networks/stylegan2-car-config-d.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-d.pkl',
    'gdrive:networks/stylegan2-car-config-e.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-e.pkl',
    'gdrive:networks/stylegan2-car-config-f.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-f.pkl',
    'gdrive:networks/stylegan2-cat-config-a.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-cat-config-a.pkl',
    'gdrive:networks/stylegan2-cat-config-f.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-cat-config-f.pkl',
    'gdrive:networks/stylegan2-church-config-a.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-church-config-a.pkl',
    'gdrive:networks/stylegan2-church-config-f.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-church-config-f.pkl',
    'gdrive:networks/stylegan2-ffhq-config-a.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-a.pkl',
    'gdrive:networks/stylegan2-ffhq-config-b.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-b.pkl',
    'gdrive:networks/stylegan2-ffhq-config-c.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-c.pkl',
    'gdrive:networks/stylegan2-ffhq-config-d.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-d.pkl',
    'gdrive:networks/stylegan2-ffhq-config-e.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-e.pkl',
    'gdrive:networks/stylegan2-ffhq-config-f.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl',
    'gdrive:networks/stylegan2-horse-config-a.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-horse-config-a.pkl',
    'gdrive:networks/stylegan2-horse-config-f.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-horse-config-f.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gorig-Dorig.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gorig-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gorig-Dresnet.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gorig-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gorig-Dskip.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gorig-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gresnet-Dorig.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gresnet-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gresnet-Dresnet.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gresnet-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gresnet-Dskip.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gresnet-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gskip-Dorig.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gskip-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gskip-Dresnet.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gskip-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gskip-Dskip.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gskip-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dorig.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dresnet.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dskip.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dorig.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dresnet.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dskip.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dorig.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dresnet.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dskip.pkl':
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dskip.pkl',
}

vgg16 = 'vgg16_zhang_perceptual.pkl'
model = 'stylegan2-ffhq-config-f.pkl'
networks_urls = {
    'default': [
        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl',
        'stylegan2-ffhq-config-f.pkl'
    ],
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
    ]
}


def download_networks(network: str = 'default'):
  network_url = networks_urls[network][0]
  network_name = networks_urls[network][1]
  network_pkl = f'{ROOT_PATH}/data/models/{network_name}'
  if network == 'default':
    load_networks('gdrive:networks/stylegan2-ffhq-config-f.pkl')
    network_path = glob.glob(
        f'{ROOT_PATH}/data/models/*stylegan2-ffhq-config-f.pkl')[0]
    print(network_path)
    os.rename(network_path, network_pkl)
  else:
    gdown.download(
        network_url,
        network_name,
        quiet=False,
    )
    os.rename(f"{os.getcwd()}/{network_name}", network_pkl)
  return network_pkl


_cached_networks = dict()


def load_networks(path_or_url_path):
  if path_or_url_path in _cached_networks:
    return _cached_networks[path_or_url_path]

  url_path = gdrive_urls.get(path_or_url_path, '')
  if dnnlib.util.is_url(url_path):
    stream = dnnlib.util.open_url(
        url_path,
        cache_dir=f'{ROOT_PATH}/data/models',
    )
  else:
    stream = open(
        path_or_url_path,
        'rb',
    )

  tflib.init_tf()
  with stream:
    G, D, Gs = pickle.load(stream, encoding='latin1')
  _cached_networks[path_or_url_path] = G, D, Gs

  return G, D, Gs


# ----------------------------------------------------------------------------
