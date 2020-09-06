import json
import os
from pathlib import Path

import facegan.dnnlib.tflib as tflib
from facegan import ROOT_PATH
from facegan.config import default


class Encoder:

  def __init__(
      self,
      config=default(),
  ) -> None:
    self._config = config
    # Calculate steps as a percent of total iterations
    self._config.models.perceptual.decay_steps *= 0.01 * self._config.models.perceptual.iterations

    for dir_name, path in vars(config.data).items():
      print(f"Creating {dir_name} directory")
      os.makedirs(path, exist_ok=True)

    ## TODO: Videos
    if self._config.video.output_video:
      import cv2
      self.synthesis_kwargs = dict(
          output_transform=dict(
              func=tflib.convert_images_to_uint8,
              nchw_to_nhwc=False,
          ),
          minibatch_size=self._config.batch_size,
      )

    # TODO: change location of this 🤔
    with open(
        f'{Path(ROOT_PATH).parent}/configs.json',
        "w",
        encoding="utf-8",
    ) as f:
      json.dump(vars(config), f, default=lambda o: o.__dict__)

  @property
  def config(self):
    return self._config
