import argparse
import bz2
import io
import math
import os
import pickle
from base64 import b64decode
from datetime import datetime

import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image
from google.colab.output import eval_js
from facegan import ROOT_PATH
#@title `Images through Webcam`
from IPython.display import HTML, Audio
from PIL import Image

VIDEO_HTML = """
<video autoplay
 width=%d height=%d style='cursor: pointer;'></video>
<script>

var video = document.querySelector('video')

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream=> video.srcObject = stream)

var data = new Promise(resolve=>{
  video.onclick = ()=>{
    var canvas = document.createElement('canvas')
    var [w,h] = [video.offsetWidth, video.offsetHeight]
    canvas.width = w
    canvas.height = h
    canvas.getContext('2d')
          .drawImage(video, 0, 0, w, h)
    video.srcObject.getVideoTracks()[0].stop()
    video.replaceWith(canvas)
    resolve(canvas.toDataURL('image/jpeg', %f))
  }
})
</script>
"""


def take_photo(
    quality=1.0,
    size=(800, 600),
):
  display(HTML(VIDEO_HTML % (size[0], size[1], quality)))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])
  f = io.BytesIO(binary)
  img = np.asarray(Image.open(f))

  timestamp_str = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
  filename = f'{ROOT_PATH}/data/raw/photo_{timestamp_str}.jpeg'
  Image.fromarray(img).save(filename)
  print('Image captured and saved to %s' % filename)


def unpack_bz2(src_path):
  data = bz2.BZ2File(src_path).read()
  dst_path = src_path[:-4]
  with open(dst_path, 'wb') as fp:
    fp.write(data)
  return dst_path


def split_to_batches(l, n):
  for i in range(0, len(l), n):
    yield l[i:i + n]


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def plot_two_images(
    img1: str = None,
    img2: str = None,
    img_id: str = None,
    fs: int = 12,
) -> None:
  f, axarr = plt.subplots(1, 2, figsize=(fs, fs))
  axarr[0].imshow(img1)
  axarr[0].title.set_text('Encoded img %d' % img_id)
  axarr[1].imshow(img2)
  axarr[1].title.set_text('Original img %d' % img_id)
  plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
  plt.show()


def display_sbs(
    folder1: str,
    folder2: str,
    res: int = 512,
) -> None:
  if folder1[-1] != '/':
    folder1 += '/'
  if folder2[-1] != '/':
    folder2 += '/'

  imgs1 = sorted([f for f in os.listdir(folder1) if ('.png' or '.jpg') in f])
  imgs2 = sorted([f for f in os.listdir(folder2) if ('.png' or '.jpg') in f])
  if len(imgs1) != len(imgs2):
    print(
        "Found different amount of images in aligned vs raw image directories. That's not supposed to happen..."
    )

  for i in range(len(imgs1)):
    img1 = Image.open(folder1 + imgs1[i]).resize((res, res))
    img2 = Image.open(folder2 + imgs2[i]).resize((res, res))
    plot_two_images(img1, img2, i)
    print("")


def move_latent_and_save(
    latent_vector,
    direction_file,
    coeffs,
    Gs_network,
    Gs_syn_kwargs,
):
  direction = np.load('data/latent_directions/' + direction_file)
  os.makedirs('data/results/' + direction_file.split('.')[0], exist_ok=True)
  for i, coeff in enumerate(coeffs):
    new_latent_vector = latent_vector.copy()
    new_latent_vector[0][:8] = (latent_vector[0] + coeff * direction)[:8]
    images = Gs_network.components.synthesis.run(
        new_latent_vector,
        **Gs_syn_kwargs,
    )
    result = PIL.Image.fromarray(images[0], 'RGB')
    result.thumbnail(size, PIL.Image.ANTIALIAS)
    result.save('data/results/' + direction_file.split('.')[0] + '/' +
                str(i).zfill(3) + '.png')
    if len(coeffs) == 1:
      return result
