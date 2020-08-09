# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import os
import shutil

import numpy as np
import tensorflow as tf

import facegan.dnnlib as dnnlib
import facegan.dnnlib.tflib as tflib
from facegan.training import dataset, misc

# ----------------------------------------------------------------------------


class Projector:

  def __init__(
      self,
      vgg16_pkl:
      str = 'https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2',
      num_steps: int = 1000,
      initial_learning_rate: float = 0.1,
      initial_noise_factor: float = 0.05,
      verbose: bool = False,
  ):

    self.vgg16_pkl = vgg16_pkl
    self.num_steps = num_steps
    self.dlatent_avg_samples = 10000
    self.initial_learning_rate = initial_learning_rate
    self.initial_noise_factor = initial_noise_factor
    self.lr_rampdown_length = 0.25
    self.lr_rampup_length = 0.05
    self.noise_ramp_length = 0.75
    self.regularize_noise_weight = 1e5
    self.verbose = verbose
    self.clone_net = True

    self._Gs = None
    self._minibatch_size = None
    self._dlatent_avg = None
    self._dlatent_std = None
    self._noise_vars = None
    self._noise_init_op = None
    self._noise_normalize_op = None
    self._dlatents_var = None
    self._noise_in = None
    self._dlatents_expr = None
    self._images_expr = None
    self._target_images_var = None
    self._lpips = None
    self._dist = None
    self._loss = None
    self._reg_sizes = None
    self._lrate_in = None
    self._opt = None
    self._opt_step = None
    self._cur_step = None

  def _info(self, *args):
    if self.verbose:
      print('Projector:', *args)

  def set_network(self, Gs, minibatch_size=1):
    assert minibatch_size == 1
    self._Gs = Gs
    self._minibatch_size = minibatch_size
    if self._Gs is None:
      return
    if self.clone_net:
      self._Gs = self._Gs.clone()

    # Find dlatent stats.
    self._info('Finding W midpoint and stddev using %d samples...' %
               self.dlatent_avg_samples)
    latent_samples = np.random.RandomState(123).randn(
        self.dlatent_avg_samples, *self._Gs.input_shapes[0][1:])
    dlatent_samples = self._Gs.components.mapping.run(latent_samples,
                                                      None)  # [N, 18, 512]
    self._dlatent_avg = np.mean(dlatent_samples, axis=0,
                                keepdims=True)  # [1, 18, 512]
    self._dlatent_std = (np.sum((dlatent_samples - self._dlatent_avg)**2) /
                         self.dlatent_avg_samples)**0.5
    self._info('std = %g' % self._dlatent_std)

    # Find noise inputs.
    self._info('Setting up noise inputs...')
    self._noise_vars = []
    noise_init_ops = []
    noise_normalize_ops = []
    while True:
      n = 'G_synthesis/noise%d' % len(self._noise_vars)
      if not n in self._Gs.vars:
        break
      v = self._Gs.vars[n]
      self._noise_vars.append(v)
      noise_init_ops.append(
          tf.assign(v, tf.random_normal(tf.shape(v), dtype=tf.float32)))
      noise_mean = tf.reduce_mean(v)
      noise_std = tf.reduce_mean((v - noise_mean)**2)**0.5
      noise_normalize_ops.append(tf.assign(v, (v - noise_mean) / noise_std))
      self._info(n, v)
    self._noise_init_op = tf.group(*noise_init_ops)
    self._noise_normalize_op = tf.group(*noise_normalize_ops)

    # Image output graph.
    self._info('Building image output graph...')
    self._dlatents_var = tf.Variable(
        tf.zeros([self._minibatch_size] + list(self._dlatent_avg.shape[1:])),
        name='dlatents_var')
    self._noise_in = tf.placeholder(tf.float32, [], name='noise_in')
    dlatents_noise = tf.random.normal(
        shape=self._dlatents_var.shape) * self._noise_in
    self._dlatents_expr = self._dlatents_var + dlatents_noise
    self._images_expr = self._Gs.components.synthesis.get_output_for(
        self._dlatents_expr, randomize_noise=False)

    # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
    proc_images_expr = (self._images_expr + 1) * (255 / 2)
    sh = proc_images_expr.shape.as_list()
    if sh[2] > 256:
      factor = sh[2] // 256
      proc_images_expr = tf.reduce_mean(tf.reshape(
          proc_images_expr,
          [-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor]),
                                        axis=[3, 5])

    # Loss graph.
    self._info('Building loss graph...')
    self._target_images_var = tf.Variable(tf.zeros(proc_images_expr.shape),
                                          name='target_images_var')
    if self._lpips is None:
      self._lpips = misc.load_pkl(self.vgg16_pkl)  # vgg16_zhang_perceptual.pkl
    self._dist = self._lpips.get_output_for(proc_images_expr,
                                            self._target_images_var)
    self._loss = tf.reduce_sum(self._dist)

    # Noise regularization graph.
    self._info('Building noise regularization graph...')
    reg_loss = 0.0
    for v in self._noise_vars:
      sz = v.shape[2]
      while True:
        reg_loss += tf.reduce_mean(
            v * tf.roll(v, shift=1, axis=3))**2 + tf.reduce_mean(
                v * tf.roll(v, shift=1, axis=2))**2
        if sz <= 8:
          break  # Small enough already
        v = tf.reshape(v, [1, 1, sz // 2, 2, sz // 2, 2])  # Downscale
        v = tf.reduce_mean(v, axis=[3, 5])
        sz = sz // 2
    self._loss += reg_loss * self.regularize_noise_weight

    # Optimizer.
    self._info('Setting up optimizer...')
    self._lrate_in = tf.placeholder(tf.float32, [], name='lrate_in')
    self._opt = dnnlib.tflib.Optimizer(learning_rate=self._lrate_in)
    self._opt.register_gradients(self._loss,
                                 [self._dlatents_var] + self._noise_vars)
    self._opt_step = self._opt.apply_updates()

  def run(self, target_images):
    # Run to completion.
    self.start(target_images)
    while self._cur_step < self.num_steps:
      self.step()

    # Collect results.
    pres = dnnlib.EasyDict()
    pres.dlatents = self.get_dlatents()
    pres.noises = self.get_noises()
    pres.images = self.get_images()
    return pres

  def start(self, target_images):
    assert self._Gs is not None

    # Prepare target images.
    self._info('Preparing target images...')
    target_images = np.asarray(target_images, dtype='float32')
    target_images = (target_images + 1) * (255 / 2)
    sh = target_images.shape
    assert sh[0] == self._minibatch_size
    if sh[2] > self._target_images_var.shape[2]:
      factor = sh[2] // self._target_images_var.shape[2]
      target_images = np.reshape(
          target_images,
          [-1, sh[1], sh[2] // factor, factor, sh[3] // factor, factor]).mean(
              (3, 5))

    # Initialize optimization state.
    self._info('Initializing optimization state...')
    tflib.set_vars({
        self._target_images_var:
            target_images,
        self._dlatents_var:
            np.tile(self._dlatent_avg, [self._minibatch_size, 1, 1])
    })
    tflib.run(self._noise_init_op)
    self._opt.reset_optimizer_state()
    self._cur_step = 0

  def step(self):
    assert self._cur_step is not None
    if self._cur_step >= self.num_steps:
      return
    if self._cur_step == 0:
      self._info('Running...')

    # Hyperparameters.
    t = self._cur_step / self.num_steps
    noise_strength = self._dlatent_std * self.initial_noise_factor * max(
        0.0, 1.0 - t / self.noise_ramp_length)**2
    lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
    lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
    lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
    learning_rate = self.initial_learning_rate * lr_ramp

    # Train.
    feed_dict = {self._noise_in: noise_strength, self._lrate_in: learning_rate}
    _, dist_value, loss_value = tflib.run(
        [self._opt_step, self._dist, self._loss], feed_dict)
    tflib.run(self._noise_normalize_op)

    # Print status.
    self._cur_step += 1
    if self._cur_step == self.num_steps or self._cur_step % 10 == 0:
      self._info('%-8d%-12g%-12g' % (self._cur_step, dist_value, loss_value))
    if self._cur_step == self.num_steps:
      self._info('Done.')

  def get_cur_step(self):
    return self._cur_step

  def get_dlatents(self):
    return tflib.run(self._dlatents_expr, {self._noise_in: 0})

  def get_noises(self):
    return tflib.run(self._noise_vars)

  def get_images(self):
    return tflib.run(self._images_expr, {self._noise_in: 0})


# ----------------------------------------------------------------------------


def project_image(
    proj,
    src_file,
    dst_dir,
    tmp_dir,
    video=False,
):
  data_dir = '%s/dataset' % tmp_dir
  if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
  image_dir = '%s/images' % data_dir
  tfrecord_dir = '%s/tfrecords' % data_dir
  os.makedirs(image_dir, exist_ok=True)
  shutil.copy(src_file, image_dir + '/')
  dataset_tool.create_from_images(tfrecord_dir, image_dir, shuffle=0)
  dataset_obj = dataset.load_dataset(
      data_dir=data_dir,
      tfrecord_dir='tfrecords',
      max_label_size=0,
      repeat=False,
      shuffle_mb=0,
  )

  print('Projecting image "%s"...' % os.path.basename(src_file))
  images, _labels = dataset_obj.get_minibatch_np(1)
  images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
  proj.start(images)
  if video:
    video_dir = '%s/video' % tmp_dir
    os.makedirs(video_dir, exist_ok=True)
  while proj.get_cur_step() < proj.num_steps:
    print(
        '\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps),
        end='',
        flush=True,
    )
    proj.step()
    if video:
      filename = '%s/%08d.png' % (video_dir, proj.get_cur_step())
      misc.save_image_grid(proj.get_images(), filename, drange=[-1, 1])
  print('\r%-30s\r' % '', end='', flush=True)

  os.makedirs(dst_dir, exist_ok=True)
  filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.png')
  misc.save_image_grid(proj.get_images(), filename, drange=[-1, 1])
  filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.npy')
  np.save(filename, proj.get_dlatents()[0])


def render_video(
    src_file,
    dst_dir,
    tmp_dir,
    num_frames,
    mode,
    size,
    fps,
    codec,
    bitrate,
):
  import PIL.Image
  import moviepy.editor

  def render_frame(t):
    frame = np.clip(np.ceil(t * fps), 1, num_frames)
    image = PIL.Image.open('%s/video/%08d.png' % (tmp_dir, frame))
    if mode == 1:
      canvas = image
    else:
      canvas = PIL.Image.new('RGB', (2 * src_size, src_size))
      canvas.paste(src_image, (0, 0))
      canvas.paste(image, (src_size, 0))
    if size != src_size:
      canvas = canvas.resize((mode * size, size), PIL.Image.LANCZOS)
    return np.array(canvas)

  src_image = PIL.Image.open(src_file)
  src_size = src_image.size[1]
  duration = num_frames / fps
  filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.mp4')
  video_clip = moviepy.editor.VideoClip(render_frame, duration=duration)
  video_clip.write_videofile(filename, fps=fps, codec=codec, bitrate=bitrate)
