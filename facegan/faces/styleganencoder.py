import glob
import os
import pickle
from types import SimpleNamespace

import numpy as np
import PIL.Image
from PIL import ImageFilter
from tensorflow.keras.models import load_model
from tqdm import tqdm

import facegan.dnnlib as dnnlib
import facegan.dnnlib.tflib as tflib
import facegan.utils.config as config
from facegan.encoder.generator_model import SG1Generator
from facegan.encoder.perceptual_model import PerceptualModel, load_images
from facegan.faces.base_encoder import Encoder
from facegan.utils.utils import split_to_batches


class StyleGANEncoder(Encoder):
  """Find latent representation of reference images using perceptual losses."""

  # Fetch a StyleGAN model to train on from this URL from # My Drive: karras2019stylegan-ffhq-1024x1024.pkl
  # Original Model here: https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ
  # (A lot of requests to this model ... sometimes Google Drive is out of quota for downloading this model)
  STYLEGAN_MODEL_URL = 'https://drive.google.com/uc?id=1o4zmXaHtd_oNL754NsXNWi9wYOlAPTC3'
  PERCEPTUALMODEL_URL = 'https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2'

  def __init__(self):
    super(StyleGANEncoder, self).__init__(
        model_url=StyleGANEncoder.STYLEGAN_MODEL_URL,
        iterations=500,  # Quicker for testing
    )

    self.build()

  def build(self) -> None:
    # Initialize generator and perceptual model

    tflib.init_tf()
    with dnnlib.util.open_url(
        self.model_url,
        cache_dir=config.cache_dir,
    ) as model:
      self.generator_network, self.discriminator_network, self.Gs_network = pickle.load(
          model)

    self.generator = SG1Generator(
        self.Gs_network,
        self.batch_size,
        clipping_threshold=self.clipping_threshold,
        tiled_dlatent=self.tile_dlatents,
        model_res=self.model_res,
        randomize_noise=self.randomize_noise,
    )
    if (self.dlatent_avg != ''):
      self.generator.set_dlatent_avg(np.load(self.dlatent_avg))

    self.perc_model = None
    if (self.use_lpips_loss > 0.00000001):
      with dnnlib.util.open_url(
          StyleGANEncoder.PERCEPTUALMODEL_URL,
          cache_dir=config.cache_dir,
      ) as model:
        self.perc_model = pickle.load(model)
    self.perceptual_model = PerceptualModel(
        SimpleNamespace(**vars(self)),
        perc_model=self.perc_model,
        batch_size=self.batch_size,
    )
    self.perceptual_model.build_perceptual_model(
        self.generator,
        self.discriminator_network,
    )

  def encode(self):

    if self.output_video:
      import cv2
      synthesis_kwargs = dict(
          output_transform=dict(func=tflib.convert_images_to_uint8,
                                nchw_to_nhwc=False),
          minibatch_size=self.batch_size,
      )

    # Get all the images
    images = glob.glob(f'{self.src_dir}/*.png') + \
              glob.glob(f'{self.src_dir}/*.jpg')

    # ref_images = [os.path.join(args.src_dir, x) for x in images]
    ref_images = list(filter(os.path.isfile, images))

    if len(ref_images) == 0:
      raise Exception('%s is empty' % self.src_dir)

    ff_model = None

    # Optimize (only) dlatents by minimizing perceptual loss
    # between reference and generated images in feature space
    for images_batch in tqdm(
        split_to_batches(ref_images, self.batch_size),
        total=len(ref_images) // self.batch_size,
    ):
      names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]

      if self.output_video:
        video_out = {}
        for name in names:
          video_out[name] = cv2.VideoWriter(
              os.path.join(self.video_dir, f'{name}.avi'),
              cv2.VideoWriter_fourcc(*self.video_codec),
              self.video_frame_rate,
              (self.video_size, self.video_size),
          )

      self.perceptual_model.set_reference_images(images_batch)

      # load previous dlatents for initialization
      dlatents = None
      if (self.load_last != ''):
        for name in names:
          dl = np.expand_dims(
              np.load(os.path.join(self.load_last, f'{name}.npy')),
              axis=0,
          )
          if (dlatents is None):
            dlatents = dl
          else:
            dlatents = np.vstack((dlatents, dl))
      else:
        if (ff_model is None):
          if os.path.exists(self.load_resnet):
            from tensorflow.keras.applications.resnet50 import preprocess_input
            print("Loading ResNet Model:")
            ff_model = load_model(self.load_resnet)

        if (ff_model is None):
          if os.path.exists(self.load_effnet):
            from efficientnet import preprocess_input
            print("Loading EfficientNet Model:")
            ff_model = load_model(self.load_effnet,)

        if (ff_model is not None):  # predict initial dlatents with ResNet model
          if (self.use_preprocess_input):
            dlatents = ff_model.predict(
                preprocess_input(
                    load_images(images_batch,
                                image_size=self.resnet_image_size)))
          else:
            dlatents = ff_model.predict(
                load_images(
                    images_batch,
                    image_size=self.resnet_image_size,
                ))

      if dlatents is not None:
        self.generator.set_dlatents(dlatents)

      op = self.perceptual_model.optimize(self.generator.dlatent_variable,
                                          iterations=self.iterations,
                                          use_optimizer=self.optimizer)

      pbar = tqdm(op, leave=False, total=self.iterations)
      vid_count, best_loss, best_dlatent, avg_loss_count = 0, None, None, 0

      if self.early_stopping:
        avg_loss = prev_loss = None

      for loss_dict in pbar:
        if self.early_stopping:
          if prev_loss is not None:
            if avg_loss is not None:
              avg_loss = 0.5 * avg_loss + (prev_loss - loss_dict["loss"])
              # count while under threshold; else reset
              if avg_loss < self.early_stopping_threshold:
                avg_loss_count += 1
              else:
                avg_loss_count = 0
              # stop once threshold is reached
              if avg_loss_count > self.early_stopping_patience:
                print("")
                break
            else:
              avg_loss = prev_loss - loss_dict["loss"]

        pbar.set_description(" ".join(names) + ": " + \
            "; ".join(["{} {:.4f}".format(k, v) for k, v in loss_dict.items()]))

        if best_loss is None or loss_dict["loss"] < best_loss:
          if best_dlatent is None or self.average_best_loss <= 0.00000001:
            best_dlatent = self.generator.get_dlatents()
          else:
            best_dlatent = (0.25 * best_dlatent +
                            0.75 * self.generator.get_dlatents())
          if self.use_best_loss:
            self.generator.set_dlatents(best_dlatent)
          best_loss = loss_dict["loss"]

        if self.output_video and (vid_count % self.video_skip == 0):
          batch_frames = self.generator.generate_images()
          for i, name in enumerate(names):
            video_frame = PIL.Image.fromarray(batch_frames[i], 'RGB').resize(
                (self.video_size, self.video_size),
                PIL.Image.LANCZOS,
            )
            video_out[name].write(
                cv2.cvtColor(
                    np.array(video_frame,).astype('uint8'),
                    cv2.COLOR_RGB2BGR,
                ))

        self.generator.stochastic_clip_dlatents()
        prev_loss = loss_dict["loss"]

      if not self.use_best_loss:
        best_loss = prev_loss
      print(" ".join(names), " Loss {:.4f}".format(best_loss))

      if self.output_video:
        for name in names:
          video_out[name].release()

      # Generate images from found dlatents and save them
      if self.use_best_loss:
        self.generator.set_dlatents(best_dlatent)

      generated_images = self.generator.generate_images()
      generated_dlatents = self.generator.get_dlatents()

      for img_array, dlatent, img_path, img_name in zip(
          generated_images,
          generated_dlatents,
          images_batch,
          names,
      ):
        mask_img = None
        if self.composite_mask and (self.load_mask or self.face_mask):
          _, im_name = os.path.split(img_path)
          mask_img = os.path.join(self.mask_dir, f'{im_name}')

        if self.composite_mask and mask_img is not None \
          and os.path.isfile(mask_img):

          orig_img = PIL.Image.open(img_path).convert('RGB',)
          width, height = orig_img.size
          imask = PIL.Image.open(mask_img).convert('L',).resize((width, height))
          imask = imask.filter(ImageFilter.GaussianBlur(self.composite_blur))
          mask = np.array(imask) / 255
          mask = np.expand_dims(mask, axis=-1)
          img_array = (mask * np.array(img_array) +
                       (1.0 - mask) * np.array(orig_img))
          img_array = img_array.astype(np.uint8)
          # img_array = np.where(mask, np.array(img_array), orig_img)

        img = PIL.Image.fromarray(img_array, 'RGB')
        img.save(os.path.join(self.generated_images_dir, f'{img_name}.png'),
                 'PNG')
        np.save(os.path.join(self.dlatent_dir, f'{img_name}.npy'), dlatent)

      self.generator.reset_dlatents()
