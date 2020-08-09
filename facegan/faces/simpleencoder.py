import os
import facegan.dnnlib as dnnlib
import facegan.dnnlib.tflib as tflib
import facegan.utils.nets as nets
from facegan import ROOT_PATH
from facegan.encoder.generator_model import Generator
from facegan.encoder.simple_perceptual_model import SimplePerceptualModel
from facegan.utils.utils import split_to_batches


class SimpleEncoder:

  def __init__(
      self,
      src_dir:
      str = f'{ROOT_PATH}/data/aligned',  # Directory with images for encoding
      generated_images_dir:
      str = f'{ROOT_PATH}/data/generated',  # Directory for storing generated images
      dlatent_dir:
      str = f'{ROOT_PATH}/data/latent_representations',  # Directory for storing dlatent representations
      data_dir:
      str = f'{ROOT_PATH}/data/models',  # Directory for storing optional models
      network_pkl:
      str = 'gdrive:networks/stylegan2-ffhq-config-f.pkl',  # Path to local copy of stylegan2-ffhq-config-f.pkl
      batch_size: int = 1,

      # Perceptual model params
      image_size: int = 256,  # Size of images for perceptual model
      lr: float = 1.,  # Learning rate for perceptual model
      iterations: int = 1000,  # Number of optimization steps for each batch

      # Noise
      randomize_noise: bool = False,  # Add noise to dlatents during optimization
  ) -> None:
    self.src_dir = src_dir
    self.generated_images_dir = generated_images_dir
    self.dlatent_dir = dlatent_dir
    self.data_dir = data_dir
    self.network_pkl = network_pkl
    self.batch_size = batch_size
    self.image_size = image_size
    self.lr = lr
    self.iterations = iterations
    self.randomize_noise = randomize_noise

    os.makedirs(self.generated_images_dir, exist_ok=True)
    os.makedirs(self.dlatent_dir, exist_ok=True)

    self.build()

  def build(self) -> None:
    # Initialize generator and perceptual model
    tflib.init_tf()
    self.generator_network, self.discriminator_network, self.Gs_network = nets.load_networks(
        self.network_pkl)

    self.generator = Generator(
        Gs_network,
        self.batch_size,
        randomize_noise=self.randomize_noise,
    )
    self.perceptual_model = SimplePerceptualModel(
        self.image_size,
        layer=9,
        batch_size=self.batch_size,
    )
    self.perceptual_model.build_perceptual_model(self.generator.generated_image)

  def encode(self):
    # Get all the images
    images = glob.glob(f'{self.src_dir}/*.png') + \
             glob.glob(f'{self.src_dir}/*.jpg')

    # ref_images = [os.path.join(self.src_dir, x) for x in images]
    ref_images = list(filter(os.path.isfile, images))

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch in tqdm(split_to_batches(ref_images, self.batch_size),
                             total=len(ref_images) // self.batch_size):
      names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]

      perceptual_model.set_reference_images(images_batch)
      op = perceptual_model.optimize(generator.dlatent_variable,
                                     iterations=self.iterations,
                                     learning_rate=self.lr)
      pbar = tqdm(op, leave=False, total=self.iterations)
      for loss in pbar:
        pbar.set_description(' '.join(names) + ' Loss: %.2f' % loss)
      print(' '.join(names), ' loss:', loss)

      # Generate images from found dlatents and save them
      generated_images = generator.generate_images()
      generated_dlatents = generator.get_dlatents()
      for img_array, dlatent, img_name in zip(generated_images,
                                              generated_dlatents, names):
        img = PIL.Image.fromarray(img_array, 'RGB')
        img.save(os.path.join(self.generated_images_dir, f'{img_name}.png'),
                 'PNG')
        np.save(os.path.join(self.dlatent_dir, f'{img_name}.npy'), dlatent)

      generator.reset_dlatents()
