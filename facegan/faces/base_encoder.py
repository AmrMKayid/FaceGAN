import os


class Encoder:

  def __init__(
      self,
      src_dir: str = './data/aligned',  # Directory with images for encoding
      generated_images_dir:
      str = './data/generated',  # Directory for storing generated images
      dlatent_dir:
      str = './data/latent_representations',  # Directory for storing dlatent representations
      data_dir: str = './data/models',  # Directory for storing optional models
      mask_dir: str = 'data/masks',  # Directory for storing optional masks
      load_last: str = '',  # Start with embeddings from directory
      dlatent_avg:
      str = '',  # Use dlatent from file specified here for truncation instead of dlatent_avg from Gs
      model_url:
      str = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ',
      model_res: int = 1024,  # The dimension of images in the StyleGAN model
      batch_size: int = 1,  # Batch size for generator and perceptual model
      optimizer='adam',
      # Optimization algorithm used for optimizing dlatents (Using adam results from Exp 12 we conducted)

      # Perceptual model params
      image_size: int = 256,  # Size of images for the Resnet model
      resnet_image_size: int = 256,  # Size of images for the Resnet model
      lr: float = 0.001,  # Learning rate for perceptual model
      iterations: int = 1500,  # Number of optimization steps for each batch
      decay_rate: float = 0.99,  # Decay rate for learning rate
      decay_steps:
      float = 9,  # Decay steps for learning rate decay (as a percent of iterations)
      early_stopping: bool = True,  # Stop early once training stabilizes
      early_stopping_threshold:
      float = 0.005,  # Stop after this threshold has been reached
      early_stopping_patience:
      int = 10,  # Number of iterations to wait below threshold
      load_effnet:
      str = './data/models/finetuned_effnet.h5',  # Model to load for EfficientNet approximation of dlatents
      load_resnet:
      str = './data/models/finetuned_resnet.h5',  # Model to load for ResNet approximation of dlatents
      use_preprocess_input:
      bool = True,  # Call process_input() first before using feed forward net
      use_best_loss:
      bool = True,  # Output the lowest loss value found as the solution
      average_best_loss:
      float = 0.25,  # Do a running weighted average with the previous best dlatents found
      sharpen_input: bool = True,  # Sharpen the input images

      # Loss function options
      use_vgg_loss:
      float = 0.4,  # Use VGG perceptual loss; 0 to disable, > 0 to scale.
      use_vgg_layer: int = 9,  # Pick which VGG layer to use.
      use_pixel_loss:
      float = 1.5,  # Use logcosh image pixel loss; 0 to disable, > 0 to scale.
      use_mssim_loss:
      float = 200,  # Use MS-SIM perceptual loss; 0 to disable, > 0 to scale.
      use_lpips_loss:
      float = 0,  # Use LPIPS perceptual loss; 0 to disable, > 0 to scale.
      use_l1_penalty:
      float = 0.3,  # Use L1 penalty on latents; 0 to disable, > 0 to scale.
      use_discriminator_loss:
      float = 0,  # Use trained discriminator to evaluate realism.
      use_adaptive_loss:
      bool = False,  # Use the adaptive robust loss function from Google Research for pixel and VGG feature loss.

      # Generator params
      randomize_noise: bool = False,  # Add noise to dlatents during optimization
      tile_dlatents:
      bool = False,  # Tile dlatents to use a single vector at each scale
      clipping_threshold:
      float = 2.0,  # Stochastic clipping of gradient values outside of this threshold

      # Masking params
      load_mask: bool = False,  # Load segmentation masks
      face_mask: bool = True,  # Generate a mask for predicting only the face area
      use_grabcut:
      bool = True,  # Use grabcut algorithm on the face mask to better segment the foreground
      scale_mask:
      float = 1.4,  # Look over a wider section of foreground for grabcut (TODO: experiment with this)
      composite_mask:
      bool = True,  # Merge the unmasked area back into the generated image
      composite_blur:
      int = 8,  # Size of blur filter to smoothly composite the images

      # Video params
      video_dir: str = './data/videos',  # Directory for storing training videos
      output_video: bool = False,  # Generate videos of the optimization process
      video_codec: str = 'MJPG',  # FOURCC-supported video codec name
      video_frame_rate: int = 24,  # Video frames per second
      video_size: int = 512,  # Video size in pixels
      video_skip: int = 1,  # Only write every n frames (1 = write every frame,)
  ) -> None:
    self.src_dir = src_dir
    self.generated_images_dir = generated_images_dir
    self.dlatent_dir = dlatent_dir
    self.data_dir = data_dir
    self.mask_dir = mask_dir
    self.load_last = load_last
    self.dlatent_avg = dlatent_avg
    self.model_url = model_url
    self.model_res = model_res
    self.batch_size = batch_size
    self.optimizer = optimizer

    self.image_size = image_size
    self.resnet_image_size = resnet_image_size
    self.lr = lr
    self.iterations = iterations
    self.decay_rate = decay_rate
    self.decay_steps = decay_steps * 0.01 * self.iterations  # Calculate steps as a percent of total iterations
    self.early_stopping = early_stopping
    self.early_stopping_threshold = early_stopping_threshold
    self.early_stopping_patience = early_stopping_patience
    self.load_effnet = load_effnet
    self.load_resnet = load_resnet
    self.use_preprocess_input = use_preprocess_input
    self.use_best_loss = use_best_loss
    self.average_best_loss = average_best_loss
    self.sharpen_input = sharpen_input

    self.use_vgg_loss = use_vgg_loss
    self.use_vgg_layer = use_vgg_layer
    self.use_pixel_loss = use_pixel_loss
    self.use_mssim_loss = use_mssim_loss
    self.use_lpips_loss = use_lpips_loss
    self.use_l1_penalty = use_l1_penalty
    self.use_discriminator_loss = use_discriminator_loss
    self.use_adaptive_loss = use_adaptive_loss

    self.randomize_noise = randomize_noise
    self.tile_dlatents = tile_dlatents
    self.clipping_threshold = clipping_threshold

    self.load_mask = load_mask
    self.face_mask = face_mask
    self.use_grabcut = use_grabcut
    self.scale_mask = scale_mask
    self.composite_mask = composite_mask
    self.composite_blur = composite_blur

    self.video_dir = video_dir
    self.output_video = output_video
    self.video_codec = video_codec
    self.video_frame_rate = video_frame_rate
    self.video_size = video_size
    self.video_skip = video_skip

    os.makedirs(self.data_dir, exist_ok=True)
    os.makedirs(self.mask_dir, exist_ok=True)
    os.makedirs(self.generated_images_dir, exist_ok=True)
    os.makedirs(self.dlatent_dir, exist_ok=True)
    os.makedirs(self.video_dir, exist_ok=True)
