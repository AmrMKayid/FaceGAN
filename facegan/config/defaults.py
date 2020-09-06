from types import SimpleNamespace

from facegan import ROOT_PATH


class Config(SimpleNamespace):
  """Wrapper around SimpleNamespace, allows dot notation attribute access."""


def default():
  return Config(
      data=Config(
          # Root folder for all data
          _root=f'{ROOT_PATH}/data/',
          # Storing raw images
          raw=f'{ROOT_PATH}/data/aligned',
          # Storing cropped images
          cropped=f'{ROOT_PATH}/data/cropped',
          # Storing aligned images
          aligned=f'{ROOT_PATH}/data/aligned',
          # Storing generated images
          generated=f'{ROOT_PATH}/data/generated',
          # Storing optional masks
          masks=f'{ROOT_PATH}/data/masks',
          # Storing optional pretrained models
          models=f'{ROOT_PATH}/data/models',
          # Storing latent_representations from StyleGAN Encoding
          latent_representations=f'{ROOT_PATH}/data/latent_representations',
          # List of latent directions (age, gender, expressions, etc...)
          latent_directions=f'{ROOT_PATH}/data/latent_directions',
          # Directory for storing training videos
          videos=f'{ROOT_PATH}/data/videos',
      ),
      resolution=Config(
          image=1024,
          model=1024,
      ),
      masks=Config(
          # Load segmentation masks
          load_mask=False,
          # Generate a mask for predicting only the face area
          face_mask=True,
          # Use grabcut algorithm on the face mask to better segment the foreground
          use_grabcut=True,
          # Look over a wider section of foreground for grabcut (TODO: experiment with this)
          scale_mask=1.4,
          # Merge the unmasked area back into the generated image
          composite_mask=True,
          # Size of blur filter to smoothly composite the images
          composite_blur=8,
      ),
      urls=Config(
          # Fetch a StyleGAN model to train on from this URL from # My Drive: karras2019stylegan-ffhq-1024x1024.pkl
          # Original Model here: https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ
          # (A lot of requests to this model ... sometimes Google Drive is out of quota for downloading this model)
          stylegan=
          'https://drive.google.com/uc?id=1o4zmXaHtd_oNL754NsXNWi9wYOlAPTC3',
          perceptual=
          'https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2',
      ),
      models=Config(
          perceptual=Config(
              # Size of images for the Resnet model
              image_size=256,

              # Learning rate for perceptual model
              lr=0.001,
              # Number of optimization steps for each batch
              iterations=750,
              # Decay rate for learning rate
              decay_rate=0.99,
              # Decay steps for learning rate decay (as a percent of iterations)
              decay_steps=9,
              # Stop early once training stabilizes
              early_stopping=True,

              # Stop after this threshold has been reached
              early_stopping_threshold=0.005,
              # Number of iterations to wait below threshold
              early_stopping_patience=10,

              # Model to load for EfficientNet approximation of dlatents
              load_effnet=f'{ROOT_PATH}/data/models/finetuned_effnet.h5',
              # Model to load for ResNet approximation of dlatents
              load_resnet=f'{ROOT_PATH}/data/models/finetuned_resnet.h5',

              # Call process_input() first before using feed forward net
              use_preprocess_input=True,
              # Output the lowest loss value found as the solution
              use_best_loss=True,
              # Do a running weighted average with the previous best dlatents found
              average_best_loss=0.25,
              # Sharpen the input images
              sharpen_input=True,
          ),
          generator=Config(
              # Use dlatent from file specified here for truncation instead of dlatent_avg from Gs
              dlatent_avg="",
              # Add noise to dlatents during optimization
              randomize_noise=False,
              # Tile dlatents to use a single vector at each scale
              tile_dlatents=False,
              # Stochastic clipping of gradient values outside of this threshold
              clipping_threshold=2.0,
          ),
      ),
      loss=Config(
          # Use VGG perceptual loss; 0 to disable, > 0 to scale.
          use_vgg_loss=0.4,
          # Pick which VGG layer to use.
          use_vgg_layer=9,
          # Use logcosh image pixel loss; 0 to disable, > 0 to scale.
          use_pixel_loss=1.5,
          # Use MS-SIM perceptual loss; 0 to disable, > 0 to scale.
          use_mssim_loss=200,
          # Use LPIPS perceptual loss; 0 to disable, > 0 to scale.
          use_lpips_loss=100,  #it was zero (TODO: from exps?! need to check) 0,
          # Use L1 penalty on latents; 0 to disable, > 0 to scale.
          use_l1_penalty=0.3,
          # Use trained discriminator to evaluate realism.
          use_discriminator_loss=0,
          # Use the adaptive robust loss function from Google Research for pixel and  VGG feature loss.
          use_adaptive_loss=False,
      ),
      video=Config(
          # Generate videos of the optimization process
          output_video=False,
          # FOURCC-supported video codec name
          video_codec='MJPG',
          # Video frames per second
          video_frame_rate=24,
          # Video size in pixels
          video_size=512,
          # Only write every n frames (1 = write every frame,)
          video_skip=1,
      ),
      # Batch size for generator and perceptual model
      batch_size=1,
      # Optimizing dlatents (Using adam results from Exp 12 we conducted)
      optimizer='adam',
      lr=0.0001,
  )
