# FaceGAN: A research framework for Face Encoding, Modification, & Enhancement Using StyleGANs

FaceGAN is a library for face enhancement. FaceGAN strives to expose simple, efficient, modularized and readable code, that serve both as reference for using implementations of StyleGANs and generators, while still providing enough flexibility to do novel work.

## Overview

FaceGAN provide both StyleGAN 1 & 2 for encoding images into the latent space and provide several generator models for changing the latent directions and improve the faces in images. If you just want to get started using FaceGAN quickly:

`export PYTHONPATH="$PYTHONPATH:facegan"`

### MultiFaces Detection

```python
import cv2

from facegan.process.multifaces import MultiFaceCropper

test_image_path = './data/raw/6faces.jpg'
mfc = MultiFaceCropper()
img = cv2.imread(test_image_path)
images = mfc.crop(img)
```

### Face Alignment

```python
import cv2

from facegan.process.align_images import FaceAligner

test_image_path = './data/raw/6faces.jpg'
img = cv2.imread(test_image_path)

face_aligner = FaceAligner()
aligned_image = face_aligner.align(img)
```

### Face Encoding Baselines

FaceGAN provide 3 encoding using StyleGANs:

- Simple Encoder (for doing **fast** low quality encoding)
- StyleGAN1 Encoder
- StyleGAN2 Projector

```python
from facegan.faces.styleganencoder import StyleGANEncoder

sg_encoder = StyleGANEncoder()
sg_encoder.encode()
```

## Installation

> TODO: pip install facegan

## Resources

- [StyleGAN 1](https://github.com/NVlabs/stylegan)
- [StyleGAN 2](https://github.com/NVlabs/stylegan2)
- [stylegan-encoder](https://github.com/pbaylies/stylegan-encoder)
- [stylegan2encoder](https://github.com/rolux/stylegan2encoder)
- [generators-with-stylegan2](https://github.com/a312863063/generators-with-stylegan2)

## Useful Links

- [Google Generative Adversarial Networks (GANs)](https://developers.google.com/machine-learning/gan)
- [A Beginner's Guide to Generative Adversarial Networks (GANs)](https://pathmind.com/wiki/generative-adversarial-network-gan)
- [Best Resources for Getting Started With GANs](https://machinelearningmastery.com/resources-for-getting-started-with-generative-adversarial-networks/)
- [StyleGAN: Use machine learning to generate and customize realistic images](https://heartbeat.fritz.ai/stylegans-use-machine-learning-to-generate-and-customize-realistic-images-c943388dc672)
- [A Gentle Introduction to StyleGAN the Style Generative Adversarial Network](https://machinelearningmastery.com/introduction-to-style-generative-adversarial-network-stylegan/)
- [Introduction to GANs, NIPS 2016 | Ian Goodfellow, OpenAI](https://www.youtube.com/watch?v=9JpdAg6uMXs) (Video)
- [Ian Goodfellow: Generative Adversarial Networks (GANs) | Artificial Intelligence (AI) Podcast](https://www.youtube.com/watch?v=Z6rxFNMGdn0) (Video)
- [Deep Learning Part 2 2018 - Generative Adversarial Networks (GANs)](https://www.youtube.com/watch?v=ondivPiwQho) (Video)
- [CMU Generative Adversarial Networks (GANs)](https://www.youtube.com/watch?v=lXliALnsNzQ) (Video)
- [Tutorial on Generative adversarial networks - GANs as Learned Loss Functions](https://www.youtube.com/watch?v=eHQglSbS1zM) (Video)
- [Face editing with Generative Adversarial Networks](https://www.youtube.com/watch?v=dCKbRCUyop8) (Video)

## Citation

```bibtex
@misc{facegan2020,
    title={A research framework for Face Encoding, Modification, & Enhancement Using StyleGANs},
    author={Kayid, Amr},
    year={2020}
}
```

```bibtex
@article{Karras2019stylegan2,
  title   = {Analyzing and Improving the Image Quality of {StyleGAN}},
  author  = {Tero Karras and Samuli Laine and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  journal = {CoRR},
  volume  = {abs/1912.04958},
  year    = {2019},
}
```

```bibtex
@misc{karras2018stylebased,
    title={A Style-Based Generator Architecture for Generative Adversarial Networks},
    author={Tero Karras and Samuli Laine and Timo Aila},
    year={2018},
    eprint={1812.04948},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
}
```
