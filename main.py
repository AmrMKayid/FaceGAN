from facegan.faces.styleganencoder import StyleGANEncoder
from facegan.faces.simpleencoder import SimpleEncoder
from facegan.process.align_images import FaceAligner
from facegan.process.multifaces import MultiFaceCropper

if __name__ == '__main__':
  mfc = MultiFaceCropper()
  images = mfc.auto_crop()

  face_aligner = FaceAligner()
  face_aligner.auto_align()

  sg_encoder = StyleGANEncoder()
  print(vars(sg_encoder))
  sg_encoder.encode()

  sg_encoder = SimpleEncoder()
  print(vars(sg_encoder))
  sg_encoder.encode()
