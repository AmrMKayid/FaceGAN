<h1 align=center> FaceGAN </h1>


## MultiFaces Detection

```python
import cv2

from facegan.process.multifaces import MultiFaceCropper

test_image_path = './data/raw/6faces.jpg'
mfc = MultiFaceCropper()
img = cv2.imread(test_image_path)
images = mfc.crop(img)
```


## Face Alignment

```python
import cv2

from facegan.process.align_images import FaceAligner

test_image_path = './data/raw/6faces.jpg'
img = cv2.imread(test_image_path)

face_aligner = FaceAligner()
aligned_image = face_aligner.align(img)
```
