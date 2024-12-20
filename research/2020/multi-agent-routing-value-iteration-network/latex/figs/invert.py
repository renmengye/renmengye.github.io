from PIL import Image
import PIL.ImageOps
import sys

image = Image.open(sys.argv[1]).convert('RGB')

inverted_image = PIL.ImageOps.invert(image)

inverted_image.save(sys.argv[2])