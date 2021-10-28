# In this part we compare two pictures. 
# this is still in early stage of the devolopment 

from PIL import Image

from pixelmatch.contrib.PIL import pixelmatch

# opening image 1
img_a = Image.open("a.png")

# opening image 2
img_b = Image.open("b.png")

# calculating image difference
img_diff = Image.new("RGBA", img_a.size)

# note how there is no need to specify dimensions
mismatch = pixelmatch(img_a, img_b, img_diff, includeAA=True)

img_diff.save("diff.png")
