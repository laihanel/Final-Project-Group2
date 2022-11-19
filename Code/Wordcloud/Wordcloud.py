import os
from os import path
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from nltk.corpus import stopwords


text = open('Data.txt').read()

background_Image = np.array(Image.open("background.png"))
img_colors = ImageColorGenerator(background_Image)
stopwords = set(stopwords.words('English'))

wc = WordCloud(margin = 2,
               scale=2,
               mask = background_Image,
               max_font_size = 140,
               stopwords = stopwords,
               background_color = 'white',
               )

wc.generate_from_text(text)

#subtract image color
wc.recolor(color_func=img_colors)

plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.show()


# wc.to_file('wordcloud.png')
# plt.savefig('wordcloud.png',dpi=200)

