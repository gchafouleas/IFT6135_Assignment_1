import os
import shutil
import re

cat_path = 'data/trainset/Cat/'
dog_path = 'data/trainset/Dog/'

cat_valid = 'data/validset/Cat/'
dog_valid = 'data/validset/Dog/'

for f in os.listdir(cat_path):
    valid = re.search('[8-9]\d\d\d', f)
    if valid is not None:
        shutil.move(os.path.join(cat_path, f), cat_valid)

for f in os.listdir(dog_path):
    valid = re.search('[8-9]\d\d\d', f)
    if valid is not None:
        shutil.move(os.path.join(dog_path, f), dog_valid)
