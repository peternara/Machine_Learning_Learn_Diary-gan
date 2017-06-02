# coding: utf-8
import os

for parent, dirnames, filenames in os.walk('./女装'):
    for key, val in enumerate(filenames):
        path = os.path.join(parent, val)
        os.rename(path, os.path.join(parent, 'T' + str(key).zfill(3) + '.jpg'))

for parent, dirnames, filenames in os.walk('./非女装'):
    for key, val in enumerate(filenames):
        path = os.path.join(parent, val)
        os.rename(path, os.path.join(parent, 'F' + str(key).zfill(3) + '.jpg'))
