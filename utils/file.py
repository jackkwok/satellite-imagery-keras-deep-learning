import os
from shutil import copyfile
from shutil import move

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)