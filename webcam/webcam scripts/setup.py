from distutils.core import setup # Need this to handle modules
import py2exe
import cv2
import numpy as np
import os
import math # We have to import all modules used in our program

setup(windows=['webcam face recognizer.py'])