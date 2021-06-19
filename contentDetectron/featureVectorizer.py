import cv2
import numpy as np
import os
import json
import pickle
from math import sqrt
from tqdm import tqdm
from . import rmac


def get_frame(frame_index, video):
	"""
	The function get_frame with the given position of frame index and videocapture
	variable returns the frame as an image object in the form of numpy ndarray.
	:param frame_index: int frame index
	:param video: local url for the video file
	:return: image of the frame in the form of numpy ndarray
	"""
	video.set(1, frame_index)
	_, img = video.read()
	return img

fouriers = [
	[1, 1, 1, 1, 1, 1, 1, 1],
	[-1, 1, -1, 1, 1, -1, 1, -1],
	[-sqrt(2)/2, 0, sqrt(2)/ 2, -1, 1, -sqrt(2)/2, 0, sqrt(2)/2],
	[-sqrt(2)/2, -1, -sqrt(2)/2, 0, 0, sqrt(2)/2, 1, sqrt(2)/2],
	[0, -1, 0, 1, 1, 0, -1, 0],
	[1, 0, -1, 0, 0, -1, 0, 1],
	[sqrt(2)/2, 0, -sqrt(2)/2, -1, 1, sqrt(2)/2, 0, -sqrt(2)/2],
	[-sqrt(2)/2, 1, -sqrt(2)/2, 0, 0, sqrt(2)/2, -1, sqrt(2)/2]
]

for i, f in enumerate(fouriers):
	f.insert(4,0)
	fouriers[i] = np.array(f)
	fouriers[i] = fouriers[i].reshape((3,3)).astype('float32')

max_vals = []
for f in fouriers:
	m = np.array([255])
	m = cv2.matchTemplate(m.astype('float32'), f, cv2.TM_CCORR).clip(0, 255)
	max_vals.append(cv2.matchTemplate(m.astype('float32'), f, cv2.TM_CCORR)[0][0])


