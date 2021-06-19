import os
import itertools
import numpy as np
import faiss
import operator
import pickle
import datetime
from natsort import natsorted, ns
from . import featureVectorizer
from . import videoUtils
from . import evaluation


def max_two_values(d):
	"""
	Creates a list with dictionaries inside (key:value pairs)
	Returns two keys with max values.
	:param d: dict key:value
	:return: list
	"""
	v = list(d.values())
	k = list(d.keys())
	result1 = k[v.index(max(v))]
	del d[result1]
	v = list(d.values())
	k = list(d.keys())
	result2 = k[v.index(max(v))]
	return [result1, result2]


def fill_gaps(sequence, lookahead):
	"""
	Fills the gap in features sequence in case of the gap between 1's and 0's values.
	:param sequence: list consisting of 0's and 1's
	:param lookahead: skipping params in sequence
	:return: list of sequence filled where gap has occured
	"""
	i = 0
	change_needed = False
	look_left = 0
	while i < len(sequence):
		look_left -= 1
		if change_needed and look_left < 1:
			change_needed = False
		if sequence[i]:
			if change_needed:
				for k in to_change:
					sequence[k] = True
			else:
				change_needed = False
			look_left = lookahead
			to_change = []
		else:
			if change_needed:
				to_change.append(i)
		i += 1
	return sequence


def get_two_longest_timestamps(timestamps):
	"""
	Returns two longest intervals within given list of
	time intervals.
	:param timestamps: list of given intervals
	:return: int
	"""
	if len(timestamps) <= 2:
		return timestamps

	d = {}
	for start, end in timestamps:
		d[(start, end)] = end - start
	return max_two_values(d)


def to_time_string(seconds):
	return str(datetime.timedelta(seconds=seconds))


def query_episodes_with_faiss(videos, vectors_dir):
	"""
	With vector of the video file name and the dir
	where vectors located function performs query
	each set of episodes feature vectors on all of
	the other feature vectors. Returns distances
	to the best match found on each frame.
	:param videos: fideo feature vectors
	:param vectors_dir: video location
	:return: result
	"""
	vector_files = [os.path.join(vectors_dir, e+'.p') for e in videos]
	vectors = []
	lengths = []

	for f in vector_files:
		episode_vectors = np.array(pickle.load(open(f, 'rb')), np.float32)
		lengths.append(episode_vectors.shape[0])
		vectors.append(episode_vectors)

	vectors = np.vstack(vectors)
	results = []

	for i, lengths in enumerate(lengths):
		print(f"Querying {videos[i]}")
		i += 1
		s = sum(lengths[:i-1])
		e = sum(lengths[:i])
		query = vectors[s:e]
		rest = np.append(vectors[:s], vectors[e:], axis=0)
		vector_size = query.shape[1]
		index = faiss.IndexFlatL2(vector_size)
		index.add(rest)
		k = 1
		scores, indexes = index.search(query, k)
		result = scores[:, 0]
		results.append((videos[i-1], result))
	return results


