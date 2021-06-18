import cv2
import ffmpeg


def get_framerate(video_file):
	"""
	Function get_framerate(video_file).
	Using pre-builtins of ffmpeg methods
	returns the value of framerate for the
	video file provided.
	:param video_file: str video file name in form of local link,
					e.g. './videos/video_one.mp4'
	:return: int framerate per video file measured in fps (frames per second).
	"""
	video = cv2.VideoCapture(video_file)
	return video.get(cv2.CAP_PROP_FPS)


def resize(video_file, outfile, resize_width):
	"""
	Function resize(input, output, resize_width).
	Using the pre-builtings of ffmpeg methods
	resizes a video file to provided resize_width ratio.
	:param video_file: video file in the form of an array or raw.
	:param outfile: resized video
	:param resize_width: resize_width ratio.
	:return: resized video.
	"""
	video = cv2.VideoCapture(video_file)
	frame_count = get_framerate(video)

	if frame_count > 0:
		stream = ffmpeg.input(video_file)
		if resize_width == 224:
			stream = ffmpeg.filter(stream, 'scale', w=244, h=244)
		else:
			# in order to return the same aspect ratio during resizing the h = trunc(ow/a/2)*2
			# more here ---> https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
			stream = ffmpeg.filter(stream, 'scale', w=resize_width, h="trunc(ow/a/2)*2")
		stream = ffmpeg.output(stream, outfile)
		try:
			ffmpeg.run(stream)
		except FileNotFoundError:
			raise Exception("ffmpeg is not found on your device, install ffmpeg")
	else:
		raise Exception(f"The video file {video_file} provided is not supported or corrupted.")
