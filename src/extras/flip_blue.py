import skvideo.io as skv
import numpy as np
import os

assert os.path.isfile('computation_recorded.mp4')

video = skv.vreader('computation_recorded.mp4')
writer = skv.FFmpegWriter('video_corrected.mp4')
for frame in video:
    writer.writeFrame(np.flip(frame, axis=-1))