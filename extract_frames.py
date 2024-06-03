import numpy as np
import pandas as pd 
import os 
import cv2

import imageio
import PIL
from PIL import Image

vid_path  = '/CMF/data/lumargot/hysterectomy/GBP_instruments/GBP 24.mp4'


save_path = 'frames/'

video_reader = imageio.get_reader(vid_path)
frame_num = 0
for frame in video_reader:    
    img_save_path = save_path + 'frame_' + str(frame_num) + ".png"
    img_result = PIL.Image.fromarray(frame)
    img_result.save(img_save_path)

    print("saving frame", frame_num)
    frame_num = frame_num + 1