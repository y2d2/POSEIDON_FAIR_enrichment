import unittest
import os
import Code.Simulation.MultiRobotClass
import numpy as np
import shutil

import pickle as pkl

from Code.UtilityCode import Measurement

import matplotlib
from PIL import Image
import moviepy.video.io.ImageSequenceClip

from Code.UtilityCode.utility_fuctions import get_4d_rot_matrix

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
class MyTestCase(unittest.TestCase):
    def resize_image_if_needed(self,img, target_size):
        if img.size != target_size:
            print(f"Resizing {img.size} to {target_size}")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        return img

    def test_convert_images(self):
        # image_folder = '/home/yuri/Documents/PhD/ROS_WS/sharedDrive/Aerolytics/serr_18_9_25_test/log0002/run/mpa/tracking_front'
        image_folder = '/home/yuri/Documents/PhD/ROS_WS/sharedDrive/Aerolytics/serr_18_9_25_test/log0000/run/mpa/qvio_overlay'
        save_folder = "./correcte_images_for_FAIRD_0"
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.mkdir(save_folder)
        # target_size = (1280, 800)
        target_size = (1280, 896)
        for i in range(0, 4694):
            # make the names with leading zeros
            filename = str(i).zfill(5) + ".png"
            # print(filename)
            image_path = os.path.join(image_folder, filename)
            save_image_path = os.path.join(save_folder, filename)
            if os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        img = self.resize_image_if_needed(img, target_size)
                        arr = np.array(img)
                        print(f"{i} image shape: {arr.shape}")
                        if len(arr.shape) == 2:  # Grayscale
                            img = img.convert("RGB")
                        img.save(save_image_path)
                except Exception as e:
                    print(f"Warning: {image_path} is unreadable or corrupted. {e}")
            else:
                print(f"Warning: {image_path} does not exist.")

    def test_create_movie(self):
        image_folder = "./correcte_images_for_FAIRD_0"
        fps = 30
        image_files = []
        for i in range(0 ,4694):
            # make the names with leading zeros
            filename = str(i).zfill(5) + ".png"
            # print(filename)
            image_path = os.path.join(image_folder, filename)
            if os.path.exists(image_path):
                image_files.append(image_path)
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile('FAIRDay_movie_0.mp4')
