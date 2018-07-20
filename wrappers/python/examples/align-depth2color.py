## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

"""
Install:

    1.  ```sh
        brew install libusb pkg-config
        brew install homebrew/core/glfw3
        brew install cmake
        ```
    
    2.  ```sh
        brew install python
        brew install opencv
        ```

Build library: (in project root)

    1.  ```sh
        mkdir build && cd build
        cmake .. -DBUILD_EXAMPLES=true -DBUILD_WITH_OPENMP=false -DHWM_OVER_XU=false -G Xcode
        cmake .. -DBUILD_PYTHON_BINDINGS=bool:true -DPYTHON_EXECUTABLE=$(which python3)
        open librealsense2.xcodeproj
        ```

    2.  build the Xcode project
"""

import sys, os, time
sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + '/../../../build/wrappers/python/Debug'))

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

save_dir = 'images'
counter = 0

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)
        images = np.vstack((color_image, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', cv2.resize(images, (640, 720)))

        # press space to save
        if cv2.waitKey(1) == ord(' '):
            while True:
                basename = save_dir + "/" + str(counter)
                color_filename = basename + '_color.png'
                if os.path.exists(color_filename):
                    counter += 1
                else:
                    break
            cv2.imwrite(color_filename, color_image)
            cv2.imwrite(basename + '_depth.png', depth_image)
            cv2.imwrite(basename + '_depth_view.png', depth_colormap)

            timestamp = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(int(time.time())))
            print('saved(' + timestamp + '): ' + basename)
finally:
    pipeline.stop()

