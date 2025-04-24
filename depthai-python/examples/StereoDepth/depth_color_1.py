#!/usr/bin/env python3

import cv2
import depthai as dai

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutDepth.setStreamName("depth")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# Configure the depth node
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)  # Use a 7x7 median filter
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)

# Link the cameras to the depth node
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)

# Link the depth output to the XLinkOut node
depth.depth.link(xoutDepth.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the depth frames
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        inDepth = depthQueue.get()  # blocking call, waits until new data arrives
        frame = inDepth.getCvFrame()  # Get depth frame
        normalizedFrame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)  # Normalize for visualization
        depthColorFrame = cv2.applyColorMap(normalizedFrame.astype('uint8'), cv2.COLORMAP_JET)  # Apply a color map
        
        cv2.imshow("depth", depthColorFrame)

        if cv2.waitKey(1) == ord('q'):
            break
