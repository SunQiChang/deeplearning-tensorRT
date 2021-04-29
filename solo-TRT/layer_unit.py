import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import cv2

def bottleneck(network, weights, idx, input, eps):
