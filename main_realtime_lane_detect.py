from cv2 import Mat
import qcsnpe as qc
import numpy as np
import cv2
import torch
import scipy.special
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum
from scipy.spatial.distance import cdist

CPU = 0
GPU = 1
DSP = 2

lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
			116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
			168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
			220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
			272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

class ModelType(Enum):
	TUSIMPLE = 0
	CULANE = 1

class ModelConfig():

	def __init__(self, model_type):

		if model_type == ModelType.TUSIMPLE:
			self.init_tusimple_config()
		else:
			self.init_culane_config()

	def init_tusimple_config(self):
		self.img_w = 1280
		self.img_h = 720
		self.row_anchor = tusimple_row_anchor
		self.griding_num = 100
		self.cls_num_per_lane = 56

	def init_culane_config(self):
		self.img_w = 1640
		self.img_h = 590
		self.row_anchor = culane_row_anchor
		self.griding_num = 200
		self.cls_num_per_lane = 18


class SNPELaneDetector():

    def __init__(self):

        # Initialize image transformation
        self.img_transform = self.initialize_image_transform()
        self.cfg = ModelConfig(model_type=ModelType.TUSIMPLE)


    @staticmethod
    def initialize_image_transform():

        # Create transfom operation to resize the input images
        img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
        ])

        return img_transforms

    def detect_lanes(self, image, draw_points=True):
        input_tensor = self.prepare_input(image)
        out = model.predict(input_tensor)
        output_tensor = out['200']
        output_tensor = np.array(output_tensor)
        output_tensor = np.reshape(output_tensor,(1,101,56,4))
         # Process output data
        self.lanes_points, self.lanes_detected = self.process_output(output_tensor, self.cfg)
        # Draw depth image
        visualization_img = self.draw_lanes(image, self.lanes_points, self.lanes_detected, self.cfg, draw_points)

        return visualization_img

    def prepare_input(self, img):
        # Transform the image for inference
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        input_img = self.img_transform(img_pil)
        input_img = np.array(input_img)
      
        return input_img
        
    @staticmethod
    def process_output(output, cfg):		
        # Parse the output of the model
        processed_output = output[0]
        # print(processed_output.shape)
        # exit(0)
        processed_output = processed_output[:, ::-1, :]
        prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        processed_output = np.argmax(processed_output, axis=0)
        loc[processed_output == cfg.griding_num] = 0
        processed_output = loc
        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        lanes_points = []
        lanes_detected = []

        max_lanes = processed_output.shape[1]
        for lane_num in range(max_lanes):
            lane_points = []
            # Check if there are any points detected in the lane
            if np.sum(processed_output[:, lane_num] != 0) > 2:

                lanes_detected.append(True)

                # Process each of the points for each lane
                for point_num in range(processed_output.shape[0]):
                    if processed_output[point_num, lane_num] > 0:
                        lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1, int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num]/288)) - 1 ]
                        lane_points.append(lane_point)
            else:
                lanes_detected.append(False)

            lanes_points.append(lane_points)
        return np.array(lanes_points), np.array(lanes_detected)

    @staticmethod
    def draw_lanes(input_img, lanes_points, lanes_detected, cfg, draw_points=True):
        # Write the detected line points in the image
        visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation = cv2.INTER_AREA)

        # Draw a mask for the current lane
        if(lanes_detected[1] and lanes_detected[2]):
            lane_segment_img = visualization_img.copy()
            
            cv2.fillPoly(lane_segment_img, pts = [np.vstack((lanes_points[1],np.flipud(lanes_points[2])))], color =(255,191,0))
            visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)

        if(draw_points):
            for lane_num,lane_points in enumerate(lanes_points):
                for lane_point in lane_points:
                    cv2.circle(visualization_img, (lane_point[0],lane_point[1]), 3, lane_colors[lane_num], -1)

        return visualization_img


out_layers = np.array(["Reshape_53"])
model = qc.qcsnpe("models/lanenet.dlc", out_layers, CPU)
# Initialize lane detection model
lane_detector = SNPELaneDetector()
cap = cv2.VideoCapture(0)

while cap.isOpened():
	try:
		# Read frame from the video
		ret, frame = cap.read()
	except:
		continue

	if ret:	

		# Detect the lanes
        
		output_img = lane_detector.detect_lanes(frame)

		cv2.imshow("Detected lanes", output_img)

	else:
		break

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()



