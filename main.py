import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import PIL
from IPython import display
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision

def cv2_imshow(a):
  a = a.clip(0, 255).astype('uint8')
  # cv2 stores colors as BGR; convert to RGB
  if a.ndim == 3:
    if a.shape[2] == 4:
      a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
    else:
      a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
  display.display(PIL.Image.fromarray(a))


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image


model_path = 'pose_landmarker_lite.task'

initial_time = time.time()


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('pose landmarker result: {}'.format(result))


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.IMAGE,
    output_segmentation_masks=True,
)
detector = vision.PoseLandmarker.create_from_options(options)


with PoseLandmarker.create_from_options(options) as landmarker:
    cam = cv2.VideoCapture(0)
    while cam.isOpened():
        flag , frame = cam.read()
        frame_timestamp_ms = int((time.time() - initial_time) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)



        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


        if detection_result.segmentation_masks is not None:
            segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
            segmentation_mask = np.squeeze(segmentation_mask)

    
            visualized_mask = (segmentation_mask * 255).astype(np.uint8)
            visualized_mask = np.stack([visualized_mask]*3, axis=-1)
    
        # ... process your mask ...
        else:
            print("No person detected, skipping mask processing.")
    