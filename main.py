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

def compair_position(xmin,ymin,xmax,ymax  ,  xmin2,ymin2,xmax2,ymax2):
  cordenada1 = (xmin, ymin , xmax, ymax)
  cordenada2 = (xmin2, ymin2 , xmax2, ymax2)
  area1 = (xmax - xmin) * (ymax - ymin)
  area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
  inter_xmin = max(xmin, xmin2)
  inter_ymin = max(ymin, ymin2)
  inter_xmax = min(xmax, xmax2)
  inter_ymax = min(ymax, ymax2)

  if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
    return 0

  inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
  iou = inter_area / (area1 + area2 - inter_area)
  if iou > 0.5:
    return 1
  return 0

def bounding_box(landmark ,  width , height):
  #print(landmark, "\n")
  
  xmin = float('inf')
  ymin = float('inf')
  xmax = float('-inf')
  ymax = float('-inf')
  for poit in landmark:
    x = int(poit.x * width)
    y = int(poit.y * height)
    if x < xmin:
      xmin = x
    if y < ymin:
      ymin = y
    if x > xmax:
      xmax = x
    if y > ymax:
      ymax = y
  print(xmin, ymin, xmax, ymax)
  return (xmin, ymin, xmax, ymax)
  

def cv2_imshow(a):
  cv2.imshow('Webcam', a)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    return True  # Indicate to exit
  return False


def draw_landmarks_on_image(rgb_image, pose_landmarks):
  
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h , w , _ = frame.shape
        frame_timestamp_ms = int((time.time() - initial_time) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)

        #lista de poseLandmarkList, contem pontos de cada pessoa detectada. Usada em duas funções, para desenhar os pontos e para calcular a bounding box de cada pessoa.
        poseLandmarkList = detection_result.pose_landmarks
        if poseLandmarkList is None:
            print("No person detected, skipping pose landmark processing.")
            if cv2_imshow(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                break
            continue
        
        for idx, poseLandmark in enumerate(poseLandmarkList):
          annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), poseLandmark)
          bounding_boxes = bounding_box(poseLandmark , w , h)
          cv2.rectangle(annotated_image, bounding_boxes[:2], bounding_boxes[2:], (0, 255, 0), 2)
        if cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)):
            break

    cam.release()
    cv2.destroyAllWindows()
        
    