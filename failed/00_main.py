import os
import sys
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def predict(chosen_model, img, classes=[], conf=0.5):
  if classes:
    results = chosen_model.predict(img, classes=classes, conf=conf)
  else:
    results = chosen_model.predict(img, conf=conf)
  return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
  results = predict(chosen_model, img, classes, conf=conf)
  for result in results:
    for box in result.boxes:
      cv2.rectangle(
        img,
        (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
        (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
         (255, 0, 0), 2
      )
      cv2.putText(
        img,
        f"{result.names[int(box.cls[0])]}",
        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (255, 0, 0),
        1
      )
  return img, results

input_file = "./samples/camping.png"
output_file = "./out/" + os.path.basename(input_file)

print("reading input file")
image = cv2.imread(input_file)
if image is None:
  print("Error loading image", input_file)
  sys.exit()
print("detecting and predicting")
result_img = predict_and_detect(model, image, classes=[], conf=0.5)
print("result:", result_img)

print("showing and writing result")
#cv2.imshow("Image", result_img)
cv2.imwrite(output_file, result_img)
cv2.waitKey(0)
