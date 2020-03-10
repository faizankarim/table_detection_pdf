from predict import detect_image_64
import time
from PIL import Image
import base64
import cv2
import pandas as pd

images = pd.read_csv("val.csv")
cv2.namedWindow('test', cv2.WINDOW_NORMAL)
for index, row in images.iterrows():
    image_path = "images/" + row["image_id"]
    image = cv2.imread(image_path)
    image_path = "./img.png"
    cv2.imwrite(image_path, image)
    image = Image.open(image_path)
    encoded_string = ""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        file = encoded_string
    # Processing Time Starting
    start_time = time.time()
    box_1 = detect_image_64(encoded_string, probability=0.1)
    image = cv2.imread(image_path)
    k = 1
    for b in box_1:
        cv2.rectangle(image, (int(b[0][0]), int(b[0][2])), (int(b[0][1]), int(b[0][3])), (255, 0, 0 + (50 * k)), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(b[2]) + " - Probability", (10, 30 * k), font, 1, (255, 0, 0 + (50 * k)), 2, cv2.LINE_AA)
        k = k + 1
    #image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
    cv2.imwrite("results/" + str(index) + ".png", image)
    cv2.imshow("test", image)
    end_time = time.time()
    print(end_time - start_time)
    cv2.waitKey(25)
