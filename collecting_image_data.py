import cv2
import os
import sys
import argparse
from imutils.video import VideoStream
import time

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--name', required=True, help='name of the directory')
#ap.add_argument('-o', '--output', required=True, help='path to save the directory.')

args = vars(ap.parse_args())

try:
    label_name = args['name']
    #output_dir = args['output']
except:
    print('Arguments missing.')
    exit(-1)

os.chdir('Dataset')
base_dir = os.getcwd()

image_dir = os.path.join(base_dir, label_name)

try:
    os.mkdir(image_dir)
except FileExistsError:
    print("{} Directory already exists".format(image_dir))
    print("Images gathered will be saved here.")

print("[INFO] Starting video stream....")
vs = VideoStream(src=0, framerate=30).start()
time.sleep(2.0)


sample_req = 100
sample_taken = 0
start = False
while True:
    frame = vs.read()
    orig = frame.copy()

    

    if sample_taken == sample_req:
        break

    #(h,w) = frame.shape[:2]
    #(x,y) = (w//2, h//2)
    cv2.rectangle(frame, (220,140), (420,340), (0, 255, 0), 1)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        start=True
    
    elif key== ord('q'):
        break

    if start:
        roi = orig[140:340, 220:420]
        save_path = os.path.join(image_dir, "{}.png".format(sample_taken))
        cv2.imwrite(save_path, roi)
        sample_taken+=1
        time.sleep(0.05)

    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(frame, "Please keep your head inside the Square and rotate clockwise.", (10,20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Setting up Face ID",frame)

print("[INFO] {} images stored".format(sample_taken))
print("[INFO] cleaning up......")

cv2.destroyAllWindows()
vs.stop()