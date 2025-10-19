import cv2

from explore.utils.utils import extract_ball_from_img

img = cv2.imread("experiments/wrist_img.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

circle_data, mask = extract_ball_from_img(img, verbose=2)
for v in mask:
    if sum(v):
        print(v)
        break
