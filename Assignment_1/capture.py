import cv2
import time
import numpy as np

# 0 is the first camera (PC), 1 is the second camera (IVCam)
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1)

prev_time = 0.00
while(True):
    current_time = time.time()

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Brightest spot in the grayscale image
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
    cv2.circle(frame, max_loc, 10, (0, 255, 255), 2) # Yellow circle

    # REDDEST spot in the image using minMaxLoc()
    # I calculate the red_score for each pixel substracting the maximum of green and blue values from red to ensure the maximum value of only the red pixels
    b, g, r = cv2.split(frame)
    red_score = r.astype(np.int16) - np.maximum(g, b).astype(np.int16)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(red_score)

    # REDDEST spot in the image using two loops
    # reddest_score = float('-inf')
    # max_loc = (0, 0)
    # rows, cols, _ = frame.shape
    # for i in range(rows):
    #     for j in range(cols):
    #         b, g, r = frame[i, j]
    #         red_score = int(r) - max(int(g), int(b))
    #         if red_score > reddest_score:
    #             reddest_score = red_score
    #             max_loc = (j, i)

    cv2.circle(frame, max_loc, 10, (0, 0, 255), 2)  # Red circle

    # Calculate FPS and display it
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255), 
        2,
        cv2.LINE_AA,
    )

    # Get the time it takes to process the frame
    # print((1/fps)*1000 if fps else 0)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Get the latency between the image capture time and the display time
    # print(time.time() - current_time)

cap.release()
cv2.destroyAllWindows()