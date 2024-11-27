import cv2
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
import time

cap = cv2.VideoCapture(0)

### REDUCING THE SIZE OF THE FRAME to increase the frame rate
resize_factor = 1 # 0.8

prev_time = 0.00
while(True):
    current_time = time.time()

    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)

    # Capture an image
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  
        break
    elif key == ord('c'):  
        cv2.imwrite('Assignment_2/image.jpg', frame)
        print("Image captured")
    img = cv2.imread('Assignment_2/image.jpg', cv2.IMREAD_GRAYSCALE)
    
    ### MODIFING CANNY EDGE DETECTION PARAMETERS to increase the frame rate
    edges = cv2.Canny(frame, 100, 200, apertureSize=7, L2gradient=True) # apertureSize=7: larger kernel results in smoother edges and better suppression of noise but can miss finer details, L2Gradient=True: More precise and often results in better-looking edges, but it's computationally more expensive

    edge_pixels = np.column_stack(np.where(edges > 0))  

    ### USING ONLY A SUBSET OF EDGE PIXELS FOR LINE FITTING i.e. 90% of the edge pixels
    percentage = 0.9
    edge_pixels = edge_pixels[:int(len(edge_pixels)*percentage)]  

    x = edge_pixels[:, 1].reshape(-1, 1)  
    y = edge_pixels[:, 0] 
    
    if len(edge_pixels) < 2:
        continue

    ransac = RANSACRegressor()
    ransac.fit(x, y)

    line_x = np.linspace(x.min(), x.max(),100).reshape(-1, 1) 
    line_y = ransac.predict(line_x)  

    # Define the line starting and ending points
    start_point = (int(line_x[0][0]), int(line_y[0]))
    end_point = (int(line_x[1][0]), int(line_y[1]))

    # Drawing the line on the frame
    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    for i in range(len(line_x) - 1):
        cv2.line(frame, (int(line_x[i]), int(line_y[i])), (int(line_x[i + 1]), int(line_y[i + 1])), (255, 0, 0), 2)
    
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
    # print(f"FPS: {fps:.2f}")

    cv2.imshow('frame_with_line', frame)


cap.release()
cv2.destroyAllWindows()

