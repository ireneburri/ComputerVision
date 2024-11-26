import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
import time

cap = cv2.VideoCapture(0)

### REDUSING THE SIZE OF THE FRAME to increase the frame rate
resize_factor = 1 #0.5

prev_time = 0.00
while(True):
    current_time = time.time()

    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  
        break

    # Capture an image
    elif key == ord('c'):  
        cv2.imwrite('Assignment_2/image.jpg', frame)
        print("Image captured")
    img = cv2.imread('Assignment_2/image.jpg', cv2.IMREAD_GRAYSCALE)
    

    ### MODIFING CANNY EDGE DETECTION PARAMETERS
    # edges = cv2.Canny(frame,100,200)
    edges = cv2.Canny(frame, 50, 150)


    edge_pixels = np.column_stack(np.where(edges > 0))  # (row, col) -> (y, x)
    # print(f"Number of edge pixels: {len(edge_pixels)}")

    ### USING ONLY A SUBSET OF EDGE PIXELS FOR LINE FITTING i.e. 90% of the edge pixels
    percentage = 0.9
    edge_pixels = edge_pixels[:int(len(edge_pixels)*percentage)]  

    x = edge_pixels[:, 1].reshape(-1, 1)  
    y = edge_pixels[:, 0] 

    # print(f"Edge pixel coordinates: \n{edge_pixels}")
    
    if len(edge_pixels) < 2:
        continue

    ransac = RANSACRegressor()
    ransac.fit(x, y)

    # Get the fitted line
    line_x = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)  # Generate x values for plotting
    line_y = ransac.predict(line_x)  # Predict y values

    # Identify inliers and outliers
    inliers = ransac.inlier_mask_  # Boolean mask for inliers
    outliers = np.logical_not(inliers)  # Boolean mask for outliers

    # Draw the fitted line on the frame
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

    cv2.imshow('frame_with_line', frame)

    # Plot results
    # plt.figure(figsize=(10, 6))
    # plt.imshow(edges, cmap='gray')
    # plt.scatter(x[inliers], y[inliers], color='green', marker='.', label='Inliers', s=1)
    # plt.scatter(x[outliers], y[outliers], color='red', marker='.', label='Outliers', s=1)
    # plt.plot(line_x, line_y, color='blue', linewidth=2, label='RANSAC Fit')
    # plt.title('Edge Detection with RANSAC Line Fit')
    # plt.legend()
    # plt.show()


cap.release()
cv2.destroyAllWindows()