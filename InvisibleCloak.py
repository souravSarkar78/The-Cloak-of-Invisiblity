import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)

# Store a single frame as background 
_, background = cap.read()
# Giving two second time delay between two frames to adjust the auto exposure of camera  
time.sleep(2)
_, background = cap.read()

#define all the kernels size  
open_kernel = np.ones((5,5),np.uint8)
close_kernel = np.ones((5,5),np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)

# Function for remove noise from mask 
def filter_mask(mask):

    close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    open_mask = cv2.morphologyEx(close_mask, cv2.MORPH_OPEN, open_kernel)

    dilation = cv2.dilate(open_mask, dilation_kernel, iterations= 1)

    return dilation


width = int(cap.get(3))
height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))   # Uncomment if you want to recorn videframes


while cap.isOpened():
    ret, frame = cap.read()  # Capture every frame

    # convert to hsv colorspace 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower bound and upper bound for Green color 
    lower_bound = np.array([50, 80, 50])     
    upper_bound = np.array([90, 255, 255])
    # Check this website to know more about how to find upper and lower bound of a colour
    # https://realpython.com/python-opencv-color-spaces/  
    

    # find the colors within the boundaries
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Remome unnecessary noise from the mask
    mask = filter_mask(mask)

    # Apply the mask to take only those region from the saved background 
    # where our cloak is present in the current frame
    cloak = cv2.bitwise_and(background, background, mask=mask)

    # create inverse mask 
    inverse_mask = cv2.bitwise_not(mask)  

    # Apply the inverse mask to take those region of the current frame where cloak is not present 
    current_background = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    # Combine cloak region and current_background region to get final frame 
    combined = cv2.add(cloak, current_background)

    # out.write(combined) # Uncomment if you want to recorn videframes


    # Finally show the output frame
    cv2.imshow("cloak", combined)
    if cv2.waitKey(1) == ord('q'):
        cap.release()

    elif cv2.waitKey(1) == ord('p'):
        cv2.imwrite("background.png", background)
        cap.release()


# out.release() # Uncomment if you want to recorn videframes
cv2.destroyAllWindows()

