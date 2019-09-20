import cv2
import os

cap = cv2.VideoCapture('test.mp4')
ret, current_frame = cap.read()
previous_frame = current_frame
count = 0
while(cap.isOpened()):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    

    frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)

    # cv2.imshow('frame diff ',frame_diff)      
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    cv2.imwrite(os.path.join('./code/framediff','%.6d.jpg'%count),frame_diff)
    
    previous_frame = current_frame.copy()
    ret, current_frame = cap.read()
    count += 1
cap.release()
cv2.destroyAllWindows()
