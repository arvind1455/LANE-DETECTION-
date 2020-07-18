import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

cap = cv2.VideoCapture('project_video.mp4')
out = cv2.VideoWriter('lane.mp4', -1, 30, (1280, 1440))

####STEP 1:
def undistort(img):
    #Camera Matrix
    K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
        [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    #Distortion Coefficients
    dist = np.array([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05,
        2.20573263e-02]])
    undistorted_image = cv2.undistort(img, K, dist, None, K)
    return undistorted_image

####STEP 2:
def laneCurve(img):
    h, w, c = img.shape
    pts1 = np.array([[534, 472], [750, 461],[220, 684], [1200,645]]).reshape(-1,1,2)
    #print(pts1)
    pts2 = np.float32([[0,0], [w,0], [0, h], [w, h]]).reshape(-1,1,2)
    Mtx, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 10)
    warp = cv2.warpPerspective(img, Mtx, (w,h))
    #imgWarp = utils.WarpImg(img, points, w, h)
    warp = cv2.GaussianBlur(warp, (9,9), 0)
    return warp

####STEP 3:
def findEdges(img):
    #convert image from BGR to HLS colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    #WHITER COLOR
    lower1 = np.array([0, 195, 0])
    upper1 = np.array([255, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    #YELLOW COLOR
    lower2 = np.array([0, 50, 100])
    upper2 = np.array([255, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)
    #RESULTANT MASK
    result = cv2.bitwise_or(mask1, mask2)
    return result

####STEP 4:
def threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (9,9), 0)
    return blur

####STEP 5:
def line_fitting(img):
    #HISTOGRAM
    histogram = np.sum(img[img.shape[0]//2:, :], axis=0)
    out_img = np.dstack((img, img, img))
    midpoint = np.int(histogram.shape[0]//2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    #IDENTIFY THE NON ZERO VALUES
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    #NUMBER OF WINDOWS AND LINE FITTING
    nwindows = 30
    min = 50
    margin = 100
    height = np.int(img.shape[0]/nwindows)
    left_lane_inds = []
    right_lane_inds = []
    left_current = left_base
    right_current = right_base

    for i in range(nwindows):
        wind_y_low = img.shape[0] - (i+1)*height
        wind_y_high = img.shape[0] - i*height
        win_left_low = left_current - margin
        win_left_high = left_current + margin
        win_right_low = right_current - margin
        win_right_high = right_current + margin 

        #VISUALIZE THE RECTANGLES
        cv2.rectangle(out_img, (win_left_low, wind_y_low), (win_left_high, wind_y_high), (0,255,0), 2)
        cv2.rectangle(out_img, (win_right_low, wind_y_low), (win_right_high, wind_y_high), (0,255,0), 2)

        #FIND THE NON ZERO SPOTS ON THE IMAGE
        good_left_inds = ((nonzeroy >= wind_y_low) & (nonzeroy < wind_y_high) & 
        (nonzerox >= win_left_low) &  (nonzerox < win_left_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= wind_y_low) & (nonzeroy < wind_y_high) & 
        (nonzerox >= win_right_low) &  (nonzerox < win_right_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > min:
            left_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > min:
            right_current = np.int(np.mean(nonzerox[good_right_inds]))
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    #POLYFIT TO FIND THE POINTS OF THE LINE
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    #FIND THE EQUATION OF THE CURVE OF THE LEFT AND RIGHT LINES
    ploty = np.linspace(0, img.shape[0]-1, num = img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    #FIND THE POINTS OF THE LINES FOUND
    left_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_points = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    points = np.hstack((left_points, right_points))

    #CREATE AN EMPTY MASK
    warp = np.zeros_like(img).astype(np.uint8)
    color = np.dstack((img, img, img))*255
    
    #DRAW LINES AND FILL
    cv2.polylines(out_img, np.int32([left_points]), False, (255,0,255), 10)
    cv2.polylines(out_img, np.int32([right_points]), False, (255,0,255), 10)
    cv2.fillPoly(color, np.int_([points]), (255, 0, 255))

    return out_img, color, left_fit, right_fit, ploty

####STEP 6:
def radius(img, left_fit, right_fit, ploty):
    y = np.max(ploty)
    left = 1+ 3.5*left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
    right = 1+ 3.5*right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]   
    actual_position = (left + right)//2
    cv2.circle(lane, (int(actual_position), lane.shape[0]),20, (0,255,0), cv2.FILLED)
    position = img.shape[1]//2
    distance = position - actual_position
    return distance

####STEP 7:
def UnWarp(img):
    h, w, c = img.shape
    pts1 = np.array([[534, 472], [750, 461],[220, 684], [1200,645]]).reshape(-1,1,2)
    #print(pts1)
    pts2 = np.float32([[0,0], [w,0], [0, h], [w, h]]).reshape(-1,1,2)
    Mtx, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 10)
    unwarp = cv2.warpPerspective(img, Mtx, (w,h))
    return unwarp

while True:
    _, frame = cap.read()
#UNDISTORTION
    undistorted = undistort(frame)
#BIRD EYE
    warp = laneCurve(undistorted)
#EDGE DETECTION
    thresh1 = findEdges(warp)
    res = cv2.bitwise_and(warp, warp, mask= thresh1)
    edges = threshold(res)
#LINE FITTING
    rectangle, lane, left, right, y = line_fitting(edges)
    radius1 = radius(lane, left, right, y)
    unwarp = UnWarp(lane)

#FINAL IMAGE
    result = cv2.addWeighted(frame, 1, unwarp, 0.5, 1)
    #cv2.putText(result, str(radius1), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0 , 255), 2)
    if(radius1 >= -40 and radius1 <= 120):
        cv2.putText(result,"STRAIGHT  " + str(radius1) , (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255 , 0), 2)
    if(radius1 > 121 and radius1 < 400):
        cv2.putText(result,"LEFT  " + str(radius1) , (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255 , 255), 2)
    if(radius1 > -400 and radius1 < -41):
        cv2.putText(result,"RIGHT  " + str(radius1) , (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255 , 0), 2)

#OVERALL OUTPUT
    a = cv2.resize(frame, (640,480))
    b = cv2.resize(warp, (640,480))
    c = cv2.resize(edges, (640,480))
    d = cv2.resize(rectangle, (640,480))
    e = cv2.resize(lane, (640,480))
    f = cv2.resize(unwarp, (640,480))
    g = cv2.resize(result, (640,480))
    out1 = np.hstack((a, b))
    out2 = np.hstack((d, e))
    out3 = np.hstack((f, g))

    overall = np.vstack((out1, out2, out3))
    #print(overall.shape)
    #full = cv2.resize(overall, (1920,480))
#output
    #plt.plot(midpoint)
    out.write(overall)
    cv2.imshow('frame', overall)
    cv2.waitKey(1)
    plt.show()
