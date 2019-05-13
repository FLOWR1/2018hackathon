import cv2
import numpy as np

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("car.mp4")

ret, frame1 = cap.read()
frame1 =  cv2.resize(frame1, (100, 50))
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=True)

while(1):
	ret, frame2 =cap.read()
	frame2 =  cv2.resize(frame2, (100, 50))
	next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
	#cv2.imshow('original',cv2.resize(next, (500, 300)))

	#next = fgbg.apply(next)
	#cv2.imshow('fgbg',cv2.resize(next, (500, 300)))

	flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	mag_mat = np.matrix(mag)
	mag_mat_mean = mag_mat.mean()
	print(mag_mat_mean)
	if mag_mat_mean > 1:
		cv2.imshow('original',cv2.resize(next, (500, 300)))

	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	
	cv2.imshow('frame2',cv2.resize(rgb,(500,300)))

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
	elif k == ord('s'):
		cv2.imwrite('opticalfb.png',frame2)
		cv2.imwrite('opticalhsv.png',rgb)

	prvs = next

cap.release()
cv2.destroyAllWindows()