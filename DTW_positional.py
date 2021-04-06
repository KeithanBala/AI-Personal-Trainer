import pandas as pd
import math
import numpy as np

from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import random

from operator import add 

df_expert = pd.read_csv('out_shoulderpress_expert.csv')
df_beginner = pd.read_csv('out_shoulderpress_mimic.csv')
df_bad = pd.read_csv('out_shoulderpress_bad.csv')

def getXY(data):
	x = []
	y = []
	for i in range(len(data)):
		x.append(float(data[i].split(' ')[0]))
		y.append(float(data[i].split(' ')[1]))

	return x, y

def angle3pt(ax,ay, bx,by, cx,cy):
    # Counterclockwise angle in degrees by turning from a to c around b Returns a float between 0.0 and 360.0
    ang = math.degrees(math.atan2(cy-by, cx-bx) - math.atan2(ay-by, ax-bx))
    #ang = ang + 360 if ang < 0 else ang

    return abs(ang)

#Right side of the body
rshoul_eX, rshoul_eY = getXY(df_expert['RShoulder']) 
relbow_eX, relbow_eY = getXY(df_expert['RElbow'])
rwrist_eX, rwrist_eY = getXY(df_expert['RWrist'])

rshoul_bX, rshoul_bY = getXY(df_beginner['RShoulder']) 
relbow_bX, relbow_bY = getXY(df_beginner['RElbow'])
rwrist_bX, rwrist_bY = getXY(df_beginner['RWrist'])

rshoul_bdX, rshoul_bdY = getXY(df_bad['RShoulder']) 
relbow_bdX, relbow_bdY = getXY(df_bad['RElbow'])
rwrist_bdX, rwrist_bdY = getXY(df_bad['RWrist'])

#Left side of the body
lshoul_eX, lshoul_eY = getXY(df_expert['LShoulder']) 
lelbow_eX, lelbow_eY = getXY(df_expert['LElbow'])
lwrist_eX, lwrist_eY = getXY(df_expert['LWrist'])

lshoul_bX, lshoul_bY = getXY(df_beginner['LShoulder']) 
lelbow_bX, lelbow_bY = getXY(df_beginner['LElbow'])
lwrist_bX, lwrist_bY = getXY(df_beginner['LWrist'])

lshoul_bdX, lshoul_bdY = getXY(df_bad['LShoulder']) 
lelbow_bdX, lelbow_bdY = getXY(df_bad['LElbow'])
lwrist_bdX, lwrist_bdY = getXY(df_bad['LWrist'])

rightArm_eangles = []
rightArm_bangles = []
rightArm_bdangles = []

leftArm_eangles = []
leftArm_bangles = []
leftArm_bdangles = []

for i in range(len(rshoul_eY)):
	rightArm_eangles.append(angle3pt(rshoul_eX[i],rshoul_eY[i], relbow_eX[i], relbow_eY[i], rwrist_eX[i], rwrist_eY[i]))

for i in range(len(rshoul_bY)):
	rightArm_bangles.append(angle3pt(rshoul_bX[i],rshoul_bY[i], relbow_bX[i], relbow_bY[i], rwrist_bX[i], rwrist_bY[i]))

for i in range(len(rshoul_bdY)):
	rightArm_bdangles.append(angle3pt(rshoul_bdX[i],rshoul_bdY[i], relbow_bdX[i], relbow_bdY[i], rwrist_bdX[i], rwrist_bdY[i]))

for i in range(len(lshoul_eY)):
	leftArm_eangles.append(angle3pt(lshoul_eX[i],lshoul_eY[i], lelbow_eX[i], lelbow_eY[i], lwrist_eX[i], lwrist_eY[i]))

for i in range(len(lshoul_bY)):
	leftArm_bangles.append(angle3pt(lshoul_bX[i],lshoul_bY[i], lelbow_bX[i], lelbow_bY[i], lwrist_bX[i], lwrist_bY[i]))

for i in range(len(lshoul_bdY)):
	leftArm_bdangles.append(angle3pt(lshoul_bdX[i],lshoul_bdY[i], lelbow_bdX[i], lelbow_bdY[i], lwrist_bdX[i], lwrist_bdY[i]))

#Angular difference between expert and user
leftArm_b = np.asarray(leftArm_bangles, dtype = np.float32)
leftArm_bd = np.asarray(leftArm_bdangles, dtype = np.float32)
leftArm_e = np.asarray(leftArm_eangles, dtype = np.float32)


'''
#X-positional difference between expert and user
bx_list = [sum(i) for i in zip(rshoul_bX, relbow_bX, rwrist_bX)]
bdx_list = [sum(i) for i in zip(rshoul_bdX, relbow_bdX, rwrist_bdX)]
ex_list = [sum(i) for i in zip(rshoul_eX, relbow_eX, rwrist_eX)]

reftArm_bX = np.asarray(bx_list, dtype = np.float32)
reftArm_bdX = np.asarray(bdx_list, dtype = np.float32)
reftArm_eX = np.asarray(ex_list, dtype = np.float32)

#Y-positional difference between expert and user
by_list = [sum(i) for i in zip(rshoul_bY, relbow_bY, rwrist_bY)]
bdy_list = [sum(i) for i in zip(rshoul_bdY, relbow_bdY, rwrist_bdY)]
ey_list = [sum(i) for i in zip(rshoul_eY, relbow_eY, rwrist_eY)]

reftArm_bY = np.asarray(by_list, dtype = np.float32)
reftArm_bdY = np.asarray(bdy_list, dtype = np.float32)
reftArm_eY = np.asarray(ey_list, dtype = np.float32)

bxy = np.asarray(list(map(add, reftArm_bX, reftArm_bY)) , dtype = np.float32)
bdxy = np.asarray(list(map(add, reftArm_bdX, reftArm_bdY)) , dtype = np.float32)
exy = np.asarray(list(map(add, reftArm_eX, reftArm_eY)) , dtype = np.float32)
'''

#X postion for both arms bad
bdxR_list = [sum(i) for i in zip(rshoul_bdX, relbow_bdX, rwrist_bdX)]
bdxL_list = [sum(i) for i in zip(lshoul_bdX, lelbow_bdX, lwrist_bdX)]

#Y postion for both arms bad
bdyR_list = [sum(i) for i in zip(rshoul_bdY, relbow_bdY, rwrist_bdY)]
bdyL_list = [sum(i) for i in zip(lshoul_bdY, lelbow_bdY, lwrist_bdY)]


#X postion for both arms beginner
bxR_list = [sum(i) for i in zip(rshoul_bX, relbow_bX, rwrist_bX)]
bxL_list = [sum(i) for i in zip(lshoul_bX, lelbow_bX, lwrist_bX)]

#Y postion for both arms beginner
byR_list = [sum(i) for i in zip(rshoul_bY, relbow_bY, rwrist_bY)]
byL_list = [sum(i) for i in zip(lshoul_bY, lelbow_bY, lwrist_bY)]


#X postion for both arms expert
exR_list = [sum(i) for i in zip(rshoul_eX, relbow_eX, rwrist_eX)]
exL_list = [sum(i) for i in zip(lshoul_eX, lelbow_eX, lwrist_eX)]

#Y postion for both arms expert
eyR_list = [sum(i) for i in zip(rshoul_eY, relbow_eY, rwrist_eY)]
eyL_list = [sum(i) for i in zip(lshoul_eY, lelbow_eY, lwrist_eY)]


bdxyR = [sum(i) for i in zip(bdxR_list, bdyR_list)]
bdxyL = [sum(i) for i in zip(bdxL_list, bdyL_list)]
bxyR = [sum(i) for i in zip(bxR_list, byR_list)]
bxyL = [sum(i) for i in zip(bxL_list, byL_list)]
exyR = [sum(i) for i in zip(exR_list, eyR_list)]
exyL = [sum(i) for i in zip(exL_list, eyL_list)]


bdxy = np.asarray(list(map(add, bdxyR, bdxyL)) , dtype = np.float32)
bxy = np.asarray(list(map(add, bxyR, bxyL)) , dtype = np.float32)
exy = np.asarray(list(map(add, exyR, exyL)) , dtype = np.float32)

#Using point DTW
distanceRshoulX = dtw.distance(rshoul_bX[:250], rshoul_eX[:250])
distanceRshoulY = dtw.distance(rshoul_bY[:250], rshoul_eY[:250])

distanceRelbowX = dtw.distance(relbow_bX[:250], relbow_eX[:250])
distanceRelbowY = dtw.distance(relbow_bY[:250], relbow_eY[:250])

distanceRwristX = dtw.distance(rwrist_bX[:250], rwrist_eX[:250])
distanceRwristY = dtw.distance(rwrist_bY[:250], rwrist_eY[:250])

distance = distanceRshoulX+distanceRshoulY +distanceRelbowX+distanceRelbowY +distanceRwristX+distanceRwristY
distX = distanceRshoulX+distanceRelbowX+distanceRwristX
distY = distanceRshoulY+distanceRelbowY+distanceRwristY
print(distance)

print(dtw.distance(bdxy,exy))

d, paths = dtw.warping_paths(bdxy, exy)
best_path = dtw.best_path(paths)

dtwvis.plot_warping(bdxy, exy, best_path, filename = "Graphs/test_point_badX.png")
#print(paths)

dtwvis.plot_warpingpaths(bdxy, exy, paths, path = best_path, filename = "Graphs/testXX.png") 