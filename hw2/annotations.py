import numpy as np
import cv2

COLORS = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 0), (0, 0, 255)]
def vis_annotations_q2a():
	'''
	annotations: (3, 2, 4) 
	3 groups of parallel lines, each group has 2 lines, each line annotated as (x1, y1, x2, y2)
	'''
	annotations = np.load('data/q2/q2a.npy')			
	img = cv2.imread('data/q2a.png')
	for i in range(annotations.shape[0]):
		COLOR = COLORS[i]
		lines = annotations[i]
		for j in range(lines.shape[0]):
			x1, y1, x2, y2 = lines[j]
			cv2.circle(img, (x1, y1), 3, COLOR, -1)
			cv2.circle(img, (x2, y2), 3, COLOR, -1)
			cv2.line(img, (x1, y1), (x2, y2), COLOR, 2)

	cv2.imshow('q2a', img)
	cv2.waitKey(0)

def vis_annnotations_q2b():
	'''
	annotations: (3, 4, 2) 
	3 squares, 4 points for each square, each point annotated as (x, y).
		pt0 - pt1
		|		|
		pt3 - pt2
	'''
	annotations = np.load('data/q2/q2b.npy').astype(np.int64)		
	img = cv2.imread('data/q2b.png')

	for i in range(annotations.shape[0]):
		COLOR = COLORS[i]
		square = annotations[i]
		for j in range(square.shape[0]):
			x, y = square[j]		 
			cv2.circle(img, (x, y), 3, COLOR, -1)
			cv2.putText(img, str(j), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, COLOR, 1, cv2.LINE_AA)
			cv2.line(img, square[j], square[(j+1) % 4], COLOR, 2)

	cv2.imshow('q2b', img)
	cv2.waitKey(0)

def vis_annotations_q3():
	'''
	annotations: (5, 4, 2)
	5 planes in the scene, 4 points for each plane, each point annotated as (x, y).
		pt0 - pt1
		|		|
		pt3 - pt2
	'''
	annotations = np.load('data/q3/q3.npy').astype(np.int64)		
	img = cv2.imread('data/q3.png')

	for i in range(annotations.shape[0]):
		COLOR = COLORS[i]
		square = annotations[i]
		for j in range(square.shape[0]):
			x, y = square[j]		 
			cv2.circle(img, (x, y), 3, COLOR, -1)
			cv2.putText(img, str(j+i*4), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, COLOR, 1, cv2.LINE_AA)
			cv2.line(img, square[j], square[(j+1) % 4], COLOR, 2)

		cv2.imshow('q3', img)
		cv2.waitKey(0)

if __name__ == '__main__':
	vis_annotations_q2a()
	vis_annnotations_q2b()
	vis_annotations_q3()