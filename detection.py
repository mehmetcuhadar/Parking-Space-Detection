import cv2
import torch
import numpy as np
from shapely.geometry import Point, Polygon
from sklearn.metrics import accuracy_score

file = "romania" 
ground_truth = {"romania" : [True, True, True, True, True, True, True, True, True, True, True ,True],
                 "sea" : [True, False, False, False, False, False, False, False, False, False, False, False , True, True, False, True],
                 "highway" : [False, True, False, True, True, False, False, True, True, False, False, False, False, True ]}

def readCords(filename):
    lines = []
    with open(filename) as f:
        lines = f.readlines()
        f.close()
    result = []
    for line in lines:
        arr = [int(el) for el in line.split(",")]
        p1 = arr[0:2]
        p2 = arr[2:4]
        p3 = arr[4:6]
        p4 = arr[6:8]
        result.append([[p1,p2,p3,p4], False])
    return result

if __name__ == "__main__":
    parking_cords = readCords(file + ".txt") #Reads coords of parking spots
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s') #Loads model yolov5
    cap = cv2.VideoCapture(file + ".mp4") #Opens video

    ### Evaluation metrics True Pos, True Neg, False Pos, False Neg ###
    tp = 0 
    tn = 0
    fp = 0
    fn = 0

    while True:
        ret, frame = cap.read()
        if ret == False : exit()
        ### Detection of Vehicles ###
        results = model(frame)
        for p in parking_cords:
            overlay = frame.copy()
            color = (0, 255, 0) if not p[1] else (0, 0, 255) 
            cv2.fillPoly(overlay, [np.array(p[0], np.int32).reshape((-1, 1, 2))], color)
            p[1] = False
            alpha = 0.5
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        res = results.pandas().xyxy[0].to_dict("records")

        ### Detection of Parking Spaces ###
        for el in res:
            if el["confidence"] > 0.4:
                x1 = el["xmin"]
                y1 = el["ymin"]
                x2 = el["xmax"]
                y2 = el["ymax"]
            center = Point(((x1 + x2) / 2 ) , ((y1 + y2) / 2 ) )
            if (el["name"] != "car" and el["name"] != "truck"): continue
            for p in parking_cords:
                poly = Polygon(p[0])
                if poly.contains(center):
                    p[1] = True
            start_point = (int(x1), int(y1))
            end_point = (int(x2), int(y2))
            alpha = 0.4
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
            #cv2.putText(frame, el["name"], start_point, cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2)
        
        ### Rest of the code for evaluation ###
        for i, val in enumerate(parking_cords):  
            if ground_truth[file][i]:
                if val[1] == ground_truth[file][i]:
                    tp += 1
                else:
                    fn += 1
            else:
                if val[1] == ground_truth[file][i]:
                    tn += 1
                else:
                    fp += 1
        
        acc = 0 if (tp + tn + fp + fn) == 0 else (tp + tn)/ (tp + tn + fp + fn)
        pre = 0 if (tp + fp) == 0 else tp / (tp + fp)
        rec = 0 if (tp + fn) == 0 else tp / (tp + fn)

        print("Accuracy : " , acc)
        print("Precision : " , pre)
        print("Recall : " , rec)

        cv2.imshow("res", frame)
        if cv2.waitKey(1) == 27: # use ESC to quit
            break
        
    