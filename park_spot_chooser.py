import cv2
import numpy as np
cord = []
pts = []
accepted_pts = []
file = "highway"

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        assert len(cord) != 8
        pts.append([x, y])
        cord.append(str(x))
        cord.append(str(y))
        print("arr:", cord)
        print(pts)

if __name__ == "__main__":
    
    
    while 1:
        img = cv2.imread(file + ".jpg", 1)
        for p in accepted_pts:
            overlay = img.copy()
            cv2.fillPoly(overlay, [np.array(p, np.int32).reshape((-1, 1, 2))], (0, 255, 0))
            alpha = 0.5
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        isClosed = True if len(pts) == 4 else False
        cv2.polylines(img, [np.array(pts, np.int32).reshape((-1, 1, 2))], isClosed, (255, 0, 0), 2)
        cv2.imshow("image", img)
        cv2.setMouseCallback("image",click_event)
        
        if cv2.waitKey(1) == 13: #Enter
            print(cord)
            if len(cord) != 8: continue
            with open(file + '.txt', 'a') as f:
                cords = ','.join(cord)
                f.write(cords + "\n")
                f.close()
            cord = []
            accepted_pts.append(pts)
            pts = []

        if cv2.waitKey(1) == 255: #Del
            pts = []
            cord = []

        if cv2.waitKey(1) == 27: #ESC
            exit()