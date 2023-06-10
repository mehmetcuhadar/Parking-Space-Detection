import cv2
def main():
    file = "highway"
    video = cv2.VideoCapture(file + ".mp4")
    path = file + ".jpg"
    while 1:
        not_finished, frame = video.read()
        if not_finished:
            cv2.imwrite(path, frame)
            exit() 

if __name__ == "__main__":
    main()