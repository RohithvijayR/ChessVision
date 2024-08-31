import cv2
import time
import numpy as np
from rvs_palmvision import Palmvision

def trans(size, square_size, alpha=0.5):
    chessboard = np.zeros((size * square_size, size * square_size, 3), dtype=np.uint8)
    color1 = (255, 255, 255)
    color2 = (0, 0, 0)
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                color = color1
            else:
                color = color2
            cv2.rectangle(chessboard, (i * square_size, j * square_size),
                          ((i + 1) * square_size, (j + 1) * square_size), color, -1)
    chessboard = cv2.addWeighted(chessboard, alpha, np.zeros_like(chessboard), 1 - alpha, 0)
    return chessboard

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    detector = Palmvision()
    chessbd = 8
    selected_box = None
    selection_active = False

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findpalm(img)
        llist = detector.findPosition(img)

        if len(llist) >= 2: 
            index_ft = llist[8]
            thumb_ft= llist[4] 
            distance = np.linalg.norm(np.array(index_ft[1:]) - np.array(thumb_ft[1:]))

            if distance < 20:
                x, y = index_ft[1], index_ft[2]
                col = int(x / square_size)
                row = int(y / square_size)
                col = max(0, min(col, chessbd - 1))
                row = max(0, min(row, chessbd - 1))
                selected_box = (col, row)
                selection_active = True
                
                print(selected_box)

            elif selection_active:
                selection_active = False
                print("Final Selection:", selected_box)
                selected_box = None

        chessboard_height = min(img.shape[0], img.shape[1]) 
        square_size = chessboard_height // chessbd
        chessboard_width = square_size * chessbd
        translucent_chessboard = trans(chessbd, square_size)
        resized_chessboard = cv2.resize(translucent_chessboard, (chessboard_width, chessboard_height))
        img[:chessboard_height, :chessboard_width] = cv2.addWeighted(img[:chessboard_height, :chessboard_width], 1, resized_chessboard, 0.5, 0)

        if selected_box is not None:
            x1, y1 = selected_box[0] * square_size, selected_box[1] * square_size
            x2, y2 = (selected_box[0] + 1) * square_size, (selected_box[1] + 1) * square_size
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
