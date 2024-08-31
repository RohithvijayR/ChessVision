import cv2
import time
import numpy as np
from rvs_palmvision import Palmvision  # Importing Palmvision module

def create_translucent_chessboard(size, square_size, alpha=0.5):
    chessboard = np.zeros((size * square_size, size * square_size, 3), dtype=np.uint8)
    color1 = (255, 255, 255)  # White
    color2 = (0, 0, 0)  # Black

    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                color = color1
            else:
                color = color2
            cv2.rectangle(chessboard, (i * square_size, j * square_size),
                          ((i + 1) * square_size, (j + 1) * square_size), color, -1)

    # Make the chessboard translucent
    chessboard = cv2.addWeighted(chessboard, alpha, np.zeros_like(chessboard), 1 - alpha, 0)

    return chessboard

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    detector = Palmvision()
    chessboard_size = 8
    selected_box = None

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)  # Horizontally flip the camera frame
        
        # Find hand landmarks using Palmvision
        img = detector.findpalm(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            index_finger_tip = lmList[8]  # Assuming the index finger tip is at position 8
            x, y = index_finger_tip[1], index_finger_tip[2]
            
            # Map finger tip coordinates to chessboard coordinates
            col = int(x / square_size)
            row = int(y / square_size)
            
            # Limit the coordinates to the chessboard size
            col = max(0, min(col, chessboard_size - 1))
            row = max(0, min(row, chessboard_size - 1))
            
            # Determine the selected box
            selected_box = (col, row)
            
            print(selected_box)

        # Determine the dimensions of the resized chessboard
        chessboard_height = min(img.shape[0], img.shape[1])  # Choose the minimum dimension as the size
        square_size = chessboard_height // chessboard_size
        chessboard_width = square_size * chessboard_size

        # Create a translucent chessboard
        translucent_chessboard = create_translucent_chessboard(chessboard_size, square_size)

        # Resize the chessboard to match the dimensions of the camera frame
        resized_chessboard = cv2.resize(translucent_chessboard, (chessboard_width, chessboard_height))

        # Overlay the resized chessboard on the camera feed
        img[:chessboard_height, :chessboard_width] = cv2.addWeighted(img[:chessboard_height, :chessboard_width], 1, resized_chessboard, 0.5, 0)

        # Highlight the selected box
        if selected_box is not None:
            x1, y1 = selected_box[0] * square_size, selected_box[1] * square_size
            x2, y2 = (selected_box[0] + 1) * square_size, (selected_box[1] + 1) * square_size
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
