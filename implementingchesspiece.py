import cv2
import time
import numpy as np
from rvs_palmvision import Palmvision

chessboard_size = 8

def create_translucent_chessboard(size, square_size, alpha=0.5):
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

    selected_piece = None
    selected_square = None
    piece_moved = False

    # Define initial chessboard setup
    board = [['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
             ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
             [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
             [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
             [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
             [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
             ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
             ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']]

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        
        img = detector.findpalm(img)
        lmList = detector.findPosition(img)

        if len(lmList) >= 2:
            index_finger_tip = lmList[8]
            thumb_finger_tip = lmList[4]
            distance = np.linalg.norm(np.array(index_finger_tip[1:]) - np.array(thumb_finger_tip[1:]))

            if distance < 20:
                x, y = index_finger_tip[1], index_finger_tip[2]
                col = int(x / square_size)
                row = int(y / square_size)
                col = max(0, min(col, chessboard_size - 1))
                row = max(0, min(row, chessboard_size - 1))
                selected_square = (col, row)
                
                if not piece_moved:
                    selected_piece = board[row][col]
                    print("Selected Piece:", selected_piece)

                else:
                    print("Move:", selected_piece, "->", selected_square)
                    # Update board with the new position of the piece
                    board[selected_square[1]][selected_square[0]] = selected_piece
                    piece_moved = False

        chessboard_height = min(img.shape[0], img.shape[1])
        square_size = chessboard_height // chessboard_size
        chessboard_width = square_size * chessboard_size
        translucent_chessboard = create_translucent_chessboard(chessboard_size, square_size)
        resized_chessboard = cv2.resize(translucent_chessboard, (chessboard_width, chessboard_height))
        img[:chessboard_height, :chessboard_width] = cv2.addWeighted(img[:chessboard_height, :chessboard_width], 1, resized_chessboard, 0.5, 0)

        if selected_square is not None:
            x1, y1 = selected_square[0] * square_size, selected_square[1] * square_size
            x2, y2 = (selected_square[0] + 1) * square_size, (selected_square[1] + 1) * square_size
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            piece_moved = True

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)

        cv2.imshow("Chess Game", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
