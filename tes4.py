import cv2
import time
import numpy as np
from rvs_palmvision import Palmvision

# Constants
chessboard_size = 8

# Function to create the translucent chessboard
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

# Function to add chess pieces to the chessboard
def add_chess_pieces(chessboard, square_size):
    for i in range(chessboard_size):
        for j in range(chessboard_size):
            if (i + j) % 2 != 0:
                piece = None
                if i == 1 or i == 6:
                    piece = cv2.imread('pawn.png')
                elif i == 0 or i == 7:
                    if j == 0 or j == 7:
                        piece = cv2.imread('rook.png')
                    elif j == 1 or j == 6:
                        piece = cv2.imread('knight.png')
                    elif j == 2 or j == 5:
                        piece = cv2.imread('bishop.png')
                    elif j == 3:
                        piece = cv2.imread('queen.png')
                    elif j == 4:
                        piece = cv2.imread('king.png')

                if piece is not None:
                    resized_piece = cv2.resize(piece, (square_size, square_size))
                    x_offset = j * square_size
                    y_offset = i * square_size
                    chessboard[y_offset:y_offset+square_size, x_offset:x_offset+square_size] = resized_piece

    return chessboard

# Main function
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    detector = Palmvision()

    selected_piece = None
    selected_square = None

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
                
                if selected_piece:
                    # Move the selected piece to the selected square
                    # This is where you would update the board state and check for valid moves
                    # For simplicity, let's just print the selected piece and square
                    print("Move:", selected_piece, "->", selected_square)
                    selected_piece = None
                else:
                    # Select the piece at the selected square
                    # For simplicity, let's just print the selected square
                    selected_piece = selected_square
                    print("Selected Square:", selected_square)

        # Draw chessboard
        chessboard_height = min(img.shape[0], img.shape[1])
        square_size = chessboard_height // chessboard_size
        chessboard_width = square_size * chessboard_size
        translucent_chessboard = create_translucent_chessboard(chessboard_size, square_size)
        chessboard_with_pieces = add_chess_pieces(translucent_chessboard, square_size)
        resized_chessboard_with_pieces = cv2.resize(chessboard_with_pieces, (chessboard_width, chessboard_height))
        img[:chessboard_height, :chessboard_width] = cv2.addWeighted(img[:chessboard_height, :chessboard_width], 1, resized_chessboard_with_pieces, 0.5, 0)

        # Highlight selected square
        if selected_square is not None:
            x1, y1 = selected_square[0] * square_size, selected_square[1] * square_size
            x2, y2 = (selected_square[0] + 1) * square_size, (selected_square[1] + 1) * square_size
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)

        cv2.imshow("Chess Game", img)
        cv2.waitKey(1)

# Run the main function
if __name__ == "__main__":
    main()
