import cv2
import numpy as np

# Function to create a virtual chessboard
def create_chessboard(size, square_size):
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

    return chessboard

# Main function
def main():
    size = 8  # Size of the chessboard
    square_size = 50  # Size of each square in pixels
    chessboard = create_chessboard(size, square_size)

    cv2.imshow('Virtual Chessboard', chessboard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
