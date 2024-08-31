import cv2

# Function to detect chessboard corners
def detect_chessboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    return ret, corners

# Main function
def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        ret, corners = detect_chessboard(frame)
        if ret:
            cv2.drawChessboardCorners(frame, (7, 7), corners, True)

        cv2.imshow('Chessboard Detection', frame)

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
