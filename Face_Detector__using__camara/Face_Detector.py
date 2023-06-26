import cv2

# Load the pre-trained data on face frontals from OpenCV
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the default camera (usually the webcam)
video_capture = cv2.VideoCapture(0)

while True:
    # Read the current frame from the video stream
    _, frame = video_capture.read()

    # Convert the frame to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    face_coordinates = trained_face_data.detectMultiScale(gray_image)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with face detections
    cv2.imshow('Real-time Face Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
video_capture.release()
cv2.destroyAllWindows()

print("Code Completed")
