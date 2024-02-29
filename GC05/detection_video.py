import cv2
from mtcnn import MTCNN

# Initialize the video capture from the camera (use 0 for the default camera)
video_capture = cv2.VideoCapture(0)

# Define the codec and create a VideoWriter object (for saving the output video)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
output_video_path = 'output_video.mp4'
output_video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (640, 480))  # Output video resolution (adjust as needed)

# Initialize the MTCNN model
detector = MTCNN()

while True:
    ret, frame = video_capture.read()  # Read a frame from the camera

    if not ret:
        break

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    # Draw rectangles around detected faces
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)  # Red color

    # Write the frame to the output video
    output_video.write(frame)

    # Display the original frame
    cv2.imshow('Camera Feed', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer and video capture
output_video.release()
video_capture.release()

# Close all windows
cv2.destroyAllWindows()
