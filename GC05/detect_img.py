import cv2
from mtcnn import MTCNN

# Load the image
image = cv2.imread('45.png')

# Initialize the MTCNN model
detector = MTCNN()

# Detect faces in the image
faces = detector.detect_faces(image)

# Draw rectangles around detected faces
for face in faces:
    x, y, width, height = face['box']
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

# Resize the image to 1080p resolution
new_width, new_height = 1080, 720
image = cv2.resize(image, (new_width, new_height))

# Display or save the image
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# To save the image with rectangles:
cv2.imwrite('output6666.jpg', image)
