import cv2


# lode the pre-trained data on face frontals form opencv
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect face in
image=cv2.imread('rdj.jpg')

# convert image back and white, becoure a.i can detect ecely black and white images
# Must convert to grayscale
gray__image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect Face
face_coordinates=trained_face_data.detectMultiScale(gray__image)
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0),2)

# print(face_coordinates)

# show image
cv2.imshow('show the image : ',image)
cv2.waitKey()

print("Code Completed")