from PIL import Image
import face_recognition

# load the image to be used as the base to identify faces
test_image = face_recognition.load_image_file("./data/stock_people.jpg")
face_locations = face_recognition.face_locations(test_image)
print("The image has {} faces".format(len(face_locations)))

# identify and locate the face of the black woman
black_wowan_image = face_recognition.load_image_file("./data/black_woman.jpg")
test_image_encoded = face_recognition.face_encodings(test_image)[2]  # encode the base image i.e. assuming the black woman is at index `2`
black_wowan_image_encoded = face_recognition.face_encodings(black_wowan_image)[0] # encode the image we seek to find
search_results = face_recognition.compare_faces([test_image_encoded], black_wowan_image_encoded)
print(search_results)

# identify and print all the faces
for face_location in face_locations:
    top, right, bottom, left = face_location
    print("a face was detected at Top:{}, Left:{}, Bottom:{}, Right:{}".format(top, left, bottom, right))
    actual_face = test_image[top:bottom, left:right]
    pil_image = Image.fromarray(actual_face)
    pil_image.show()