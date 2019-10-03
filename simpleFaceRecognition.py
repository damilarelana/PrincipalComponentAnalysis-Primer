from PIL import Image
import face_recognition

test_image = face_recognition.load_image_file("./data/stock_people.jpg")
face_locations = face_recognition.face_locations(test_image)
print("The image has {} faces".format(len(face_locations)))

for face_location in face_locations:
    top, right, bottom, left = face_location
    print("a face was detected at Top:{}, Left:{}, Bottom:{}, Right:{}".format(top, left, bottom, right))
    actual_face = test_image[top:bottom, left:right]
    pil_image = Image.fromarray(actual_face)
    pil_image.show()
