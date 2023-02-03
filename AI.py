from datetime import datetime
import face_recognition
import numpy
import cv2
import os


class FacialRecognition:
    def __init__(self):
        self.images = []
        self.class_names = []
        self.list_of_names = os.listdir("images")

        self.process()

    def find_image_encodings(self, images):
        list_of_encodings = []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(image)[0]
            list_of_encodings.append(encoding)
        return list_of_encodings

    def images_for_known_faces(self):
        for cls in self.list_of_names:
            current_image = cv2.imread("images/{}".format(cls))
            self.images.append(current_image)
            self.class_names.append(cls.split('.')[0])

    def mark_attendance(self, full_name):
        with open("attendance.csv", "r+") as file:
            students = file.readlines()
            full_names = []
            for student in students:
                student_full_name = student.split(",")[0]
                full_names.append(student_full_name)
            if full_name not in full_names:
                time_now = datetime.now().strftime("%H:%M:%S")
                file.writelines("\n{},{}".format(full_name, time_now))

    def process(self):
        self.images_for_known_faces()

        encodings_for_known_faces = self.find_image_encodings(self.images)
        print("Image Encoding Completed.")

        video_capture = cv2.VideoCapture(0)

        while True:
            _, image = video_capture.read()

            current_frame_faces = face_recognition.face_locations(image)
            encodings_cff = face_recognition.face_encodings(
                image, current_frame_faces)

            for encodings_of_face, face_location in zip(encodings_cff,
                                                        current_frame_faces):
                matches = face_recognition.compare_faces(encodings_for_known_faces,
                                                         encodings_of_face)
                face_distance = face_recognition.face_distance(encodings_for_known_faces,
                                                               encodings_of_face)
                best_match = numpy.argmin(face_distance)

                if matches[best_match]:
                    full_name = self.class_names[best_match]
                    y_one, x_two, y_two, x_one = face_location
                    cv2.rectangle(image,
                                  (x_one, y_one),
                                  (x_two, y_two),
                                  (0, 255, 0),
                                  2)
                    cv2.putText(image,
                                full_name,
                                (x_one-30, y_two),
                                cv2.FONT_HERSHEY_DUPLEX,
                                1,
                                (255, 255, 255),
                                1)
                    self.mark_attendance(full_name)

                else:
                    y_one, x_two, y_two, x_one = face_location
                    cv2.rectangle(image,
                                  (x_one, y_one),
                                  (x_two, y_two),
                                  (0, 0, 255),
                                  2)
                    cv2.putText(image,
                                "Unknown",
                                (x_one-30, y_two),
                                cv2.FONT_HERSHEY_DUPLEX,
                                1,
                                (255, 255, 255),
                                1)

            cv2.imshow("Facial Recognition", image)
            key = cv2.waitKey(1)

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    FacialRecognition()
