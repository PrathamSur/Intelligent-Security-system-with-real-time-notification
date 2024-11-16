import cv2
import numpy as np
import smtplib
import os
from threading import Thread
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import time

# Global variables for email sending
last_sent_time = time.time() - 60  # Initialize to a time more than 60 seconds ago
detection_count_threshold = 5  # Number of detections needed to send an email


def send_email(frame):
    global last_sent_time
    current_time = time.time()

    # Only send email if 60 seconds have passed since the last email
    if current_time - last_sent_time >= 60:
        last_sent_time = current_time

        os.makedirs("./detected_img", exist_ok=True)
        ImgFileName = "./detected_img/img.jpg"
        cv2.imwrite(ImgFileName, frame)

        with open(ImgFileName, 'rb') as f:
            img_data = f.read()

        msg = MIMEMultipart()
        msg['Subject'] = 'Object Detected'
        msg['From'] = 'projectuseonly943@gmail.com'
        msg['To'] = 'yourgmail@gmail.com'        #provide your own gmail here

        text = MIMEText("Object Detected!!!")
        msg.attach(text)

        image = MIMEImage(img_data, name="DetectedObject.jpg")
        msg.attach(image)

        try:
            s = smtplib.SMTP('smtp.gmail.com', 587)
            s.starttls()
            s.login("projectuseonly943@gmail.com", "vukx vkta aaey lghx")
            s.sendmail(msg['From'], msg['To'], msg.as_string())
            s.quit()
            print("Email sent successfully.")
        except Exception as e:
            print(f"Failed to send email: {e}")


def main():
    print("Starting main function")

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detection_count = 0  # Count the number of object detections

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame to grayscale for detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        # Draw bounding boxes around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box for faces
            detection_count += 1  # Increment detection count

        # Send email if enough detections occur
        if detection_count >= detection_count_threshold:
            print("Face detected")
            thread = Thread(target=send_email, args=(frame,))
            thread.start()
            detection_count = 0  # Reset the count after sending the email

        cv2.imshow("Frame", frame)

        # Use a shorter wait time to improve responsiveness
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)  # Capture frames at a higher frequency

    cap.release()
    cv2.destroyAllWindows()
    print("Main function ended")


if __name__ == "__main__":
    main()
