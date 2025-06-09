import cv2
import os
import numpy as np  # Import numpy here, required for image processing.  Crucial!

def face_recognition_with_name():
    """
    Captures the user's name, captures a photo, trains a face recognizer,
    and then performs live face recognition.
    """

    # 1. Get User's Name
    name = input("Enter your name: ")

    # 2. Create a Directory for the User's Images (if it doesn't exist)
    data_folder = "face_data"  # You can change this folder name
    user_folder = os.path.join(data_folder, name)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
        print(f"Created directory: {user_folder}")

    # 3. Capture Images for Training
    capture_images(user_folder)

    # 4. Train the Face Recognizer
    train_face_recognizer(data_folder)

    # 5. Perform Live Face Recognition
    live_face_recognition(data_folder)


def capture_images(user_folder, num_images=50):
    """
    Captures images from the webcam and saves them in the specified user folder.
    """
    video_capture = cv2.VideoCapture(0) # Use 0 for default webcam

    if not video_capture.isOpened():
        print("Error: Could not access the webcam.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image_count = 0

    print(f"Capturing {num_images} images for {user_folder}...")

    while image_count < num_images:
        ret, frame = video_capture.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) # Adjusted parameters

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle around face

            # Save the face image
            image_path = os.path.join(user_folder, f"image_{image_count}.jpg")
            face_roi = gray[y:y+h, x:x+w]  # Extract Region of Interest (the face)
            cv2.imwrite(image_path, face_roi)
            print(f"Saved: {image_path}")
            image_count += 1

        cv2.imshow('Capturing Images', frame)

        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("Image capture complete.")


def train_face_recognizer(data_folder):
    """
    Trains a face recognizer using the images in the specified data folder.
    """
    faces = []
    labels = []
    label_id = 0
    id_map = {}  # Maps folder names (user names) to integer IDs

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for root, dirs, files in os.walk(data_folder):
        for dir_name in dirs:
            id_map[dir_name] = label_id
            label_id += 1

    current_id = 0
    for root, dirs, files in os.walk(data_folder):
        for dir_name in dirs:
            path = os.path.join(root, dir_name)
            for file in files:
                if file.endswith("jpg") or file.endswith("jpeg") or file.endswith("png"):
                    image_path = os.path.join(path, file)
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Read in grayscale

                    if img is None:
                        print(f"Error: Could not read image {image_path}")
                        continue  # Skip to the next image

                    # Detect faces *in each individual training image*
                    face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                    if len(face) > 0:
                        x, y, w, h = face[0]
                        roi = img[y:y+h, x:x+w]
                        faces.append(roi)
                        labels.append(id_map[dir_name])
                    else:
                        print(f"Warning: No face detected in {image_path}. Skipping.")

    if not faces:
        print("Error: No faces found in the training data.  Make sure you have captured images and the face detector is working.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create() # Local Binary Patterns Histograms (LBPH)
    try:
        recognizer.train(faces, np.array(labels))  # Convert labels to a NumPy array
    except Exception as e:
        print(f"Error during training: {e}")
        return

    recognizer.save("trainer.yml") # Save the trained model
    print("Face recognizer trained and saved as trainer.yml")

    # Save the id_map
    with open("id_map.txt", "w") as f:
        for name, id in id_map.items():
            f.write(f"{name}:{id}\n")
    print("ID map saved as id_map.txt")

def live_face_recognition(data_folder):
    """
    Performs live face recognition using the trained model.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read("trainer.yml") # Load the trained model
    except cv2.error as e:
        print(f"Error loading trainer.yml: {e}")
        print("Make sure you have trained the model first by running the previous steps.")
        return
    except FileNotFoundError:
        print("trainer.yml not found.  Make sure you have trained the model first.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load id_map from file
    id_map = {}
    try:
        with open("id_map.txt", "r") as f:
            for line in f:
                name, id = line.strip().split(":")
                id_map[int(id)] = name  # Reverse the mapping for prediction
    except FileNotFoundError:
        print("id_map.txt not found.  Make sure you have trained the model first.")
        return

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not access the webcam.")
        return

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]

            try:
                id_, confidence = recognizer.predict(roi_gray)
                confidence = 100 - confidence  # Flip for easier interpretation
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue  # Skip to the next face

            if confidence > 50:  # Adjust confidence threshold as needed
                if id_ in id_map:  # Make sure ID exists in our id_map
                    name = id_map[id_]
                else:
                    name = "Unknown"  # ID not found

                cv2.putText(frame, f"{name} ({confidence:.2f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_recognition_with_name()