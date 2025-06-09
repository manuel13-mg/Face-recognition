import cv2
import pickle
import numpy as np
import os

try:  # Wrap the entire code in a try-except block for global error handling
    video=cv2.VideoCapture(0)
    facedetect=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Load from built-in path

    if not video.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    faces_data=[]

    i=0

    name=input("Enter Your Name: ")

    while True:
        ret,frame=video.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=facedetect.detectMultiScale(gray, 1.3 ,5)
        for (x,y,w,h) in faces:
            crop_img=frame[y:y+h, x:x+w, :]
            resized_img=cv2.resize(crop_img, (50,50))
            if len(faces_data)<=100 and i%10==0:
                faces_data.append(resized_img)
            i=i+1
            cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        cv2.imshow("Frame",frame)
        k=cv2.waitKey(1)
        if k==ord('q') or len(faces_data)==100:
            break
    video.release()
    cv2.destroyAllWindows()

    faces_data=np.asarray(faces_data)
    if faces_data.size > 0:  # Check if there are any faces
        faces_data=faces_data.reshape(len(faces_data), -1) #Adjust this code so that the code accounts for the faces that has been acquired.
    else:
        print("No faces were captured.") #Check if this is actually a 100 or 0 problem.
        exit()

    data_dir = 'data/'  # Define the data directory
    if not os.path.exists(data_dir): #The code looks to create the directory
        os.makedirs(data_dir)

    names_file = os.path.join(data_dir, 'names.pkl')
    faces_data_file = os.path.join(data_dir, 'faces_data.pkl')

    try:
        if not os.path.exists(names_file):
            names=[name]*len(faces_data)  # Adjust list length, so that no error may occurs.
            with open(names_file, 'wb') as f:
                pickle.dump(names, f)
        else:
            with open(names_file, 'rb') as f:
                names=pickle.load(f)
            names=names+ [name]*len(faces_data) # Adjust list length, so that no error may occurs.
            with open(names_file, 'wb') as f:
                pickle.dump(names, f)
    except Exception as e:
        print(f"Error processing names.pkl: {e}")

    try:
        if not os.path.exists(faces_data_file):
            with open(faces_data_file, 'wb') as f:
                pickle.dump(faces_data, f)
        else:
            with open(faces_data_file, 'rb') as f:
                faces=pickle.load(f)

            if faces.shape[1] == faces_data.shape[1]: # added, to ensure the number of faces that has been acquired
                faces=np.concatenate((faces, faces_data), axis=0) # Added a new array and concantenate with faces to ensure correct image amount.
            else:
                print(f"Mismatched feature size: Existing faces have {faces.shape[1]} features, new faces have {faces_data.shape[1]} features. Exiting.")
                exit()
            with open(faces_data_file, 'wb') as f:
                pickle.dump(faces, f)

    except Exception as e:
        print(f"Error processing faces_data.pkl: {e}")

except Exception as e:
    print(f"A top-level error occurred: {e}") #If any code fails, show the error message