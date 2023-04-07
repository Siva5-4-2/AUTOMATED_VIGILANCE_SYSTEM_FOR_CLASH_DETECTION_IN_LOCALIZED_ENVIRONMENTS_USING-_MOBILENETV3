# -*- coding: utf-8 -*-
"""
Created on Fri Mar 3 18:52:44 2023

@author: sivag
"""


import tkinter as tk
import cv2
import numpy as np
import time
import os
from twilio.rest import Client
from PIL import Image
from PIL import ImageTk
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('modelnew.h5')

# Start the video capture
cap = cv2.VideoCapture(0)
count=1
# Set the duration of the violence detection period in seconds
#detection_duration = 2

# Set the start time of the violence detection period
start_time = time.time()

# Create the main window
root = tk.Tk()
root.title("Violence Detection")

# Create a label for displaying the video feed
label = tk.Label(root)
label.pack()

def update_video_feed():
    global cap, label, model, count, start_time
    
    # Capture a frame from the video feed
    ret, frame = cap.read()

    # Resize the frame to 224x224 and pre-process it for the model
    resized_frame = cv2.resize(frame, (128,128))
    np_frame = np.expand_dims(resized_frame, axis=0)
    np_frame = np_frame.astype('float32') / 255.0

    # Use the model to predict if the frame contains violence
    prediction = model.predict(np_frame)

    # Show the prediction on the frame
    if prediction > 0.5:
        label_text = 'Violence'  
        count=count+1
        if count==10:
        #if time.time() - start_time >= detection_duration:
            account_sid = "ACeed5e5f6d6f651629af657537d183315"
            auth_token = "633cc46b4982bc86145d6ee89040bf96"
            client = Client(account_sid, auth_token)
           
            call = client.calls.create(
              twiml='<Response><Say>Violence detected in your area, immediately take necessary action.</Say></Response>',
              to="+918610607601",
              from_="+15739282755"
            )
           
            print(call.sid)  
            filename = f"{time.strftime('%Y%m%d-%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
             
           
    else:
        label_text = 'No Violence'
        count=1
    start_time = time.time()
    cv2.putText(frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    
    # Schedule the next update of the video feed
    label.after(1, update_video_feed)

# Start updating the video feed
update_video_feed()

# Run the main loop of the GUI
root.mainloop()

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()