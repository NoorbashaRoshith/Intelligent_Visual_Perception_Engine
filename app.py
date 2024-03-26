import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import numpy as np

# Load the pre-trained YOLO model
model = YOLO('../models/best.pt')

# Read the class list from a file
with open('models/vehicles.txt', 'r') as my_file:
    class_list = my_file.read().split("\n")


# Function to check if user is registered
def is_registered(email):
    # Check if email exists in database (you can replace this with your own database logic)
    # For demo purposes, assuming a list of registered emails
    registered_emails = ['user1@example.com', 'admin@example.com']  # Example registered emails
    return email in registered_emails


# Streamlit interface
st.title('Intelligent Visual Perception Engine')

# Pages
if 'page' not in st.session_state:
    st.session_state.page = "Login"

if st.session_state.page == "Login":
    # Login Page
    st.subheader("Login Page")
    login_email = st.text_input("Email", key="login_email")
    login_password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        # Check credentials (you can replace this with your own authentication logic)
        if is_registered(login_email):
            # Set session state to logged in
            st.session_state.is_logged_in = True
            st.session_state.page = "Object Detection"
        else:
            st.error("Invalid email or password. Please register if you don't have an account.")
            st.session_state.page = "Registration"

elif st.session_state.page == "Registration":
    # Registration Page
    st.subheader("Registration Page")
    reg_email = st.text_input("Email", key="reg_email")
    reg_username = st.text_input("Username", key="reg_username")
    reg_password = st.text_input("Password", type="password", key="reg_password")

    if st.button("Register"):
        # Add registration logic (you can replace this with your own registration logic)
        # For demo purposes, just print registration details
        st.write("Registered Successfully!")
        st.write("Username:", reg_username)
        st.write("Email:", reg_email)
        st.write("Password:", reg_password)
        st.session_state.page = "Login"

else:
    # Object Detection Page
    st.write('Upload an image and the model will detect Vehicles in your.')

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Read the image
        frame = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Resize the frame to a fixed size (if necessary)
        frame = cv2.resize(frame, (1020, 500))

        # Perform object detection on the frame
        results = model.predict(frame)
        detections = results[0].boxes.data
        px = pd.DataFrame(detections).astype("float")

        # Display the detected objects
        st.image(frame, channels="BGR", caption="Uploaded Image with Object Detection", use_column_width=True)

        # Display the detected objects and their class labels
        truck_count = 0
        for index, row in px.iterrows():
            x1, y1, x2, y2, _, d = map(int, row)
            c = class_list[d]

            # Draw bounding boxes with custom style
            color = (0, 0, 255)  # Red color for bounding boxes
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw class labels with custom style
            label_position = (x1, y1 - 10)  # Position just above the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_color = (0, 0, 0)  # Black color for text
            cv2.putText(frame, c, label_position, font, font_scale, font_color, thickness)

            # List of keywords representing heavy load vehicles
            heavy_load_keywords = ["truck", "lorry", "container"]

            # Check if any keyword is present in the lowercase class label
            if any(keyword in c.lower() for keyword in heavy_load_keywords):
                truck_count += 1

        # Display the frame with objects detected
        st.image(frame, channels="BGR", caption="Detected Objects", use_column_width=True)

        # Use HTML and CSS to style the text
        st.markdown(f"""
        <style>
            .truck_count {{
                font-size: 24px;
                color: red;
            }}
        </style>
        <div class="truck_count">Number of Heavy Load Vehicles detected: {truck_count}</div>
        """, unsafe_allow_html=True)
