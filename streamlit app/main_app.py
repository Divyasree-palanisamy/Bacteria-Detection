import numpy as np
import streamlit as st
import cv2
from PIL import Image
from keras.models import load_model
from tensorflow.keras.preprocessing import image

st.markdown("""
    <style>
        body {
            background-color: lightblue;
        }
        h1 {
            color: red;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    .main {
        background-image: url("https://example.com/background.jpg");
        background-size: cover;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# Load the pre-trained model for bacteria prediction
model = load_model(r'C:\Users\divya\Desktop\mini project\streamlit app\one.h5')
CLASS_NAMES = ['Yeast', 'Euglena', 'Amoeba', 'Paramecium', 'Hydra', 'Rod Bacteria', 'Spherical Bacteria', 'Spiral Bacteria']

# Set page configuration for better layout
st.set_page_config(page_title="Bacteria Detection System", page_icon="🦠", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "About", "FAQs", "Comments"))

def preprocess_image_for_prediction(image):
    """Preprocess image to the expected (128, 128, 3) shape for prediction."""
    image = image.resize((128, 128))
    image_array = np.array(image)
    if image_array.shape[-1] != 3:
        image_array = np.stack([image_array] * 3, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0
    return image_array

def predict_organism(image):
    """Predict the class of the organism in the uploaded image."""
    processed_image = preprocess_image_for_prediction(image)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    return predicted_class_name

def preprocess_image_for_counting(image):
    """Preprocess image for bacteria counting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return binary

def count_bacteria(image):
    """Count the bacteria in the image and return the count and processed image."""
    binary_image = preprocess_image_for_counting(image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bacteria_count = len(contours)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    return bacteria_count, image

if page == "Home":
    st.markdown("<h1 class='main-title'>🦠 Bacteria Detection on Hands</h1>", unsafe_allow_html=True)

    
    st.subheader("Project Overview")
    st.write("""
        This project is designed to detect bacteria present on hands using image processing techniques. 
        The application analyzes uploaded images to provide feedback on hand hygiene, helping to promote better health practices.
    """)

      # Prediction Section
    st.markdown("<h2 class='sub-title'>Organism Prediction</h2>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload an image of bacteria or organism (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])

    if st.button('Predict Organism'):
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            try:
                result = predict_organism(image)
                st.success(f"Prediction: {result}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please upload an image.")

    # Counting Section
    st.subheader("Upload an Image for Bacteria Counting")
    uploaded_file = st.file_uploader("Upload an image of hands (png, jpg, jpeg)", type=['png', 'jpg', 'jpeg'], key="counting")

    if uploaded_file is not None:
        # Read and display the image
        image = Image.open(uploaded_file)

        # Set the desired width for the displayed image
        display_width = 500
        st.image(image, caption='Uploaded Image', use_column_width=False, width=display_width)

        # Convert the image for OpenCV processing
        image = np.array(image)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Count bacteria and prepare for output
        bacteria_count, output_image = count_bacteria(image.copy())

        # Set thresholds and color indicator
        if bacteria_count < 30:
            color, message = "green", "Low bacteria count - Clean hands!"
        elif 30 <= bacteria_count < 100:
            color, message = "yellow", "Moderate bacteria count - Wash hands soon!"
        else:
            color, message = "red", "High bacteria count - Immediate cleaning required!"

        # Display the results
        st.subheader(f"Bacteria Count: {bacteria_count}")
        st.markdown(f"<h3 style='color: {color};'>{message}</h3>", unsafe_allow_html=True)

        # Show output image with contours at reduced size
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        st.image(output_image_rgb, caption='Bacteria Detection Output', use_column_width=False, width=display_width)

elif page == "About":
    st.title("About This Project")
    st.write("""
        This project is designed to detect the bacteria present on hands using image processing techniques.
        The application processes uploaded images to analyze the bacterial count and provide feedback on hand hygiene.
    """)

    st.subheader("Technologies Used")
    st.write("""
        **Software:**
        - Python
        - Streamlit
        - OpenCV
        - NumPy
        - PIL
        **Hardware:**
        - Any standard laptop/PC with a webcam (optional for live capture)
    """)

elif page == "FAQs":
    st.title("Frequently Asked Questions")
    st.write("""
        **Q: How does the bacteria counting process work?**
        A: The application analyzes the uploaded image to detect and count the contours representing bacteria on hands. 

        **Q: What does the bacteria count indicate?**
        A: The bacteria count provides an estimate of the microbial load on the hands, which can be an indicator of hygiene. 

        **Q: What are the bacteria count thresholds?**
        A: 
        - A count below 30 indicates clean hands.
        - A count between 30 and 100 suggests moderate bacterial presence, indicating a need for washing hands soon.
        - A count above 100 suggests high bacterial presence, recommending immediate cleaning.

        **Q: Can the application differentiate between types of bacteria?**
        A: No, the application counts contours in the image, but it does not identify specific types of bacteria. It provides a general count based on visible markers.

        **Q: How accurate is the bacteria count?**
        A: The accuracy may vary based on image quality and lighting conditions. Proper lighting and a clear image of hands can improve results.

        **Q: Can this application be used for other surfaces or just hands?**
        A: Currently, the application is designed specifically for detecting bacteria on hands. Future updates may expand its functionality.

        **Q: What should I do if the bacteria count seems unusually high?**
        A: If the count is high, it is recommended to wash your hands thoroughly with soap and water to reduce bacterial load.
    """)

elif page == "Comments":
    st.title("Feedback and Comments")
    st.write("We value your feedback! Please leave your comments below:")

    # Email input
    email = st.text_input("Your Email")

    # Subject input
    subject = st.text_input("Subject")

    # Message input
    message = st.text_area("Message")

    # Button to submit feedback
    if st.button("Submit"):
        # Debugging print to verify values received
        print(f"Button clicked. Email: {email}, Subject: {subject}, Message: {message}")  

        if email and subject and message:
            try:
                # Ensure the file exists before writing (create if doesn't exist)
                file_path = "user_messages.txt"
                if not os.path.exists(file_path):
                    print(f"File {file_path} does not exist. It will be created.")
                
                # Write to the file
                with open(file_path, "a") as f:
                    f.write(f"Email: {email}\nSubject: {subject}\nMessage: {message}\n\n")
                
                st.success("Thank you for your feedback! We will get back to you soon.")
                print("Message written to user_messages.txt.")
            except Exception as e:
                # Handle and log any file writing issues
                print(f"Error writing to file: {e}")
                st.error(f"An error occurred: {e}")
        else:
            # If fields are missing, log the issue
            print("Please fill in all fields.")  
            st.error("Please fill in all fields.")

st.markdown("<div class='footer'>© 2024 Bacteria Detection Project. All rights reserved.</div>", unsafe_allow_html=True)
