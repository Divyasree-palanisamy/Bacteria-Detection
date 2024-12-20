import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import os
from pathlib import Path



# Load your pre-trained model
model = load_model(r'C:\Users\divya\OneDrive\Desktop\mini project\bacteria_class.keras')


# Define classes for the model's prediction
data_cat = ['Amoeba', 'Euglena', 'Hydra', 'Paramecium', 
            'Rod Bacteria', 'Spherical Bacteria', 
            'Spiral Bacteria', 'Yeast']

# Define image dimensions for resizing
img_height = 180
img_width = 180

# Dictionary containing information for each bacteria type
bacteria_info = {
    'Amoeba': """ Amoeba are single-celled organisms known for their irregular shape.",
            "They move using pseudopodia, which are temporary projections of their cytoplasm.",
            "Amoebae are commonly found in freshwater environments.",
            "They are heterotrophic, feeding on bacteria and other small particles.",
            "Amoeba reproduce asexually through binary fission.",
            "Some species can cause diseases in humans, such as amoebic dysentery.",
            "Amoeba exhibit behaviors such as phagocytosis to engulf food.",
            "They play an essential role in the food web as both predators and prey.",
            "Amoeba can survive harsh conditions by forming cysts.",
            "Their study helps understand cellular processes and evolution.""",
    
    'Euglena': """Euglena are flagellated unicellular organisms that can photosynthesize.",
            "They have a unique mix of plant and animal characteristics.",
            "Euglena possess chloroplasts, allowing them to perform photosynthesis.",
            "They can also absorb nutrients from their environment.",
            "Euglena are typically found in freshwater bodies.",
            "They move using a whip-like flagellum, which propels them forward.",
            "Some species can reproduce asexually through longitudinal fission.",
            "Euglena are indicators of water quality and ecological health.",
            "They have a red eyespot that helps detect light.",
            "Research on Euglena contributes to understanding photosynthesis and biofuels.""",
    
    'Hydra': """Hydra are small, freshwater organisms belonging to the phylum Cnidaria.",
            "They are known for their tubular body and tentacles that capture prey.",
            "Hydra can regenerate lost body parts, showcasing remarkable regeneration abilities.",
            "They reproduce both asexually through budding and sexually.",
            "Hydra typically live in slow-moving waters with abundant vegetation.",
            "Their diet mainly consists of small aquatic animals and zooplankton.",
            "Hydra have specialized cells called cnidocytes that contain stinging structures.",
            "They serve as a model organism in developmental biology studies.",
            "Hydra are sensitive to environmental changes, making them indicators of pollution.",
            "They have been used in research to study cellular differentiation and aging.""",
    
    'Paramecium': """Paramecium are slipper-shaped, ciliated protozoa found in freshwater environments.",
            "They move by beating tiny hair-like structures called cilia.",
            "Paramecium feed on bacteria and small particles through oral grooves.",
            "They reproduce asexually through binary fission and can also undergo conjugation.",
            "Paramecium play a role in nutrient cycling within aquatic ecosystems.",
            "They are sensitive to changes in water quality and pollution levels.",
            "Paramecium have a complex structure with a pellicle for support.",
            "They are widely studied in genetics and cellular biology.",
            "Some species of Paramecium can be used in ecological monitoring.",
            "They are a common subject in microbiology education.""",
    
    'Rod Bacteria': """Rod-shaped bacteria (bacilli) are elongated bacteria that can be found in a variety of environments, including on human skin and hands. Some rod-shaped bacteria are harmless, while others can cause infections.

Common Examples:

Escherichia coli (E. coli) — can cause gastrointestinal and urinary infections.
Bacillus species — may cause foodborne illnesses or infections in immunocompromised individuals.
Clostridium species — can cause tetanus or botulism under certain conditions.
Risk Factors:

Rod-shaped bacteria can be present on the hands, especially after touching contaminated surfaces, raw food, or soil.""",
    
    'Spherical Bacteria': """Spiral-shaped bacteria, or spirochetes, have a helical structure and are known for their ability to move in a corkscrew motion. Some species of spirochetes are pathogenic and can be transmitted through direct contact.

Common Examples:

Treponema pallidum — causes syphilis, a sexually transmitted infection.
Borrelia burgdorferi — responsible for Lyme disease, transmitted by ticks but can be present on the skin after a bite.
Leptospira species — cause leptospirosis, a bacterial infection that can be spread through direct contact with water contaminated by the urine of infected animals.
Risk Factors:

Spirochetes are usually transmitted through broken skin or mucous membranes, so hand contact with contaminated water or surfaces, particularly with abrasions, can increase the risk.
The bacteria can live in moist environments like ponds, rivers, or contaminated standing water, and exposure through unclean hands can lead to infection.""",
    
    'Spiral Bacteria': """Spherical bacteria, also known as cocci, can be found in various shapes: clusters, chains, or pairs. These bacteria can live on the skin and are responsible for a variety of infections, including skin and respiratory illnesses.

Common Examples:

Staphylococcus aureus — can cause skin infections, pneumonia, and even more serious infections like sepsis.
Streptococcus pyogenes — can cause strep throat, skin infections, and other serious conditions like toxic shock syndrome.
Neisseria gonorrhoeae — causes gonorrhea, a sexually transmitted infection.
Risk Factors:

Spherical bacteria, particularly Staphylococcus and Streptococcus, are common on human skin and mucous membranes. However, these bacteria can become problematic if they enter the body through cuts, scrapes, or through poor hand hygiene.
Cocci can be transmitted from person to person, especially in high-touch environments like hospitals, schools, and public places.""",
    
    'Yeast': """Yeast are unicellular fungi commonly used in baking and fermentation.",
            "They reproduce asexually by budding, forming new yeast cells.",
            "Yeast play a crucial role in alcohol production and bread-making processes.",
            "They are found in various environments, including soil and plant surfaces.",
            "Some yeast species are beneficial, while others can cause infections.",
            "Yeast cells are eukaryotic, with complex cellular structures.",
            "They ferment sugars, producing carbon dioxide and alcohol as byproducts.",
            "Research on yeast contributes to biotechnology and genetics.",
            "Yeast is a model organism in molecular biology studies.",
            "Their ability to thrive in various conditions makes them essential for food production."""
}
# Function to preprocess the image for counting bacteria
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding
    _, binary_image = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    
    return binary_image

# Function to count bacteria in the image
def count_bacteria(image): 
    binary_image = preprocess_image(image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bacteria_count = len(contours)
    return bacteria_count
# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "About the Project", "Feedback"])


# --- Home Section ---
if page == "Home":
    
    st.title("🦠 Welcome to the Bacteria Detection Website")
    st.markdown(""" 
        Welcome to the Bacteria Detection Project! We design this tool to aid researchers, students, and educators by providing an efficient way to identify different bacteria types through image analysis. By uploading an image of a bacterial sample, our model can classify and provide  the insights about  detected bacteria type, and also the  essential recommendations based on the bacteria count.

Project Objectives
In a world where microorganisms play a vital role, understanding bacteria is essential. Our project serves as a bridge between microbiology and artificial intelligence, utilizing deep learning to simplify bacterial identification. Our AI model can recognize various bacterial types, including Rod Bacteria , Spiral Bacteria and Spherical Bacteria.

Benefits of this Tool
With applications in research, healthcare, and environmental studies, this project aims to:

Support Academic Research: By quickly identifying bacterial types, students and researchers can focus on deeper analysis.
Enhance Public Health Monitoring: By identifying harmful bacteria types, this tool can aid in assessing health risks.
Improve Educational Resources: Educators can use this tool as a hands-on resource in teaching microbiology.
Explore the different sections of our page to learn more about bacteria, analyze samples, and obtain safety recommendations. Whether you're here to classify, learn, or conduct research, this project is built to support your journey in understanding the microbial world.


    """)

# --- Prediction Section ---
# --- Prediction Section ---
# --- Prediction Section ---
# --- Prediction Section ---
elif page == "Prediction":
    
    st.title("Bacteria Detection Model")
    st.markdown("Upload an image of bacteria for detection and counting.")

    # File uploader for the image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    predict_button = st.button("Predict Bacteria and Count Bacteria")

    if predict_button:
        if uploaded_file is not None:
            # Convert the uploaded file to a numpy array suitable for OpenCV
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            # Display the uploaded image
            st.image(opencv_image, channels="BGR", caption="Uploaded Image", width=300)

            # Resize the image and preprocess it for the model
            processed_image = cv2.resize(opencv_image, (img_height, img_width))
            processed_image = np.expand_dims(processed_image, axis=0)  # Create batch dimension

            # Make the prediction
            prediction = model.predict(processed_image)
            score = tf.nn.softmax(prediction)

            # Get the predicted label and confidence
            predicted_label = data_cat[np.argmax(score)]
            confidence = np.max(score) * 100

            # Display prediction and confidence
            st.write(f"This is a **{predicted_label}** with an accuracy of **{confidence:.2f}**%.")

            # Display additional information about the detected bacteria type
            st.markdown(f"### Information about {predicted_label}:")
            st.write(bacteria_info[predicted_label])

            # Count bacteria and display results
            bacteria_count = count_bacteria(opencv_image)

            # Define thresholds for bacteria count
            if bacteria_count > 100:
                count_color = "red"
                count_text = f"High bacteria detected: {bacteria_count}"
                recommendations = """
                Handwashing Protocol:

Wash your hands for at least 20 seconds with soap and water after touching potentially contaminated surfaces, handling samples, or before eating.
Use alcohol-based hand sanitizers (containing at least 60% alcohol) if soap and water are not available, especially when entering or leaving high-risk areas.
Avoid Touching Face: Keep hands away from your face, especially your eyes, nose, and mouth, to avoid transferring bacteria from hands to mucous membranes.

                """
            elif 50 < bacteria_count <= 100:
                count_color = "yellow"
                count_text = f"Average bacteria detected: {bacteria_count}"
                recommendations = """
                Hand Hygiene:

Wash your hands thoroughly with soap and water for at least 20 seconds after touching surfaces or objects that could be contaminated.
Use hand sanitizer with at least 60% alcohol if soap and water are not available, especially after touching commonly used items.
Keep an eye on the bacteria levels or any symptoms that may arise. Ensure that contaminated areas are addressed with increased hygiene practices.
                """
            else:
                count_color = "green"
                count_text = f"Low bacteria detected: {bacteria_count}"
                recommendations = """
                Basic Hand Hygiene:

Wash hands thoroughly with soap and water for at least 20 seconds before eating, after using the restroom, or after touching surfaces that might be exposed to bacteria.
When soap and water aren’t available, use hand sanitizer with at least 60% alcohol to ensure bacteria are minimized.
Encourage healthy habits like regular exercise, maintaining a balanced diet, and getting enough sleep to help your body resist infections caused by bacteria.
                """

            # Display count text and recommendations
            # Display count text and recommendations
            st.markdown(f"""
    <div style='border: 2px solid {count_color}; border-radius: 10px; padding: 10px;'>
        <h3 style='color: {count_color}; font-size: 24px;'>{count_text}</h3>
    </div>
""", unsafe_allow_html=True)

# Prepare recommendations with similar styling
            recommendations_styled = f"""
    <div style='border: 2px solid {count_color}; border-radius: 10px; padding: 10px; margin-top: 10px;'>
        <h4 style='color: {count_color};'>Recommendations:</h4>
        <p>{recommendations.replace('\n', '<br>')}</p>
    </div>
"""

# Display recommendations with the styled box
            st.markdown(recommendations_styled, unsafe_allow_html=True)


            # Prepare result for download as a text file
            result_text = f"Prediction: {predicted_label}\nConfidence: {confidence:.2f}%\nBacteria Count: {bacteria_count}"
            st.download_button(
                label="Download Prediction and Count Result",
                data=result_text,
                file_name="prediction_result.txt",
                mime="text/plain"
            )
        else:
            st.error("Please upload an image file to proceed.")



# --- About the Project Section ---
elif page == "About the Project":
    st.title("About the Project")
    st.markdown(""" 
        ### Project Overview 
        This project was developed to utilize deep learning techniques for identifying different types of bacteria. 

        ### Techniques Used 
        - 🧠 **Deep Learning Model**: A convolutional neural network (CNN) was used for image classification. 
        - 🔄 **Transfer Learning**: Leveraged pre-trained models for more accurate predictions. 
        - 🖼️ **Image Preprocessing**: Each image is resized and normalized before it is fed into the model. 

        ### Technology Stack 
        - ⚙️ **TensorFlow and Keras** for building and training the model 
        - 📷 **OpenCV** for image processing 
        - 🌐 **Streamlit** for creating the web interface 
    """)

# --- Comments Section ---
elif page == "Feedback":
    st.title("✉️ Feedback")
    
    st.write("We value your feedback! Please leave your comments below:")
    st.subheader("💡 Share your thoughts!")


    # Email input
    email = st.text_input("📧 Your Email")
    subject = st.text_input("📝 Subject")
    message = st.text_area("💬 Message")

    # Button to submit comment
    if st.button("✅ Submit"):
        # Check if all fields are filled
        if email and subject and message:
            try:
                # Log for debugging
                feedback_entry = (
                    f"Email: {email}\n"

                    f"Subject: {subject}\n"
                    f"Message: {message}\n"
                    f"{'-'*40}\n"
                )

                
                # Define the file path
                file_path = r"C:\Users\divya\OneDrive\Desktop\mini project\streamlit app\user_messages.txt"
                
                
                # Append feedback to the file
                with open(file_path, "a") as f:
                    f.write(feedback_entry)

                st.success("✅ Thank you for your feedback! We will get back to you soon.")
            except Exception as e:
                st.write(f"❌ Error: {e}")
                st.error(f"❌ Error saving feedback: {e}")