import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import spacy
import requests
import base64
from transformers import pipeline  # For advanced NLP tasks

def get_image_as_base64(image_file):
    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Initialize spaCy (local model) for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Initialize advanced NLP model (Hugging Face Transformers)
qa_pipeline = pipeline("question-answering")

# Set the page configuration
st.set_page_config(page_title="PAGE READER BOT", initial_sidebar_state="auto")

# Custom CSS for full background image and improved style
image_url = "https://images.unsplash.com/photo-1519389950473-47ba0277781c"  # Replace with any of the above URLs

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url('{image_url}');
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        font-family: 'Arial', sans-serif;
        color: rgb(232, 25, 25);
        margin: 0;
        padding: 0;
        height: 100vh;
    }}
    .main {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 3rem;
        border-radius: 15px;
        color: rgb(142, 156, 20);
        box-shadow: 0px 0px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 15vh;
    }}
    h1 {{
        color: #FFA500;
        text-align: center;
        margin-top: -40px;
        font-size: 4rem;
        font-weight: 700;
    }}
    .footer {{
        position: fixed;
        bottom: 0;
        width: 60%;
        background-color: hsla(0, 0.00%, 0.00%, 0.50);
        color: #ffffff;
        text-align: center;
        padding: 15px 0;
        font-size: 1rem;
    }}
    </style>
""", unsafe_allow_html=True)

# App title
main_heading = "<h1>📸 PAGE READER BOT</h1>"
st.markdown(main_heading, unsafe_allow_html=True)

# File uploader for images and PDFs
uploaded_file = st.file_uploader("Choose an image or PDF", type=["jpg", "pdf", "jpeg", "png"])

# Page range inputs for PDFs
if uploaded_file and uploaded_file.type == "application/pdf":
    min_page = st.number_input("Start Page", min_value=1, value=1, step=1)
    max_page = st.number_input("End Page", min_value=1, value=1, step=1)

# Minimum confidence input
min_confidence = st.number_input("Minimum confidence:", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

# List to store detected text
detected_text = []

# Enhanced function for preprocessing image for handwritten text recognition
def preprocess_handwritten_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresholded = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

# Function to detect text using EasyOCR
def detect_text_with_easyocr(image):
    result = reader.readtext(image)
    return result

# Function to process PDF into images
def convert_pdf_to_images(pdf_path, first_page, last_page):
    images = convert_from_path(pdf_path, first_page=first_page, last_page=last_page)
    return images

# Function to fetch web search results
def fetch_web_search_results(query):
    response = requests.get(f"https://www.googleapis.com/customsearch/v1?q={query}&key=AIzaSyBGLy0pA6OrvWhP-ZnM_5fUHQRrZ-WROP0")
    if response.status_code == 200:
        return response.json()
    return None

# Fallback method for generating response using advanced NLP model
def fallback_response(user_input, document):
    response = "Hmm, I'm not sure I understood that. Could you clarify?"

    # Generate answer using advanced NLP model
    if document.strip():
        result = qa_pipeline(question=user_input, context=document)
        response = result['answer']

    # Fetch web search results
    web_results = fetch_web_search_results(user_input)
    if web_results:
        response += "\n\nAdditionally, I found some information online:\n"
        for result in web_results.get('items', []):
            response += f"- {result['title']}: {result['snippet']}\n"

    return response

# Button to detect text
if st.button("Detect text"):
    if uploaded_file is not None:
        image_path = "uploaded_file"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption='Original Image', use_container_width=True)

            st.write("Preprocessing image for handwriting recognition...")
            image = np.array(original_image)
            preprocessed_image = preprocess_handwritten_image(image)

            st.write("Detecting text using EasyOCR...")
            try:
                result = detect_text_with_easyocr(preprocessed_image)

                detected_text.clear()

                for detection in result:
                    detected_text.append(detection[1])
                    st.write(f"Detected text: **{detection[1]}**")

                for detection in result:
                    points = detection[0]
                    start_x, start_y = points[0]
                    end_x, end_y = points[2]
                    cv2.rectangle(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 255, 0), 2)

                processed_image = Image.fromarray(image)
                st.image(processed_image, caption='Processed Image', use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")

        elif uploaded_file.type == "application/pdf":
            images = convert_pdf_to_images(image_path, min_page, max_page)

            for idx, page in enumerate(images):
                st.image(page, caption=f"Page {min_page + idx}", use_container_width=True)
                page_np = np.array(page)
                result = detect_text_with_easyocr(page_np)

                detected_text.clear()

                for detection in result:
                    detected_text.append(detection[1])
                    st.write(f"Detected text on page {min_page + idx}: **{detection[1]}**")

                for detection in result:
                    points = detection[0]
                    start_x, start_y = points[0]
                    end_x, end_y = points[2]
                    cv2.rectangle(page_np, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 255, 0), 2)

                processed_image = Image.fromarray(page_np)
                st.image(processed_image, caption=f"Processed Image - Page {min_page + idx}", use_container_width=True)

# Chatbot Section for User Input
st.subheader("🤖 Chatbot Query - Ask Questions About the Text")

# Initialize conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Get user input for chatbot interaction
user_input = st.text_input("Ask a question:", key="chatbot_input", placeholder="Type your question here...")

if user_input:
    document = " ".join(detected_text)
    if document.strip():
        response = fallback_response(user_input, document)
        st.session_state.conversation_history.append(("User: " + user_input, "Bot: " + response))

# Display chatbot conversation
if st.session_state.conversation_history:
    for question, answer in st.session_state.conversation_history:
        st.write(question)
        st.write(answer)

# Footer for the app
st.markdown('<div class="footer">Made with ❤️ by SMVR</div>', unsafe_allow_html=True)
