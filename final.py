import streamlit as st
from PIL import Image
import os
import pyttsx3
import pytesseract
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
import threading  # Import threading for parallel execution

os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR'
# Configure Tesseract OCR path (Update if Tesseract is not in PATH)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

f = open("keys/gemini.txt")
google_api_key = f.read()

os.environ["GOOGLE_API_KEY"] = google_api_key

llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key=google_api_key)

engine = pyttsx3.init()

st.title("AI vision assist")

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def text_to_speech(text):
    def speak():
        engine.say(text)
        engine.runAndWait()

    # Use threading to avoid blocking the main event loop
    thread = threading.Thread(target=speak)
    thread.start()

def generate_scene_description(input_prompt, image_data):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

uploaded_file = st.file_uploader("Drag and Drop or browse an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
text = None  # Initialize text variable to avoid NameError

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

col1, col2, col3 = st.columns(3)
scene_button = col1.button("Describe Scene")
ocr_button = col2.button("Extract Text")
tts_button = col3.button("Text to Speech")

input_prompt = """
You are an AI assistant helping visually individuals by describing the scene in the image. Provide:
1. list of items detected in the image with their purpose.
2. Overall description of the image.
3. Suggestions for actions or precautions for the visually impaired.
"""
if uploaded_file:
    image_data = input_image_setup(uploaded_file)

    if scene_button:
        with st.spinner("Generating scene description..."):
            response = generate_scene_description(input_prompt, image_data)
            st.write(response)

    if ocr_button:
        with st.spinner("Extracting text from Image..."):
            text = extract_text_from_image(image)
            st.write(f"Extracted Text: {text}")  # Show extracted text for debugging
            if text.strip():
                st.success("Text extraction completed.")
            else:
                st.warning("No text found in the image.")

    if tts_button:
        with st.spinner("Converting text to speech..."):
            if text and text.strip():  # Ensure there is some text before triggering TTS
                text_to_speech(text)
                st.success("Text to Speech conversion completed!")
            else:
                st.warning("No text to convert to speech.")
