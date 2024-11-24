import streamlit as st
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage
import pytesseract
from gtts import gTTS
import io
import base64
import logging
import os

# Static Google API Key (ensure it's secure in production)
GOOGLE_API_KEY = "yours_api_key"

# Initialize models through LangChain with correct model names
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
vision_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# Error handling function
def handle_error(error):
    logging.error(error)
    st.error(f"Error: {str(error)}")

# Scene understanding function
def scene_understanding(image):
    try:
        # Generate detailed scene description using LangChain Vision model
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()

        message = HumanMessage(
            content=[{
                "type": "text",
                "text": """As an AI assistant for visually impaired individuals, provide a detailed description of this image.
                Include:
                1. Overall scene layout
                2. Main objects and their positions
                3. People and their activities (if any)
                4. Colors and lighting
                5. Notable features or points of interest

                Format the response in clear, easy-to-understand sections."""
            }, {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
            }]
        )

        response = vision_llm.invoke([message])
        return response.content
    except Exception as e:
        handle_error(e)

# Function to extract text from image using Tesseract OCR
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text.strip() if text else None
    except Exception as e:
        handle_error(e)
        return None

# Function to convert text to speech using gTTS
def text_to_speech(text):
    try:
        tts = gTTS(text, lang="en")
        audio_file = "output.mp3"
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        return f"Error generating speech: {str(e)}"

# Streamlit configuration
st.set_page_config(page_title="AI Visual Assistant", page_icon="🎨", layout="wide")

# Custom CSS for professional website design with animated navbar and footer
st.markdown("""
    <style>
        body {
            background: linear-gradient(to bottom right, #f8fafc, #e9eff5);
            font-family: 'Arial', sans-serif;
        }

        /* Navigation Bar Styles */
        .navbar {
            background-color: #1d3557;
            color: white;
            padding: 10px 20px;
            text-align: center;
            font-size: 20px;
            position: sticky;
            top: 0;
            width: 100%;
            z-index: 9999;
        }

        .navbar a {
            color: white;
            text-decoration: none;
            margin: 0 20px;
            font-weight: bold;
        }

        .navbar a:hover {
            color: #f1faee;
            transition: 0.3s;
        }

        /* Footer Styles with Animation */
        .footer {
            text-align: center;
            color: #6c757d;
            font-size: 14px;
            margin-top: 30px;
            padding: 20px 0;
            border-top: 5px solid #1d3557;
            background: linear-gradient(90deg, #1d3557 25%, #457b9d 50%, #1d3557 75%);
            animation: colorAnimation 5s infinite linear;
        }

        @keyframes colorAnimation {
            0% {
                background: linear-gradient(90deg, #1d3557 25%, #457b9d 50%, #1d3557 75%);
            }
            50% {
                background: linear-gradient(90deg, #457b9d 25%, #1d3557 50%, #457b9d 75%);
            }
            100% {
                background: linear-gradient(90deg, #1d3557 25%, #457b9d 50%, #1d3557 75%);
            }
        }

        .card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .card-title {
            color: #1d3557;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("""
    <div class="navbar">
        <a href="#image-description">Image Description</a>
        <a href="#ocr-text-to-speech">OCR and Text-to-Speech</a>
    </div>
""", unsafe_allow_html=True)

# App Title and Subtitle
st.markdown("<div class='main-title'>AI Visual Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Empowering users with AI-driven image descriptions and text-to-speech conversion.</div>", unsafe_allow_html=True)

# Layout with two sections
col1, col2 = st.columns(2)

# Left Column: Image Description
with col1:
    st.markdown("<div class='card' id='image-description'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Image Description</div>", unsafe_allow_html=True)
    st.write("Upload an image, and the app will generate a description of the scene, including actions, emotions, and visual elements.")

    uploaded_file = st.file_uploader("Upload an image for description...", type=["jpg", "jpeg", "png"], label_visibility="visible")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Generating description..."):
            description = scene_understanding(image)
            st.subheader("Generated Description:")
            st.write(description)

    st.markdown("</div>", unsafe_allow_html=True)

# Right Column: OCR and Text-to-Speech
with col2:
    st.markdown("<div class='card' id='ocr-text-to-speech'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>OCR and Text-to-Speech</div>", unsafe_allow_html=True)
    st.write("Upload an image with text, and the app will extract the text and convert it to speech.")

    ocr_uploaded_file = st.file_uploader("Upload an image with text...", type=["jpg", "jpeg", "png"], label_visibility="visible")

    if ocr_uploaded_file:
        image = Image.open(ocr_uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Extracting text..."):
            text = extract_text_from_image(image)

        if text:
            st.subheader("Extracted Text:")
            st.write(text)

            with st.spinner("Converting text to speech..."):
                audio_file = text_to_speech(text)

            if os.path.exists(audio_file):
                st.subheader("Audio Playback:")
                audio = open(audio_file, "rb")
                st.audio(audio, format="audio/mp3")
                audio.close()
                os.remove(audio_file)
        else:
            st.warning("No text found in the image. Please try another image with visible text.")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>🔍 Powered by Tesseract OCR, gTTS, and Google's Generative AI | Built with ❤ using Streamlit</div>", unsafe_allow_html=True)
