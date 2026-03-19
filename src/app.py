import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="Multimodal Image Agent", layout="wide")
st.title("🖼️ Multimodal AI Agent - Image Q&A")

# Configure Gemini
@st.cache_resource
def configure_gemini():
    """Initialize Gemini API"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found! Please add it in Render dashboard.")
        st.stop()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')  # Fast & free
    return model

# Initialize model
model = configure_gemini()

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for image upload
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.session_state.current_image = image
        st.success("✅ Image loaded successfully!")

# Main chat interface
if 'current_image' in st.session_state:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the image..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get Gemini response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing image..."):
                try:
                    # Prepare the prompt with image
                    response = model.generate_content([prompt, st.session_state.current_image])
                    
                    # Display response
                    st.markdown(response.text)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response.text}
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.info("👈 Please upload an image from the sidebar to start asking questions!")