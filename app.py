
import streamlit as st
import joblib
from pathlib import Path

# Application settings
st.set_page_config(
    page_title="LinguaDetect",
    page_icon="🌍",
    layout="centered"
)

# Supported languages
LANGUAGES = {
    "rus": "Russian 🇷🇺",
    "eng": "English 🇬🇧",
    "deu": "German 🇩🇪",
    "fra": "French 🇫🇷",
    "spa": "Spanish 🇪🇸",
    "ita": "Italian 🇮🇹",
    "por": "Portuguese 🇵🇹",
    "nld": "Dutch 🇳🇱",
    "pol": "Polish 🇵🇱",
    "tur": "Turkish 🇹🇷"
}

# Load trained model
MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "linguadetect_model.joblib"
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model was not found."
        )
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Model loading error: {error}")
    st.stop()

# Interface
st.title("🌍 LinguaDetect")
st.subheader("AI Language Identification System")

st.write(
    "Enter a text and let LinguaDetect "
    "identify its language."
)

st.divider()

user_text = st.text_area(
    "Enter your text:",
    height=180,
    placeholder="Type or paste your text here..."
)

if st.button("🔍 Detect Language", type="primary",
             use_container_width=True):

    if not user_text.strip():
        st.warning("Please enter some text first.")

    else:
        try:
            prediction = model.predict([user_text])[0]
            language = LANGUAGES.get(
                prediction,
                "Unknown language"
            )

            st.success("Language successfully detected!")
            st.markdown("### Detected language")
            st.title(language)

        except Exception as error:
            st.error(f"Detection error: {error}")

if st.button("Clear text", use_container_width=True):
    st.rerun()

st.divider()

st.caption(
    "LinguaDetect | 10 languages | "
    "TF-IDF + LinearSVC"
)
