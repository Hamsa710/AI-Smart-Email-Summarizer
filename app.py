import streamlit as st
from transformers import pipeline

# Page Title
st.set_page_config(page_title="Smart Email Summarizer", page_icon="📧")

st.title("📧 AI-Powered Smart Email & Article Summarizer")
st.write("Generate concise summaries from lengthy emails and articles using NLP.")

# Load Model
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

summarizer = load_model()

# Summary Length Selection
length = st.selectbox(
    "Summary Length",
    ["Short", "Medium", "Long"]
)

if length == "Short":
    max_len = 50
elif length == "Medium":
    max_len = 100
else:
    max_len = 150

# File Upload
uploaded_file = st.file_uploader(
    "Upload Text File",
    type=["txt"]
)

default_text = ""

if uploaded_file is not None:
    default_text = uploaded_file.read().decode("utf-8")

# Text Area
text = st.text_area(
    "Paste Email or Article Here",
    value=default_text,
    height=300
)

# Generate Summary Button
if st.button("Generate Summary"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:

        with st.spinner("Generating Summary..."):

            original_words = len(text.split())

            summary = summarizer(
                text,
                max_length=max_len,
                min_length=20,
                do_sample=False
            )

            summary_text = summary[0]["summary_text"]

            summary_words = len(summary_text.split())

        st.success("Summary Generated Successfully!")

        st.subheader("📄 Summary")
        st.download_button(
        label="📥 Download Summary",
        data=summary_text,
        file_name="summary.txt",
        mime="text/plain"
)

        st.markdown("---")

        st.subheader("📊 Statistics")
        st.write(f"Original Words: {original_words}")
        st.write(f"Summary Words: {summary_words}")

        reduction = round(
            ((original_words - summary_words) / original_words) * 100,
            2
        )

        st.write(f"Text Reduction: {reduction}%")