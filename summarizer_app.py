import streamlit as st
import fitz  # PyMuPDF
from google.generativeai import GenerativeModel, configure
import os

# Gemini API Key from Streamlit secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# configure Gemini
if GEMINI_API_KEY:
    configure(api_key=GEMINI_API_KEY)
    model = GenerativeModel("gemini-1.5-flash")
else:
    model = None

st.set_page_config(page_title="🧠 Research Paper Summarizer", layout="wide")

st.title("🔬 Research Paper Summarizer")

st.markdown(
    "Upload your research paper (PDF) and I’ll help you summarize it with Gemini!"
)

uploaded_file = st.file_uploader(
    "Upload a PDF", type=["pdf"], label_visibility="visible"
)

if uploaded_file is not None:
    # Load PDF
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    # extract text
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # preview first 5000 characters
    preview_text = full_text[:5000]
    st.subheader("📄 Extracted Text Preview (first 5000 chars)")
    st.text_area("Paper text preview", preview_text, height=300)

    # extract images
    st.subheader("🖼️ Extracted Figures")
    for page_num, page in enumerate(doc, start=1):
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            st.image(
                image_bytes,
                caption=f"Page {page_num} - Image {img_index}",
                use_container_width=True,
                width=400,  # moderate size
            )

    # summarization
    st.subheader("📝 Gemini Summary")

    if model:
        with st.spinner("Summarizing, please wait..."):
            try:
                response = model.generate_content(
    f"""
    You are an academic research assistant.
    Read the following scientific paper and produce a comprehensive, detailed summary including:
    - Background and motivation of the study
    - Research questions or hypotheses
    - Methodology, data, and experimental design
    - Key results and findings with quantitative details
    - Thorough explanation of all figures, tables, and diagrams, referring to them by their numbers
    - Implications and limitations of the work
    - Suggestions for future research
    - Important references or prior works mentioned
    Write in a clear, structured academic style appropriate for a graduate-level reader, and aim for at least 800–1000 words to ensure depth.
    
    TEXT TO SUMMARIZE:
    {preview_text}
    """
)
                st.success("Summary generated successfully!")
                st.write(response.text)
            except Exception as e:
                st.error(f"Gemini summarization failed: {e}")
    else:
        st.error(
            "🚨 No API key found. Please set GEMINI_API_KEY as a Streamlit secret."
        )

