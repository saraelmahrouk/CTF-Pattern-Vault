import os
import time
import streamlit as st
from vectorstore import load
from rag_chain import build_rag_chain, build_hint_chain, build_coaching_chain
from ingestion import add_user_writeup
from config import CORPUS_PATH

# ==========================================================
# 1. PIXEL THEME & CSS INJECTION (Including Sidebar Styling)
# ==========================================================
def apply_pixel_theme():
    pixel_css = """
    <style>
    /* 1. Import Pixel Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Silkscreen:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');
    /* 2. Global Variables */
    :root {
        --pixel-bg: #0d1117;
        --pixel-border: #062d17;
        --pixel-card-bg: #161b22;
        --pixel-shadow: #000000;
    }

    @font-face {
        font-family: 'PlumpPixel';
        src: url('app/static/PlumpPixel.ttf') format('truetype');
    }

    /* Update this selector inside your CSS string */
    [data-testid="stAppViewContainer"] {
        /* linear-gradient creates a dark overlay so text remains readable over the image */
        background-image: linear-gradient(rgba(13, 17, 23, 0.50), rgba(13, 17, 23, 0.50)), 
                        url("app/static/bgvid.gif") !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }


    /* Make top Streamlit header bar transparent so the image extends to the top */
    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0) !important;
    }

    .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp code, .stApp pre,
    [data-testid="stMarkdownContainer"] * {
        font-family: 'Silkscreen', monospace !important;
    }

    h1, h2, h3, h4, h5, h6, 
    h1 *, h2 *, h3 *, h4 * {
        font-family: 'Press Start 2P', monospace !important;
    }


    /* 4. CRT Scanline Overlay Effect */
    [data-testid="stAppViewContainer"]::before {
        content: " ";
        display: block;
        position: absolute;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%);
        background-size: 100% 4px;
        z-index: 2;
        pointer-events: none;
        opacity: 0.6;
    }

    /* 5. SIDEBAR & SIDE TAB STYLING */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        color: #7dddfa !important;
        display: none !important;
    }


    /* Style the menu items into full-width navigation bars */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
        border: none !important;
        color: #7dddfa !important;
        border-left: 4px solid transparent !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        border-left: 4px solid #ffaded !important;
    }

    /* Hover effect */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        transform: translateX(4px);
        background-color: #ffaded !important;
    }

    /* Hide the radio button circles/dots entirely */
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] {
        display: none !important;
    }
    /* Active Selected Tab (Pixel version of the highlighted tab in your image) */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background-color: #ffaded !important;
        border-color: #ffaded !important;
        box-shadow: inset 2px 2px 0px #000000 !important;
    }

    /* Force text inside the selected active tab to turn black */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) * {
        color: #000000 !important;
        font-weight: bold !important;
    }

    /* 6. Stepped Pixelated Cards */
    div[data-testid="stVerticalBlock"] > div.css-card {
        background-color: var(--pixel-card-bg);
        border: 4px solid var(--pixel-border);
        box-shadow: 
            4px 4px 0px 0px var(--pixel-shadow),
            inset -4px -4px 0px 0px #052e16;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* 7. 8-Bit Pixelated Buttons */
    .stButton > button {
        font-family: 'Press Start 2P', monospace !important;
        font-size: 11px !important;
        background-color: #ff7de3 !important;
        color: #000000 !important;
        border: 3px solid #000000 !important;
        border-radius: 0px !important;
        padding: 12px 20px !important;
        box-shadow: 4px 4px 0px 0px #000000 !important;
        transition: transform 0.1s ease, box-shadow 0.1s ease !important;
    }

    .stButton > button:hover {
        background-color: #ffaded !important;
        transform: translate(-2px, -2px);
        box-shadow: 6px 6px 0px 0px #000000 !important;
    }

    /* 8. Pixelated Text Inputs & Textareas */
    .stTextInput input, .stTextArea textarea {
        font-family: 'Silkscreen', monospace !important;
        background-color: #000000 !important;
        color: #ffaded !important;
        border: 3px solid var(--pixel-border) !important;
        border-radius: 0px !important;
        box-shadow: inset 3px 3px 0px #052e16 !important;
    }

    [data-testid="stFileUploaderDropzone"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
    gap: 10px !important;
    padding: 12px !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        margin-top: 8px !important;
        position: static !important;
    }

    [data-testid="stFormSubmitButton"] button {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 11px !important;
    background-color: #ff7de3 !important;
    color: #000000 !important;
    border: 3px solid #000000 !important;
    border-radius: 0px !important;
    padding: 12px 20px !important;
    box-shadow: 4px 4px 0px 0px #000000 !important;
    transition: transform 0.1s ease, box-shadow 0.1s ease !important;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #ffaded !important;
        transform: translate(-2px, -2px);
        box-shadow: 6px 6px 0px 0px #000000 !important;
    }

    .stTextInput input, .stTextArea textarea {
    font-family: 'Silkscreen', monospace !important;
    background-color: #000000 !important;
    color: #cfccce !important;
    border: 3px solid var(--pixel-border) !important;
    border-radius: 0px !important;
    box-shadow: inset 3px 3px 0px #052e16 !important;
    }   

    /* File uploader dropzone — same look as text inputs */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #000000 !important;
        border: 3px solid var(--pixel-border) !important;
        border-radius: 0px !important;
        box-shadow: inset 3px 3px 0px #052e16 !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #ffaded !important;
        font-family: 'Silkscreen', monospace !important;
    }

    /* Sidebar container — same look as text inputs */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 3px solid var(--pixel-border) !important;
        box-shadow: inset 3px 3px 0px #052e16 !important;
    }

    .logo-title, .logo-title span {
        font-family: 'PlumpPixel', monospace !important;
        font-size: 48px;
        color: #a14dbf;
        text-shadow: 7px 7px 0px #000000;
        letter-spacing: 2px;
        text-align: center;
        width: 100%;
    }

    [data-testid="stTextInputRootElement"],
    [data-testid="stTextInputRootElement"] *,
    [data-testid="stTextInputRootElement"]:focus,
    [data-testid="stTextInputRootElement"]:focus-within,
    [data-testid="stTextInputRootElement"][data-focused="true"],
    .stTextInput input,
    .stTextInput input:focus,
    .stTextInput input:focus-visible,
    .stTextArea textarea,
    .stTextArea textarea:focus,
    .stTextArea textarea:focus-visible {
        outline: none !important;
        box-shadow: inset 3px 3px 0px #052e16 !important;
        border-color: var(--pixel-border) !important;
    }

    *:focus {
        outline: none !important;
    }

    [data-testid="stSidebar"] h1 {
        color: #a14dbf !important;
    }

    /* Hide Default Header/Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    span[data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Outlined' !important;
        font-size: inherit;
    }
    </style>

    """
    st.markdown(pixel_css, unsafe_allow_html=True)

# Apply theme immediately on startup
apply_pixel_theme()

# ==========================================================
# 2. VECTORSTORE & CHAIN INITIALIZATION
# ==========================================================
INDEX_FILE = os.path.join(CORPUS_PATH, "index.faiss")

@st.cache_resource
def load_everything():
    if not os.path.exists(INDEX_FILE):
        st.error(f"No corpus found at {CORPUS_PATH}. Run `python build_corpus.py` first.")
        st.stop()

    vectorstore = load(CORPUS_PATH)
    rag_chain = build_rag_chain(vectorstore)
    hint_chain = build_hint_chain(vectorstore)
    coaching_chain = build_coaching_chain(vectorstore)
    return vectorstore, rag_chain, hint_chain, coaching_chain

vectorstore, rag_chain, hint_chain, coaching_chain = load_everything()

# ==========================================================
# 3. SIDEBAR NAVIGATION ("SIDE TABS")
# ==========================================================
with st.sidebar:
    st.markdown('<h1 class="logo-title">CTF COACH</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    selected_tab = st.radio(
        "NAVIGATION",
        ["Direct Help", "Hints", "Coaching", "Upload Writeup"],
        key="side_nav"
    )

# ==========================================================
# 4. MAIN CONTENT AREA (Based on active side tab)
# ==========================================================
st.markdown('<h1 class="logo-title">CTF PATTERN VAULT</h1>', unsafe_allow_html=True)

if selected_tab == "Direct Help":
    st.subheader("[+] ASK A DIRECT QUESTION")
    question = st.text_input("What are you stuck on?", key="direct_question")
    if st.button("GET ANSWER", key="direct_button"):
        def response_generator():
            for chunk in rag_chain.stream(question):
                yield chunk
        st.write_stream(response_generator)

elif selected_tab == "Hints":
    st.subheader("[+] GRADUATED HINTS")
    hint_question = st.text_input("What topic are you stuck on?", key="hint_question")

    if st.button("GET HINTS", key="hint_button"):
        with st.spinner("GENERATING HINTS..."):
            result = hint_chain.invoke(hint_question)
        st.session_state["hint_result"] = result
        st.session_state["hint_level"] = 1  # start by showing just tier 1

    if "hint_result" in st.session_state:
        result = st.session_state["hint_result"]
        level = st.session_state["hint_level"]

        st.markdown(f"**Category hint:** {result['category_hint']}")

        if level >= 2:
            st.markdown(f"**Tool hint:** {result['tool_hint']}")
        if level >= 3:
            st.markdown(f"**Walkthrough hint:** {result['walkthrough_hint']}")

        if level < 3:
            if st.button("SHOW NEXT HINT", key="next_hint_button"):
                st.session_state["hint_level"] += 1
                st.rerun()

elif selected_tab == "Coaching":
    st.subheader("[+] TOPIC COACHING")
    topic = st.text_input("What topic do you want to learn about?", key="topic_question")
    if st.button("GET SUMMARY", key="coaching_button"):
        def response_generator():
            for chunk in coaching_chain.stream(topic):
                yield chunk
            time.sleep(0.20)
        st.write_stream(response_generator)

elif selected_tab == "Upload Writeup":
    st.subheader("[+] ADD WRITEUP TO CORPUS")

    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Or upload a file", type=["md", "txt"], key="writeup_file")
        label = st.text_input("Give this writeup a label/name", key="writeup_label")
        writeup_text = st.text_area("Or paste your writeup here", key="writeup_text", height=300)
        submitted = st.form_submit_button("SUBMIT WRITEUP")

    if submitted:
        if uploaded_file is not None:
            final_text = uploaded_file.read().decode("utf-8", errors="replace")
            final_label = label or uploaded_file.name
        else:
            final_text = writeup_text
            final_label = label
        doc, result, error = add_user_writeup(vectorstore, final_text, final_label)
        if error:
            st.error(error)
        else:
            st.success("Writeup added successfully!")
            st.markdown(f"**Vulnerability:** {result['vulnerability_class']}")
            tools = result['tools_used']
            st.markdown(f"**Tools:** {', '.join(tools) if isinstance(tools, list) else tools}")
            st.markdown(f"**Difficulty:** {result['difficulty']}")
            st.markdown(f"**Key insight:** {result['key_insight']}")