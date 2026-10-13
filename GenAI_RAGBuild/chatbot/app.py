
# ============================================================
# TAMIL NADU AGRICULTURE ASSISTANT
# Final Streamlit frontend
# ============================================================

from pathlib import Path
import streamlit as st

# Connect to the existing RAG backend.
from rag_chatbot import ask_question


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tamil Nadu Agriculture Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Project folders.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"

# Processed documents folder.
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Find the available local Tamil Nadu logo.
LOGO_CANDIDATES = [
    ASSETS_DIR / "tn_agriculture_logo.png",
    ASSETS_DIR / "tamil_nadu_logo.png",
    ASSETS_DIR / "logo.png",
]

LOGO_IMAGE = next(
    (path for path in LOGO_CANDIDATES if path.is_file()),
    None,
)

# Farmer photograph.
FARMER_IMAGE = ASSETS_DIR / "farmer_tamil_nadu.png"


# ============================================================
# 2. DESIGN AND STYLING
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --forest: #17452f;
        --green: #397b4e;
        --gold: #e8c56f;
        --cream: #f4f5ed;
        --paper: #fffefa;
        --border: #dce5d5;
        --ink: #28392d;
        --muted: #687568;
    }

    .stApp {
        background: var(--cream);
        color: var(--ink);
    }

    .block-container {
        max-width: 1600px;
        padding: 0.65rem 1rem 0.5rem;
    }

    /* Hide the toolbar where supported by the Streamlit version. */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0;
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    /* Header branding. */
    .brand-title {
        color: var(--forest);
        font-size: clamp(18px, 1.8vw, 26px);
        font-weight: 800;
        line-height: 1.4;
    }

    .brand-subtitle {
        color: var(--muted);
        font-size: 10px;
        letter-spacing: 0.5px;
    }

    /* Green and gold banner. */
    .hero-panel {
        background: linear-gradient(
            135deg,
            #17452f 0%,
            #245c3c 55%,
            #397b4e 100%
        );
        border: 1px solid #245c3c;
        border-radius: 18px;
        padding: 22px;
        color: #ffffff;
    }

    .hero-kicker {
        color: #ffe19a !important;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.1px;
    }

    .hero-title {
        color: #ffffff !important;
        font-size: clamp(24px, 2vw, 33px);
        font-weight: 800;
        line-height: 1.3;
        margin-top: 14px;
    }

    .hero-copy {
        color: #ffffff !important;
        font-size: 13px;
        line-height: 1.8;
    }

    .hero-rule {
        width: 58px;
        height: 4px;
        border-radius: 4px;
        background: var(--gold);
        margin: 15px 0;
    }

    .section-heading {
        color: var(--forest);
        font-size: 17px;
        font-weight: 800;
    }

    .small-note {
        color: var(--muted) !important;
        font-size: 11px;
        line-height: 1.7;
    }

    /* Horizontal menu buttons. */
    div.stButton > button {
        width: 100%;
        min-height: 40px;
        padding: 0.4rem 0.45rem;
        background: var(--paper);
        color: var(--forest);
        border: 1px solid var(--border);
        border-radius: 10px;
        font-size: 12px;
        font-weight: 650;
        box-shadow: none;
        white-space: normal;
    }

    div.stButton > button:hover {
        background: #e7efdf;
        color: var(--forest);
        border-color: #8eae88;
    }

    /* Compact chat messages. */
    div[data-testid="stChatMessage"] {
        background: var(--paper);
        border: 1px solid #e3e8dc;
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li {
        font-size: 13px;
        line-height: 1.65;
    }

    /* Question input. */
    div[data-testid="stTextInput"] input {
        min-height: 44px;
        background: var(--paper);
        color: var(--ink);
        border: 1px solid #cbd9c5;
        border-radius: 10px;
        font-size: 14px;
    }

    /* Send button. */
    div[data-testid="stFormSubmitButton"] button {
        min-height: 44px;
        background: var(--forest);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-weight: 750;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: var(--green);
        color: #ffffff;
    }

    div[data-testid="stForm"] {
        border: none;
        padding: 0;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border);
        border-radius: 13px;
    }

    [data-testid="stImage"] img {
        border-radius: 12px;
    }

    @media (max-width: 800px) {
        .block-container {
            padding: 0.4rem;
        }

        .hero-panel {
            padding: 16px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. SESSION STATE
# ============================================================

# Retain messages during Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Store a question selected from the menu.
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# 4. HEADER WITH TAMIL NADU LOGO
# ============================================================

with st.container(border=True):

    logo_col, title_col = st.columns(
        [1, 12],
        vertical_alignment="center",
    )

    with logo_col:
        if LOGO_IMAGE:
            st.image(str(LOGO_IMAGE), width=58)
        else:
            st.markdown("## 🌾")

    with title_col:
        st.markdown(
            '<div class="brand-title">'
            'தமிழ்நாடு வேளாண்மை உதவி மையம்'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="brand-subtitle">'
            'TAMIL NADU AGRICULTURE ASSISTANT · AI SCHEME GUIDANCE'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# 5. HORIZONTAL SCHEME MENU
# ============================================================

st.markdown(
    '<div class="section-heading">'
    '🌱 Explore farmer support'
    '</div>',
    unsafe_allow_html=True,
)

st.caption("Choose a topic to ask a suggested question.")

menu_items = [
    (
        "🌱 Crop & seed schemes",
        "What seed subsidies are available for farmers?",
    ),
    (
        "💰 Subsidies & assistance",
        "What subsidies are available for small farmers?",
    ),
    (
        "💧 Water & irrigation",
        "What irrigation support schemes are available?",
    ),
    (
        "🎓 Training & guidance",
        "What farmer training schemes are available?",
    ),
    (
        "📞 Contact & Help",
        "__CONTACT_HELP__",
    ),
]

# Display five buttons horizontally.
menu_columns = st.columns(
    [1.0, 1.0, 0.95, 0.95, 1.0],
    gap="small",
)

for index, (label, prompt) in enumerate(menu_items):

    with menu_columns[index]:

        if st.button(
            label,
            key=f"horizontal_menu_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_question = prompt
            st.rerun()


# ============================================================
# 6. MAIN SPLIT LAYOUT
# ============================================================

left_col, right_col = st.columns(
    [0.88, 1.55],
    gap="medium",
)


# ============================================================
# 7. LEFT COLUMN: BANNER AND FARMER PHOTOGRAPH
# ============================================================

with left_col:

    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">
                FARMER SUPPORT · விவசாயி நலன்
            </div>
            <div class="hero-title">
                Growing a greener Tamil Nadu 🌱
            </div>
            <div class="hero-rule"></div>
            <div class="hero-copy">
                Your friendly agriculture information assistant.
                <br><br>
                Explore government scheme information, seed support,
                subsidies, irrigation assistance and farmer training.
                <br><br>
                Ask a question or choose a topic from the menu above.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show the local farmer photograph.
    if FARMER_IMAGE.is_file():
        st.image(
            str(FARMER_IMAGE),
            use_container_width=True,
        )
    else:
        with st.container(border=True):
            st.markdown("### 🌾 Better information. Better farming.")
            st.caption(
                "Place the farmer photograph at "
                "assets/farmer_tamil_nadu.png to display it here."
            )

    # Information card.
    with st.container(border=True):
        st.markdown("🌾 🌿 💧")
        st.markdown("**Better information. Better farming.**")
        st.caption(
            "Explore agriculture documents through a simple conversation."
        )

    # Disclaimer.
    st.markdown(
        """
        <div class="small-note">
            Confirm current eligibility, benefit amounts and deadlines
            with the relevant government department.
            <br><br>
            This is an informational project, not an official government website.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 8. RIGHT COLUMN: FIXED-HEIGHT CHAT WINDOW
# ============================================================

with right_col:

    st.markdown("## 💬 Agriculture Assistant")

    st.caption(
        "Ask in English or Tamil about schemes, subsidies, seeds, "
        "irrigation and farmer training."
    )

    # Messages scroll inside this fixed-height panel.
    with st.container(height=430, border=True):

        # Welcome message for a new conversation.
        if not st.session_state.messages:

            with st.chat_message("assistant", avatar="🌾"):
                st.markdown(
                    """
                    **Vanakkam! வணக்கம்! 👋**

                    Welcome to your Tamil Nadu Agriculture Assistant.

                    Type your question below or choose a topic from the menu.

                    **Try asking:**
                    - What seed subsidies are available?
                    - What irrigation schemes are mentioned?
                    - What support is available for small farmers?
                    - What farmer training schemes are available?
                    """
                )

        # Render the stored conversation.
        for message in st.session_state.messages:

            avatar = (
                "🌾"
                if message["role"] == "assistant"
                else "👨‍🌾"
            )

            with st.chat_message(
                message["role"],
                avatar=avatar,
            ):
                st.markdown(message["content"])


# ============================================================
# 9. QUESTION INPUT BELOW THE CHAT WINDOW
# ============================================================

with right_col:

    with st.form(
        "agriculture_question_form",
        clear_on_submit=True,
        border=False,
    ):

        input_col, send_col = st.columns(
            [6, 1],
            vertical_alignment="bottom",
        )

        with input_col:
            typed_question = st.text_input(
                "Your question",
                placeholder=(
                    "Type your question here... / "
                    "உங்கள் கேள்வியை உள்ளிடவும்"
                ),
                label_visibility="collapsed",
            )

        with send_col:
            submitted = st.form_submit_button(
                "Send ➤",
                use_container_width=True,
            )


# ============================================================
# 10. DETERMINE WHICH QUESTION TO PROCESS
# ============================================================

question = ""

# Process a typed question when Send is clicked.
if submitted and typed_question.strip():
    question = typed_question.strip()

# Otherwise, process the selected menu question.
if not question and st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None


# ============================================================
# 11. PROCESS THE QUESTION
# ============================================================

if question:

    # Display a friendly label for the Help menu.
    if question == "__CONTACT_HELP__":
        display_question = "Contact & Help"
    else:
        display_question = question

    # Add the user's question to the conversation.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": display_question,
        }
    )

    # --------------------------------------------------------
    # 11A. CONTACT AND HELP RESPONSE
    # --------------------------------------------------------

    if question == "__CONTACT_HELP__":

        answer = (
            "**Tamil Nadu Agriculture Assistant — Contact & Help**\n\n"
            "Welcome! Explore information about agriculture schemes "
            "and farmer support using this assistant.\n\n"
            "**🌱 Topics we can help with**\n"
            "- Crop and seed schemes\n"
            "- Seed distribution and subsidies\n"
            "- Farmer assistance and equipment subsidies\n"
            "- Irrigation and water-saving schemes\n"
            "- Farmer training and cultivation guidance\n\n"
            "**📧 Email Support**\n\n"
            "tamilnaduulavan@gmail.com\n\n"
            "**📞 Phone Support**\n\n"
            "+91 9922012637\n\n"
            "**How to get a useful answer**\n\n"
            "Mention your crop, the type of assistance you need, "
            "or the scheme name. The assistant will search the "
            "available project documents for relevant information.\n\n"
            "Please confirm scheme eligibility and current benefits "
            "with the relevant government department. This is an "
            "informational project, not an official government website."
        )

    # --------------------------------------------------------
    # 11B. NORMAL RAG QUESTION
    # --------------------------------------------------------

    else:

        try:
            with st.spinner("Searching agriculture documents..."):

                # Use the existing RAG backend and FAISS index.
                result = ask_question(question)

            status = result.get("status", "error")

            answer = result.get(
                "answer",
                "Sorry, I could not prepare an answer.",
            )

            if status == "no_evidence":
                answer = (
                    "I couldn't find enough supporting information "
                    "in the available documents. Try a more specific "
                    "question about a crop, scheme, subsidy or type "
                    "of assistance."
                )

            elif status == "error":
                answer = (
                    "I couldn't complete your request. Please try again. "
                    "If the problem continues, check the PowerShell "
                    "terminal for the backend error."
                )

        except Exception as error:

            # Technical details appear in the terminal, not the chat.
            print(
                f"RAG request failed: "
                f"{type(error).__name__}: {error}"
            )

            answer = (
                "Something went wrong while searching the documents. "
                "Please try again. Check the PowerShell terminal "
                "if the issue continues."
            )

    # Save the answer in the chat history.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    # Refresh the page to display the latest response.
    st.rerun()


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.caption(
    "🌿 Tamil Nadu Agriculture Assistant · "
    "Document-based information support"
)

st.caption(
    "This is an informational project, not an official government website."
)
