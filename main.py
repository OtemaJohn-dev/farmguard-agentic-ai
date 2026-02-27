import streamlit as st
import uuid
import os
import speech_recognition as sr

from langchain_core.messages import HumanMessage, BaseMessage
from agent import generate_response, speak_and_generate_audio

st.set_page_config(page_title="FarmGuard AI", layout="wide")
st.title("🌾 FARMGUARD Autonomous Agentic AI")


if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if "user_phone" not in st.session_state:
    st.session_state.user_phone = ""

if "graph_state" not in st.session_state:
    st.session_state.graph_state = None


# USER REGISTRATION
if not st.session_state.user_email:
    email = st.text_input("📧 Enter your email")
    phone = st.text_input("📱 Enter your phone number")

    if st.button("Start Session"):
        if email and phone:
            st.session_state.user_email = email
            st.session_state.user_phone = phone
            st.session_state.graph_state = {
                "message": [],
                "thread_id": str(uuid.uuid4()),
                "user_email": email,
                "user_phone": phone,
            }
            st.rerun()
        else:
            st.warning("Provide both email and phone.")
    st.stop()


# DISPLAY HISTORY
for message in st.session_state.graph_state["message"]:
    if isinstance(message, BaseMessage):
        role = "assistant" if message.type == "ai" else "user"
        with st.chat_message(role):
            st.write(message.content)


# VOICE INPUT
def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        st.warning("Could not understand audio.")
        return ""
    except sr.RequestError as e:
        st.error(f"Speech recognition API error: {e}")
        return ""
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return ""


col1, col2 = st.columns([1, 4])

with col1:
    voice_btn = st.button("🎤 Speak")

with col2:
    text_input = st.chat_input("Ask about your farm...")

user_input = ""

if voice_btn:
    user_input = get_voice_input()
elif text_input:
    user_input = text_input


# PROCESS INPUT
if user_input:
    human_message = HumanMessage(content=user_input)
    st.session_state.graph_state["message"].append(human_message)

    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("AI thinking..."):
        updated_state = generate_response(st.session_state.graph_state)
        ai_response = updated_state["message"][-1].content

        with st.chat_message("assistant"):
            st.write(ai_response)

            audio_file = speak_and_generate_audio(ai_response)

            if audio_file and os.path.exists(audio_file):
                with open(audio_file, "rb") as f:
                    st.audio(f.read(), format="audio/wav", autoplay=True)

        st.session_state.graph_state = updated_state

    st.rerun()
