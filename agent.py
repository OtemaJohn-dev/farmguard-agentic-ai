# import os
import uuid
import smtplib
# import requests
from email.message import EmailMessage
from typing import Optional, Annotated
from typing_extensions import TypedDict
from dotenv import dotenv_values

from twilio.rest import Client

from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    SystemMessage,
    AIMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import pyttsx3


# CONFIG
config = dotenv_values(".env")

llm = ChatGroq(
    api_key=config["GROQ_API_KEY"],
    model=config["GROQ_API_MODEL"],
    temperature=0.3,
)


# MEMORY-CHROMA
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

memory_db = Chroma(
    persist_directory=config.get("CHROMA_DIR", "./chroma_db"),
    embedding_function=embedding,
)

# VOICE ENGINE
engine = pyttsx3.init()
engine.setProperty("rate", 170)


def speak_and_generate_audio(text: str, filename=None):
    if not filename:
        filename = f"audio_{uuid.uuid4().hex}.wav"
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename


# EMAIL
def send_email(subject, body, receiver):
    msg = EmailMessage()
    msg["From"] = config["EMAIL_USER"]
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(config["EMAIL_USER"], config["EMAIL_PASS"])
        smtp.send_message(msg)


# AI PHONE CALL + SMS
def send_sms(message, phone):
    client = Client(config["TWILIO_SID"], config["TWILIO_AUTH"])
    client.messages.create(
        body=message,
        from_=config["TWILIO_PHONE"],
        to=phone,
    )


def ai_make_call(message, phone):
    """
    AI makes call and speaks message itself.
    """
    client = Client(config["TWILIO_SID"], config["TWILIO_AUTH"])

    twiml = f"""
    <Response>
        <Say voice="alice">{message}</Say>
    </Response>
    """

    client.calls.create(
        twiml=twiml,
        from_=config["TWILIO_PHONE"],
        to=phone,
    )


# TOOLS
@tool
def store_farmer_profile(info: str) -> str:
    """Store farmer information in long term memory."""
    memory_db.add_texts([info])
    return "Farmer profile stored."


@tool
def retrieve_memory(query: str) -> str:
    """Retrieve stored farmer information."""
    docs = memory_db.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])


@tool
def send_alert(
    message: str, email: Optional[str] = None, phone: Optional[str] = None
):
    """Send AI alert via Email, SMS and Phone Call."""
    if email:
        send_email("FarmGuard AI Alert", message, email)

    if phone:
        send_sms(message, phone)
        ai_make_call(message, phone)

    return "AI alert dispatched via all channels."


# STATE
class State(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]
    thread_id: str
    user_email: str
    user_phone: str


# MAIN AGENT NODE
def main_node(state: State):
    agent = llm.bind_tools(
        [store_farmer_profile, retrieve_memory, send_alert]
    )

    system_prompt = f"""
You are FarmGuard — an autonomous AI agricultural agent in Uganda.

User email: {state["user_email"]}
User phone: {state["user_phone"]}

Autonomous Rules:
1. Store important farmer data.
2. Retrieve memory before making decisions.
3. If serious risk detected → use send_alert automatically.
4. Escalate urgent risks via AI phone call.
5. Think step by step.
"""

    response = agent.invoke(
        [SystemMessage(content=system_prompt)]
        + state["message"]
    )

    return {"message": [response]}


# TOOL NODE
def tool_node(state: State):
    last_msg: AIMessage = state["message"][-1]
    tool_messages = []

    for tool_call in last_msg.tool_calls:
        name = tool_call["name"]
        args = tool_call.get("args", {}) or {}

        if name == "send_alert":
            args["email"] = state["user_email"]
            args["phone"] = state["user_phone"]

        try:
            if name == "store_farmer_profile":
                result = store_farmer_profile.invoke(args)
            elif name == "retrieve_memory":
                result = retrieve_memory.invoke(args)
            elif name == "send_alert":
                result = send_alert.invoke(args)
            else:
                result = "Unknown tool"

        except Exception as e:
            result = f"Tool failed: {e}"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"message": tool_messages}


def router(state: State):
    last = state["message"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool"
    return END


builder = StateGraph(State)
builder.add_node("main", main_node)
builder.add_node("tool", tool_node)
builder.add_edge(START, "main")
builder.add_conditional_edges("main", router)
builder.add_edge("tool", "main")

compiled = builder.compile()


def generate_response(state: State):
    return compiled.invoke(
        state,
        config={"configurable": {"thread_id": state["thread_id"]}},
    )
