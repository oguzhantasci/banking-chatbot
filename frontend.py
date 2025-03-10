import streamlit as st
import requests

API_URL = "https://banking-chatbot-k0qe.onrender.com/chat"

st.title("💳 AI Banking Chatbot")
st.write("Ask about your balance, recent transactions, or perform fund transfers.")

customer_id = st.text_input("Enter your Customer ID:")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Your Message:")
if st.button("Send") and user_input:
    response = requests.post(API_URL, json={"customer_id": customer_id, "message": user_input}).json()
    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Chatbot", response["response"]))

for role, text in st.session_state.messages:
    st.write(f"**{role}:** {text}")
