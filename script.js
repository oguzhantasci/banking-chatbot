const API_URL = "https://banking-chatbot-k0qe.onrender.com/chat";
const WS_URL = "https://banking-chatbot-k0qe.onrender.com/ws";

let websocket;
let isVoiceActive = false;

// 🎤 Start Real-time Voice Chat
function toggleVoiceChat() {
    if (!isVoiceActive) {
        websocket = new WebSocket(WS_URL);
        websocket.onmessage = (event) => playAudio(event.data);
        isVoiceActive = true;
        alert("🎤 Voice Chat Started! Speak into your mic.");
    } else {
        websocket.close();
        isVoiceActive = false;
        alert("❌ Voice Chat Stopped!");
    }
}

// 💬 Send Text Message
function sendMessage() {
    const customerId = document.getElementById("customerId").value;
    const userInput = document.getElementById("userInput").value;
    const chatbox = document.getElementById("chatbox");

    fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ customer_id: customerId, message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        chatbox.innerHTML += `<p>🗣️ You: ${userInput}</p>`;
        chatbox.innerHTML += `<p>💬 Bot: ${data.response}</p>`;
    });
}

// 🔊 Play AI-Generated Speech Response
function playAudio(audioBlob) {
    const audio = new Audio(URL.createObjectURL(new Blob([audioBlob])));
    audio.play();
}