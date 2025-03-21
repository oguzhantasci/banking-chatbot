const API_URL = "https://banking-chatbot-k0qe.onrender.com/chat";
const WS_URL = "https://banking-chatbot-k0qe.onrender.com/ws";

let websocket;
let isVoiceActive = false;
let mediaRecorder;
let audioChunks = [];

// 🎤 Start Real-time Voice Chat
function toggleVoiceChat() {
    if (!isVoiceActive) {
        websocket = new WebSocket(WS_URL);
        websocket.binaryType = "arraybuffer";

        websocket.onopen = () => {
            alert("🎤 Voice Chat Started! Speak into your mic.");
            startRecording();
        };

        websocket.onmessage = (event) => {
            playAudio(event.data);
        };

        isVoiceActive = true;
    } else {
        stopRecording();
        websocket.close();
        isVoiceActive = false;
        alert("❌ Voice Chat Stopped!");
    }
}

// 🎙 Start browser recording and send to WebSocket
function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (e) => {
            audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            audioChunks = [];

            if (websocket.readyState === WebSocket.OPEN) {
                const arrayBuffer = await audioBlob.arrayBuffer();
                websocket.send(arrayBuffer);
            }

            if (isVoiceActive) {
                // Continue recording for next message
                startRecording();
            }
        };

        mediaRecorder.start();

        // Stop recording after 5 seconds
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();
            }
        }, 5000);
    });
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
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
