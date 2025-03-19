const API_URL = "https://banking-chatbot-k0qe.onrender.com/chat";
const STT_API_URL = "https://banking-chatbot-k0qe.onrender.com/stt";
const TTS_API_URL = "https://banking-chatbot-k0qe.onrender.com/tts";

let isRecording = false;
let mediaRecorder;
let audioChunks = [];

// 🎤 Toggle Voice Recording
function toggleRecording() {
    const recordButton = document.getElementById("recordButton");

    if (!isRecording) {
        startRecording();
        recordButton.innerText = "⏹ Stop Voice";
    } else {
        stopRecording();
        recordButton.innerText = "🎤 Start Voice";
    }
    isRecording = !isRecording;
}

// 🎤 Start Voice Recording
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            audioChunks = [];
            processSpeech(audioBlob);
        };

        mediaRecorder.start();
    } catch (error) {
        console.error("Error accessing microphone:", error);
    }
}

// ⏹ Stop Recording & Send to STT API
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
}

// 🎤 Convert Speech to Text (STT)
async function processSpeech(audioBlob) {
    showLoading(true);

    const formData = new FormData();
    formData.append("file", audioBlob);

    try {
        const response = await fetch(STT_API_URL, {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        document.getElementById("userInput").value = data.transcription;
        sendMessage();
    } catch (error) {
        console.error("STT Error:", error);
        alert("Error converting speech to text.");
    }

    showLoading(false);
}

// ✉️ Send Text Message to Chatbot
async function sendMessage() {
    const customerId = document.getElementById("customerId").value;
    const userInput = document.getElementById("userInput").value;
    const chatbox = document.getElementById("chatbox");

    if (!customerId || !userInput) {
        alert("Please enter your Customer ID and message.");
        return;
    }

    // Display User Message
    chatbox.innerHTML += `<div class="chat-message user-message">You: ${userInput}</div>`;

    showLoading(true);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ customer_id: customerId, message: userInput })
        });
        const data = await response.json();

        // Display Bot Response
        chatbox.innerHTML += `<div class="chat-message">💬 Bot: ${data.response}</div>`;

        // 🎤 Convert Response to Speech (TTS)
        textToSpeech(data.response);
    } catch (error) {
        console.error("Error:", error);
        chatbox.innerHTML += `<div class="chat-message error">❌ Error: Unable to fetch response.</div>`;
    }

    document.getElementById("userInput").value = "";
    showLoading(false);
}

// 🔊 Convert Bot Response to Speech (TTS)
async function textToSpeech(text) {
    showLoading(true);

    try {
        const response = await fetch(TTS_API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({ text: text })
        });

        const data = await response.json();
        const audio = new Audio(data.audio_url);
        audio.play();
    } catch (error) {
        console.error("TTS Error:", error);
        alert("Error generating speech.");
    }

    showLoading(false);
}

// ⏳ Show/Hide Loading Indicator
function showLoading(isLoading) {
    document.getElementById("loadingIndicator").classList.toggle("hidden", !isLoading);
}