const API_URL = "https://banking-chatbot-k0qe.onrender.com/chat";
const TTS_API_URL = "https://banking-chatbot-k0qe.onrender.com/tts"; // Endpoint for Text-to-Speech
const STT_API_URL = "https://banking-chatbot-k0qe.onrender.com/stt"; // Endpoint for Speech-to-Text

function handleKeyPress(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
}

function sendMessage() {
    const customerId = document.getElementById("customerId").value;
    const userInput = document.getElementById("userInput").value;
    const chatbox = document.getElementById("chatbox");

    if (!customerId || !userInput) {
        alert("Please enter your Customer ID and message.");
        return;
    }

    // Show user message
    const userMessage = document.createElement("div");
    userMessage.classList.add("chat-message", "user-message");
    userMessage.innerText = `You: ${userInput}`;
    chatbox.appendChild(userMessage);

    // Send request to backend
    fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ customer_id: customerId, message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        const botMessage = document.createElement("div");
        botMessage.classList.add("chat-message");
        botMessage.innerText = `💬 Bot: ${data.response}`;
        chatbox.appendChild(botMessage);

        // Convert response to speech using TTS
        textToSpeech(data.response);
    })
    .catch(error => {
        console.error("Error:", error);
        const errorMessage = document.createElement("div");
        errorMessage.classList.add("chat-message");
        errorMessage.innerText = "❌ Error: Unable to fetch response.";
        chatbox.appendChild(errorMessage);
    });

    // Clear input field
    document.getElementById("userInput").value = "";
}

// Convert text to speech (TTS)
function textToSpeech(text) {
    fetch(TTS_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        const audio = new Audio(data.audio_url);
        audio.play();
    })
    .catch(error => console.error("TTS Error:", error));
}

// Speech-to-Text (STT) using OpenAI Whisper
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        let audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            const formData = new FormData();
            formData.append("file", audioBlob, "audio.wav");

            try {
                const response = await fetch(STT_API_URL, {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();
                document.getElementById("userInput").value = data.transcription;
                sendMessage(); // Auto-send after transcribing
            } catch (error) {
                console.error("STT Error:", error);
            }
        };

        mediaRecorder.start();

        setTimeout(() => {
            mediaRecorder.stop();
        }, 4000); // Record for 4 seconds

    } catch (error) {
        console.error("Microphone access denied:", error);
    }
}
