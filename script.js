const API_URL = "https://banking-chatbot-k0qe.onrender.com/chat";  // Replace with your actual API URL

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