const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatBox = document.getElementById("chat-box");
const customerIdInput = document.getElementById("customerId");
const recordButton = document.getElementById("record-button");
const audioPlayer = document.getElementById("audio-player");

const backendHost = "https://banking-chatbot-k0qe.onrender.com";
let mediaRecorder;
let audioChunks = [];
let socket;

// ✉️ Yazılı mesaj gönderildiğinde
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = chatInput.value.trim();
  const customerId = customerIdInput.value.trim();
  if (!query || !customerId) return;

  appendMessage("🗣️ Siz", query);
  chatInput.value = "";

  try {
    const response = await fetch(`${backendHost}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query, customer_id: customerId }),
    });

    const data = await response.json();

    // Yazılı yanıtı göster
    const botText = data.response || data.text || "⚠️ Yanıt alınamadı.";
    appendMessage("🤖 Bot", botText);

    // Sesli yanıt varsa çal
    if (data.audio) {
      const audio = new Audio(`data:audio/wav;base64,${data.audio}`);
      audio.play();
    }
  } catch (err) {
    console.error("❌ Chat hatası:", err);
    appendMessage("🤖 Bot", "⚠️ Sistem hatası oluştu.");
  }
});

// 🎙️ Sesli mesaj gönderme
recordButton.addEventListener("click", async () => {
  const customerId = customerIdInput.value.trim();
  if (!customerId) {
    alert("Lütfen müşteri ID girin.");
    return;
  }

  socket = new WebSocket(`${backendHost.replace("https", "wss")}/ws?customer_id=${customerId}`);
  socket.binaryType = "arraybuffer";

  socket.onopen = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      mediaRecorder.start();

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
        socket.send(await audioBlob.arrayBuffer());
        audioChunks = [];
      };

      setTimeout(() => mediaRecorder.stop(), 4000);
    } catch (err) {
      console.error("🎙️ Mikrofon hatası:", err);
      appendMessage("🤖 Bot", "🎤 Mikrofon erişimi reddedildi.");
    }
  };

  socket.onmessage = (event) => {
    if (typeof event.data === "string") {
      try {
        const data = JSON.parse(event.data);
        const botText = data.response || data.text || "⚠️ Bot cevabı alınamadı.";
        appendMessage("🤖 Bot", botText);
      } catch (e) {
        console.warn("🧩 Geçersiz JSON mesajı:", event.data);
      }
    } else {
      const audioBlob = new Blob([event.data], { type: "audio/wav" });
      const audioUrl = URL.createObjectURL(audioBlob);
      audioPlayer.src = audioUrl;
      audioPlayer.play();
    }
  };

  socket.onerror = (error) => {
    console.error("WebSocket hatası:", error);
    appendMessage("🤖 Bot", "⚠️ Sesli bağlantı hatası.");
  };

  socket.onclose = () => {
    console.log("🔌 WebSocket bağlantısı kapandı.");
  };
});

// 💬 Mesaj kutusuna yeni mesaj ekle
function appendMessage(sender, message) {
  const messageElem = document.createElement("div");
  messageElem.className = "message";
  messageElem.innerHTML = `<strong>${sender}:</strong> ${message}`;
  chatBox.appendChild(messageElem);
  chatBox.scrollTop = chatBox.scrollHeight;
}
