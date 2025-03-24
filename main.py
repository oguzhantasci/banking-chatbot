import sys
import asyncio
import base64
from langchain_core.messages import HumanMessage
from graph import build_app
from tools import is_valid_customer, transcribe_audio, generate_speech_base64

async def run_chatbot(app, query: str, customer_id: str, config: dict) -> str:
    """
    LangGraph üzerinden AI yanıtı alır. Mesajlardan temiz içerik döner.
    """
    if not is_valid_customer(customer_id):
        return f"❌ Müşteri ID '{customer_id}' geçerli değil. Lütfen doğru ID giriniz."

    query = f"Müşteri ID: {customer_id}\n{query}"
    inputs = {"messages": [HumanMessage(content=query)]}
    result = ""

    try:
        async for chunk in app.astream(inputs, config, stream_mode="values"):
            response = chunk["messages"][-1].content
            # Bot prefix veya tekrar input gönderimini filtrele
            if response.startswith("Bot:") or response.startswith(f"Müşteri ID: {customer_id}") or response == query:
                continue
            result += response + "\n"

        return result.strip()

    except Exception as e:
        print(f"⚠️ Chatbot işlem hatası: {e}")
        return "⚠️ Bot cevabı alınamadı."

async def interactive_mode(app):
    """
    Terminalde etkileşimli AI bankacılık deneyimi (text + sesli yanıt).
    """
    print("\n💳 AI Bankacılık Asistanına Hoş Geldiniz!")
    print("Bakiyenizi öğrenin, işlem yapın veya son hareketlerinizi sorgulayın.")
    print("Çıkmak için 'exit' yazın.\n")

    customer_id = input("Lütfen Müşteri ID'nizi girin (örn. CUST0001): ").strip()
    while not is_valid_customer(customer_id):
        print(f"❌ Müşteri ID '{customer_id}' geçerli değil. Lütfen tekrar deneyin.\n")
        customer_id = input("Müşteri ID: ").strip()

    print(f"\n✅ '{customer_id}' ile giriş yapıldı.\n")

    config = {
        "configurable": {
            "thread_id": customer_id,
            "checkpoint_ns": "banking_session",
            "checkpoint_id": f"session_{customer_id}"
        }
    }

    while True:
        query = input("\n📝 Sormak istediğiniz şey: ").strip()
        if query.lower() == 'exit':
            print("👋 Görüşmek üzere!")
            break

        print("🔄 İşleniyor...")
        response = await run_chatbot(app, query, customer_id, config)
        print(f"\n🤖 AI Yanıtı:\n{response}")

        # ✅ Text-to-Speech
        audio_base64 = await generate_speech_base64(response)
        if audio_base64:
            audio_path = "response_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(audio_base64))
            print(f"🔊 Yanıt ses dosyası: {audio_path}")

            try:
                play_audio(audio_path)
            except Exception as e:
                print(f"⚠️ Ses çalınamadı: {e}")

async def main():
    app = build_app()
    if len(sys.argv) < 2:
        await interactive_mode(app)
    else:
        customer_id = input("Müşteri ID: ").strip()
        query = " ".join(sys.argv[1:])
        config = {
            "configurable": {
                "thread_id": customer_id,
                "checkpoint_ns": "banking_session",
                "checkpoint_id": f"session_{customer_id}"
            }
        }
        response = await run_chatbot(app, query, customer_id, config)
        print(response)

if __name__ == '__main__':
    asyncio.run(main())
