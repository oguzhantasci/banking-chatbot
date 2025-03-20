import sys
import asyncio
from langchain_core.messages import HumanMessage
from graph import build_app
from tools import is_valid_customer
from tools import transcribe_audio_whisper, record_audio, text_to_speech

async def run_chatbot(app, query: str, customer_id: str, config: dict) -> str:
    """Müşteri ID doğrulama ve JSON veri kullanımı ile chatbot işlemi."""

    if not is_valid_customer(customer_id):
        return f"❌ Müşteri ID '{customer_id}' geçerli değil. Lütfen doğru ID giriniz."

    query = f"Müşteri ID: {customer_id}\n{query}"
    inputs = {"messages": [HumanMessage(content=query)]}
    result = ""

    async for chunk in app.astream(inputs, config, stream_mode="values"):
        response = chunk["messages"][-1].content
        if response.startswith("Bot:") or response.startswith(f"Müşteri ID: {customer_id}") or response == query:
            continue
        result += response + "\n"

    text_to_speech(result.strip(), "response_audio.wav")  # Yanıtı sesli oku

    print(f"\n✅ AI Yanıtı:\n{result.strip()}")
    return result.strip()

async def interactive_mode(app):
    """Start an AI-powered banking assistant session."""
    print("\n💳 Welcome to the AI Banking Assistant!")
    print("You can ask about your balance, recent transactions, and perform fund transfers.")
    print("Type 'exit' to end the session.\n")

    customer_id = input("Please enter your Customer ID: ").strip()
    while not is_valid_customer(customer_id):
        customer_id = input("Please enter your Customer ID: ").strip()
        if not is_valid_customer(customer_id):
            print(f"❌ Müşteri ID '{customer_id}' geçerli değil. Lütfen doğru ID giriniz.\n")

    print(f"\n✅ Müşteri ID '{customer_id}' doğrulandı. Şimdi bankacılık işlemlerinizi yapabilirsiniz.\n")

    config = {"configurable": {"thread_id": customer_id, "checkpoint_ns": "banking_session",
                               "checkpoint_id": f"session_{customer_id}"}}

    while True:
        query = input("\nYour banking request: ").strip()
        if query.lower() == 'exit':
            print("Thank you for using our AI Banking Assistant. Goodbye!")
            break

        print("\nProcessing your request...")
        await run_chatbot(app, query, customer_id, config)
        print("\nResponse complete.")

async def main():
    """Initialize AI-powered banking assistant."""
    app = build_app()
    if len(sys.argv) < 2:
        await interactive_mode(app)
    else:
        customer_id = input("Please enter your Customer ID: ").strip()
        query = " ".join(sys.argv[1:])
        config = {"configurable": {"thread_id": customer_id, "checkpoint_ns": "banking_session", "checkpoint_id": f"session_{customer_id}"}}
        await run_chatbot(app, query, customer_id, config)

if __name__ == '__main__':
    asyncio.run(main())