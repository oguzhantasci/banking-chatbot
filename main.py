import sys
import asyncio
from langchain_core.messages import HumanMessage
from graph import build_app
from tools import is_valid_customer


async def run_chatbot(app, query: str, customer_id: str, config: dict) -> str:
    """Process banking queries dynamically with AI-driven responses."""
    # ✅ Step 1: Validate Customer ID before sprocessing
    if not is_valid_customer(customer_id):
        error_msg = f"❌ Müşteri ID '{customer_id}' geçerli değil. Lütfen doğru ID giriniz."
        print(error_msg)  # ✅ Show error in terminal for debugging
        return error_msg  # ✅ Return error message instead of proceeding

    query = f"Müşteri ID: {customer_id}\n{query}"
    inputs = {"messages": [HumanMessage(content=query)]}

    result = ""  # Store chatbot response
    async for chunk in app.astream(inputs, config, stream_mode="values"):
        response = chunk["messages"][-1].content
        result += response + "\n"  # Append to result

    final_response = result.strip()
    print("\n✅ Final Response:", final_response)  # ✅ Print final response to terminal

    return final_response  # ✅ Return response for API


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

    config = {"configurable": {"thread_id": "1", "customer_id": customer_id}}

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
        config = {"configurable": {"thread_id": "1", "customer_id": customer_id}}
        await run_chatbot(app, query, customer_id, config)


if __name__ == '__main__':
    asyncio.run(main())
