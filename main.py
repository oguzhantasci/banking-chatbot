import sys
import asyncio
from langchain_core.messages import HumanMessage
from graph import build_app


async def run_chatbot(app, query: str, customer_id: str, config: dict):
    """Process banking queries dynamically with AI-driven responses."""
    query = f"Müşteri ID: {customer_id}\n{query}"
    inputs = {"messages": [HumanMessage(content=query)]}
    async for chunk in app.astream(inputs, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()


async def interactive_mode(app):
    """Start an AI-powered banking assistant session."""
    print("\n💳 Welcome to the AI Banking Assistant!")
    print("You can ask about your balance, recent transactions, and perform fund transfers.")
    print("Type 'exit' to end the session.\n")

    customer_id = input("Please enter your Customer ID: ").strip()
    if not customer_id:
        print("Customer ID is required to proceed. Restart the chatbot and enter a valid ID.")
        return

    config = {"configurable": {"thread_id": "1", "customer_id": customer_id}}

    while True:
        query = input("\nYour banking request: ").strip()
        if query.lower() == 'exit':
            print("Thank you for using our AI Banking Assistant. Goodbye!")
            break

        print("\nProcessing your request...")
        await run(app, query, customer_id, config)
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
        await run(app, query, customer_id, config)


if __name__ == '__main__':
    asyncio.run(main())
