from langchain_core.tools import tool
import json
from typing import List
from openai import AsyncOpenAI
import base64
import openai

client = AsyncOpenAI()

CUSTOMER_DATA_FILE = "custom_banking_data.json"

def load_customer_data():
    """Müşteri bilgilerini JSON dosyasından yükler."""
    with open(CUSTOMER_DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data  # Veriyi direkt olarak döndürüyoruz çünkü anahtarlar müşteri ID'leri

def is_valid_customer(customer_id: str) -> bool:
    """Müşteri ID'nin geçerli olup olmadığını kontrol eder."""
    data = load_customer_data()
    return customer_id in data  # Artık direkt ID'yi kontrol edebiliriz


@tool
def fetch_cards(customer_id: str) -> List[str]:
    """Fetches the customer's unique credit and debit card numbers from transaction history."""
    data = load_customer_data()

    # Find transactions belonging to the customer
    customer_data = next((entry for entry in data if entry["customer_id"] == customer_id), None)

    if not customer_data:
        return "Müşteri bulunamadı veya işlem kaydı yok."

    # Extract unique card numbers
    unique_cards = list(set(transaction["card_number"] for transaction in customer_data["transactions"]))

    return unique_cards

@tool
def fetch_credit_limits(customer_id: str) -> dict:
    """Fetches total and available credit limits for a customer."""
    data = load_customer_data()
    customer = data.get(customer_id)
    if not customer:
        return "Müşteri bulunamadı."

    total_limit = sum(card.get("credit_limit", 0) for card in customer.get("cards", []))
    available_limit = sum(card.get("available_limit", 0) for card in customer.get("cards", []))

    return {"total_limit": total_limit, "available_limit": available_limit}

@tool
def fetch_current_debt(customer_id: str) -> dict:
    """Fetches total outstanding credit card debt for a customer."""
    transactions = load_customer_data()
    customer_transactions = [t for t in transactions if t["customer_id"] == customer_id]

    if not customer_transactions:
        return "Müşteri bulunamadı veya işlem kaydı yok."

    total_debt = sum(t["amount"] for t in customer_transactions)
    return {"total_debt": total_debt}

@tool
def fetch_statement_debt(customer_id: str) -> str:
    """Fetches the statement debt and due date for a customer's credit cards."""
    data = load_customer_data()
    customer = data.get(customer_id)
    if not customer:
        return "Müşteri bulunamadı."

    statement_info = []
    for card in customer.get("cards", []):
        statement_info.append({
            "card_number": card["card_number"],
            "statement_debt": card.get("statement_debt", "N/A"),
            "due_date": card.get("statement_due_date", "N/A"),
        })

    return statement_info

@tool
def fetch_card_settings(customer_id: str, card_number: str) -> dict:
    """Fetches a card's settings (e.g., online shopping, QR payment)."""
    data = load_customer_data()
    customer = data.get(customer_id)
    if not customer:
        return "Müşteri bulunamadı."

    for card in customer.get("cards", []):
        if str(card["card_number"]) == str(card_number):
            return {
                "online_shopping": card.get("online_shopping_enabled", "Unknown"),
                "qr_payment": card.get("qr_payment_enabled", "Unknown"),
                "statement_preference": card.get("statement_preference", "Unknown"),
            }

    return "Kart bulunamadı."

@tool
def fetch_accounts(customer_id: str) -> list:
    """Fetches all bank accounts associated with a customer."""
    data = load_customer_data()
    customer = data.get(customer_id)
    if not customer:
        return "Müşteri bulunamadı."

    return customer.get("accounts", [])

@tool
def fetch_account_balance(customer_id: str, account_number: str) -> str:
    """Fetches the balance of a specific bank account."""
    data = load_customer_data()
    customer = data.get(customer_id)
    if not customer:
        return "Müşteri bulunamadı."

    for account in customer.get("accounts", []):
        if str(account["Account_Number"]) == str(account_number):
            return f"Mevcut Bakiye: {account['Balance']} TL"

    return "Hesap bulunamadı."


@tool
def fetch_customer_info(customer_id: str) -> dict:
    """Fetches customer information including name, surname, and gender."""
    data = load_customer_data()
    return data.get(customer_id, {})

# ✅ Türkçe Sesli Girdiyi Metne Çevir
async def transcribe_audio(audio_file_path: str) -> str:
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="tr",
                response_format="text"
            )
        return transcript.strip()
    except Exception as e:
        print(f"⚠️ Ses tanıma hatası: {e}")
        return "⚠️ Ses çözümlenemedi."


# ✅ Text-to-Speech - WAV Ses Dosyası Oluşturur
async def generate_speech_base64(text: str) -> str:
    try:
        response = await client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="nova",
            input=text
        )
        audio_bytes = await response.aread()
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print(f"🔊 Ses üretim hatası: {e}")
        return ""
