import pandas as pd
import datetime
from langchain_core.tools import tool
import pytz

# CSV Dosya Yolu
BANK_DATA_FILE = "custom_banking_data.csv"

def is_valid_customer(customer_id: str) -> bool:
    """
    Checks if the given customer ID exists in the dataset.
    """
    # ✅ Define dtype to prevent mixed-type warning
    dtype_mapping = {
        "Customer_ID": str,  # Ensure Customer_ID is always a string
        "Balance": float,  # Convert balances to numeric type
    }

    df = pd.read_csv(BANK_DATA_FILE, dtype=dtype_mapping, low_memory=False)

    return customer_id in df["Customer_ID"].values  # Check if ID exists

def load_bank_data():
    """CSV dosyasını yükler ve veri türlerini uygun şekilde ayarlar."""
    dtype_mapping = {
        "Customer_ID": str,
        "Card_Number": str,
        "Card_Type": str,
        "Credit_Limit": float,
        "Current_Debt": float,
        "Statement_Debt": float,
        "Statement_Due_Date": str,
        "Online_Shopping_Enabled": str,
        "QRCode_Payment_Enabled": str,
        "Statement_Preference": str,
        "Account_Number": str,
        "Account_Type": str,
        "Balance": float,
        "Name": str,
        "Surname": str,
        "Gender": str
    }
    return pd.read_csv(BANK_DATA_FILE, dtype=dtype_mapping)

@tool
def fetch_cards(customer_id: str) -> str:
    """Müşterinin kart numaralarını ve tiplerini getirir."""
    df = load_bank_data()
    customer_data = df[df['Customer_ID'] == customer_id]
    if customer_data.empty:
        return "Müşteri bulunamadı."
    return customer_data[['Card_Number', 'Card_Type']].to_string(index=False)

@tool
def fetch_credit_limits(customer_id: str) -> str:
    """Müşterinin kredi limitlerini getirir."""
    df = load_bank_data()
    customer_data = df[df['Customer_ID'] == customer_id]
    if customer_data.empty:
        return "Müşteri bulunamadı."
    return customer_data[['Card_Number', 'Credit_Limit']].to_string(index=False)

@tool
def fetch_current_debt(customer_id: str, card_number: str) -> str:
    """Belirli bir kartın mevcut borcunu getirir."""
    df = load_bank_data()
    customer_data = df[(df['Customer_ID'] == customer_id) & (df['Card_Number'] == card_number)]
    if customer_data.empty:
        return "Kart veya müşteri bulunamadı."
    return f"Kart Borcu: {customer_data.iloc[0]['Current_Debt']} TL"

@tool
def fetch_statement_debt(customer_id: str, card_number: str) -> str:
    """Belirli bir kartın ekstre borcunu ve son ödeme tarihini getirir."""
    df = load_bank_data()
    customer_data = df[(df['Customer_ID'] == customer_id) & (df['Card_Number'] == card_number)]
    if customer_data.empty:
        return "Kart veya müşteri bulunamadı."
    return f"Ekstre Borcu: {customer_data.iloc[0]['Statement_Debt']} TL, Son Ödeme Tarihi: {customer_data.iloc[0]['Statement_Due_Date']}"

@tool
def fetch_card_settings(customer_id: str, card_number: str) -> str:
    """Kartın ayarlarını getirir (İnternet alışverişi, QR Kod ödeme vb.)."""
    df = load_bank_data()
    customer_data = df[(df['Customer_ID'] == customer_id) & (df['Card_Number'] == card_number)]
    if customer_data.empty:
        return "Kart veya müşteri bulunamadı."
    return f"İnternet Alışverişi: {customer_data.iloc[0]['Online_Shopping_Enabled']}, QR Kod Ödeme: {customer_data.iloc[0]['QRCode_Payment_Enabled']}, Ekstre Tercihi: {customer_data.iloc[0]['Statement_Preference']}"

@tool
def fetch_accounts(customer_id: str) -> str:
    """Müşterinin hesaplarını getirir."""
    df = load_bank_data()
    customer_data = df[df['Customer_ID'] == customer_id]
    if customer_data.empty:
        return "Müşteri bulunamadı."
    return customer_data[['Account_Number', 'Account_Type']].to_string(index=False)

@tool
def fetch_account_balance(account_number: str) -> str:
    """Belirli bir banka hesabının bakiyesini getirir."""
    df = load_bank_data()
    account_data = df[df['Account_Number'] == account_number]
    if account_data.empty:
        return "Hesap bulunamadı."
    return f"Mevcut Bakiye: {account_data.iloc[0]['Balance']} TL"

@tool
def fetch_customer_info(customer_id: str) -> str:
    """Müşterinin adı, soyadı ve cinsiyetini getirir."""
    df = load_bank_data()
    customer = df[df["Customer_ID"] == customer_id]
    if customer.empty:
        return None

    return {
        "customer_id": customer.iloc[0]["Customer_ID"],
        "name": customer.iloc[0]["Name"],
        "surname": customer.iloc[0]["Surname"],
        "gender": customer.iloc[0]["Gender"]
    }
