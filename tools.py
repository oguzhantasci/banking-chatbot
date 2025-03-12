import pandas as pd
import datetime

BANK_DATA_FILE = "Bank_Transaction.csv"


def is_valid_customer(customer_id: str) -> bool:
    """
    Checks if the given customer ID exists in the dataset.
    """
    # ✅ Define dtype to prevent mixed-type warning
    dtype_mapping = {
        "Customer_ID": str,  # Ensure Customer_ID is always a string
        "Account_Balance": float,  # Convert balances to numeric type
        "Transaction_Amount": float  # Ensure transactions are numeric
    }

    df = pd.read_csv(BANK_DATA_FILE, dtype=dtype_mapping, low_memory=False)

    return customer_id in df["Customer_ID"].values  # Check if ID exists

def load_bank_data():
    """Load banking data from CSV file."""
    return pd.read_csv(BANK_DATA_FILE)


def save_bank_data(df):
    """Save banking data back to CSV file."""
    df.to_csv(BANK_DATA_FILE, index=False)


import pandas as pd
import datetime
from langchain_core.tools import tool

@tool
def fetch_balance(customer_id: str) -> str:
    """Retrieve the balance of a customer using the correct column name."""
    df = pd.read_csv("Bank_Transaction.csv")

    # Ensure the Customer_ID column is a string for comparison
    df['Customer_ID'] = df['Customer_ID'].astype(str)

    customer_data = df[df['Customer_ID'] == str(customer_id)]
    if customer_data.empty:
        return f"Müşteri numarası {customer_id} bulunamadı. Lütfen tekrar kontrol edin."

    balance = customer_data.iloc[-1]['Account_Balance']  # Use correct column name
    return f"Mevcut bakiye: {balance} TL"

@tool
def fetch_transactions(customer_id: str, limit: int = 5) -> str:
    """Retrieve last N transactions of a customer."""
    df = pd.read_csv("Bank_Transaction.csv")

    # Ensure Customer_ID is a string for matching
    df['Customer_ID'] = df['Customer_ID'].astype(str)

    transactions = df[df['Customer_ID'] == str(customer_id)].tail(limit)
    if transactions.empty:
        return f"Müşteri numarası {customer_id} için işlem bulunamadı."

    return transactions[['Transaction_Date', 'Transaction_Amount', 'Transaction_Description']].to_string(index=False)

@tool
def transfer_funds(sender_id: str, recipient_id: str, amount: float) -> str:
    """
    Transfers funds between accounts if both customer IDs are valid.

    Args:
        sender_id (str): The sender's customer ID.
        recipient_id (str): The recipient's customer ID.
        amount (float): The amount to transfer.

    Returns:
        str: Transfer success or failure message.
    """
    df = pd.read_csv("Bank_Transaction.csv", dtype={"Customer_ID": str})  # Ensure correct data type

    if sender_id not in df["Customer_ID"].astype(str).values:
        return "Gönderici müşteri ID'si bulunamadı. Lütfen geçerli bir hesapla tekrar deneyin."

    if recipient_id not in df["Customer_ID"].astype(str).values:
        return "Alıcı müşteri ID'si bulunamadı. Lütfen geçerli bir müşteri ID'si girin."

    return f"{amount} TL başarıyla {recipient_id} hesabına gönderildi."

@tool
def update_transaction_history(customer_id: str, transaction_type: str, amount: float):
    """Log a new transaction in the dataset."""
    df = pd.read_csv("Bank_Transaction.csv", dtype=str)
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_transaction = pd.DataFrame(
        [{'CustomerID': customer_id, 'Date': date, 'TransactionType': transaction_type, 'Amount': amount}])
    df = pd.concat([df, new_transaction], ignore_index=True)
    df.to_csv("Bank_Transaction.csv", index=False)

@tool
def format_banking_response(balance: str, transactions: str) -> str:
    """
    Formats banking information into a professional banking statement.

    Args:
        balance (str): The current balance information.
        transactions (str): The recent transaction history.

    Returns:
        str: A formatted response containing banking details.
    """
    return f"""
🏦 **XYZ Bankası Hesap Bilgileri**  
📅 Tarih: {datetime.datetime.now().strftime('%d %B %Y')}  
{balance}  
📜 Son İşlemler:  
{transactions}  
🔔 **Banka Notu:** İşlemler en son güncellendi.
    """




