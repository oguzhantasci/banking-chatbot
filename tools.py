import pandas as pd
import datetime, timedelta
from langchain_core.tools import tool
import pytz

# CSV Dosya Yolu
BANK_DATA_FILE = "custom_banking_data.csv"
TRANSACTION_DATA_FILE = "credit_card_transactions.csv"

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

def load_transaction_data():
    """Kredi kartı işlemlerini içeren CSV dosyasını yükler."""
    dtype_mapping = {
        "Transaction_ID": str,
        "Customer_ID": str,
        "Card_Number": str,
        "Transaction_Date": str,
        "Transaction_Type": str,
        "Amount": float,
        "Currency": str,
        "Merchant": str,
        "Category": str,
        "Installment_Count": int,
        "Remaining_Installments": int
    }
    return pd.read_csv(TRANSACTION_DATA_FILE, dtype=dtype_mapping)


@tool
def fetch_cards(customer_id: str) -> str:
    """Müşterinin kart numaralarını ve tiplerini getirir."""
    df = load_bank_data()
    customer_data = df[df['Customer_ID'] == customer_id]
    if customer_data.empty:
        return "Müşteri bulunamadı."
    return customer_data[['Card_Number', 'Card_Type']].to_string(index=False)


@tool
def fetch_card_transactions(customer_id: str) -> str:
    """Müşterinin tüm kredi kartı işlemlerini getirir."""
    df = load_transaction_data()
    customer_data = df[df['Customer_ID'] == customer_id]
    if customer_data.empty:
        return "Müşteri bulunamadı veya işlem kaydı yok."

    return customer_data[
        ['Transaction_Date', 'Transaction_Type', 'Amount', 'Currency', 'Merchant', 'Category']].to_string(index=False)


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


@tool
def fetch_transactions_by_category(customer_id: str, category: str) -> str:
    """Müşterinin belirli bir kategoride yaptığı harcamaları getirir."""
    df = load_transaction_data()
    category_lower = category.lower().strip()
    df['Category'] = df['Category'].str.lower().str.strip()

    customer_data = df[(df['Customer_ID'] == customer_id) & (df['Category'] == category_lower)]
    if customer_data.empty:
        return f"Müşteri '{category}' kategorisinde işlem yapmamış."

    return customer_data[['Transaction_Date', 'Amount', 'Currency', 'Merchant']].to_string(index=False)


@tool
def fetch_installment_transactions(customer_id: str) -> str:
    """Müşterinin taksitli işlemlerini ve kalan taksit sayılarını getirir."""
    df = load_transaction_data()
    customer_data = df[(df['Customer_ID'] == customer_id) & (df['Installment_Count'] > 1)]

    if customer_data.empty:
        return "Müşteri bulunamadı veya taksitli işlem kaydı yok."

    return customer_data[['Transaction_Date', 'Merchant', 'Amount', 'Installment_Count', 'Remaining_Installments']].to_string(index=False)

@tool
def fetch_total_spent(customer_id: str, months: int = 3) -> str:
    """Müşterinin son X ayda toplam yaptığı harcamayı getirir."""
    df = load_transaction_data()
    start_date = datetime.now() - timedelta(days=30 * months)

    customer_data = df[(df['Customer_ID'] == customer_id) & (df['Transaction_Date'] >= start_date)]
    if customer_data.empty:
        return f"Son {months} ayda herhangi bir harcama bulunmamaktadır."

    total_spent = customer_data["Amount"].sum()
    return f"Sayın müşterimiz, son {months} ayda toplam {total_spent:.2f} TL harcama yaptınız."

@tool
def fetch_card_transactions(customer_id: str) -> str:
    """Müşterinin tüm kredi kartı işlemlerini getirir."""
    df = load_transaction_data()
    customer_data = df[df['Customer_ID'] == customer_id]
    if customer_data.empty:
        return "Müşteri bulunamadı veya işlem kaydı yok."
    return customer_data[['Transaction_Date', 'Transaction_Type', 'Amount', 'Currency', 'Merchant', 'Category']].to_string(index=False)


@tool
def fetch_recent_transactions(customer_id: str, months: int = None, year: int = None) -> str:
    """
    Müşterinin belirli bir zaman aralığında (son X ay veya belirli bir yıl) yaptığı kredi kartı işlemlerini getirir.
    - `months` değeri girilirse son X ayın harcamaları listelenir.
    - `year` değeri girilirse sadece o yıl içindeki harcamalar listelenir.
    """
    df = load_transaction_data()
    customer_data = df[df['Customer_ID'] == customer_id]

    if customer_data.empty:
        return "Müşteri bulunamadı veya işlem kaydı yok."

    # Tarih formatını datetime'a çevir
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])

    # Ay bazlı filtreleme
    if months is not None:
        start_date = datetime.now() - timedelta(days=30 * months)
        filtered_data = customer_data[customer_data['Transaction_Date'] >= start_date]

    # Yıl bazlı filtreleme
    elif year is not None:
        filtered_data = customer_data[customer_data['Transaction_Date'].dt.year == year]

    else:
        return "Lütfen belirli bir yıl veya ay aralığı giriniz."

    if filtered_data.empty:
        return f"Belirtilen zaman diliminde ({months} ay veya {year}) işlem kaydı bulunmamaktadır."

    return filtered_data[
        ['Transaction_Date', 'Transaction_Type', 'Amount', 'Currency', 'Merchant', 'Category']].to_string(index=False)


@tool
def fetch_top_expenses(customer_id: str, top_n: int = 5) -> str:
    """Müşterinin en yüksek harcamalarını getirir."""
    df = load_transaction_data()
    customer_data = df[df['Customer_ID'] == customer_id]

    if customer_data.empty:
        return "Müşteri bulunamadı veya işlem kaydı yok."

    top_expenses = customer_data.nlargest(top_n, 'Amount')

    return top_expenses[['Transaction_Date', 'Transaction_Type', 'Amount', 'Currency', 'Merchant', 'Category']].to_string(index=False)


@tool
def fetch_transactions_by_card(customer_id: str, card_number: str) -> str:
    """Belirli bir kart numarası ile yapılan harcamaları getirir."""
    df = load_transaction_data()

    # Kart numarası string olarak tutulduğu için formatı düzeltiyoruz
    card_number = str(card_number).strip()

    # Kullanıcının tüm harcamalarını al ve sadece ilgili kart numarasını filtrele
    customer_data = df[(df['Customer_ID'] == customer_id) & (df['Card_Number'] == card_number)]

    if customer_data.empty:
        return f"Sayın müşterimiz, {card_number} kartınızla yapılmış herhangi bir işlem bulunmamaktadır."

    return customer_data[
        ['Transaction_Date', 'Transaction_Type', 'Amount', 'Currency', 'Merchant', 'Category']].to_string(index=False)


@tool
def fetch_transactions_by_type(customer_id: str, transaction_type: str) -> str:
    """Müşterinin belirli bir işlem türüne göre harcamalarını getirir."""
    df = load_transaction_data()

    # İşlem türünü küçük harfe çevirerek filtreleme yap
    transaction_type_lower = transaction_type.lower().strip()
    df['Transaction_Type'] = df['Transaction_Type'].str.lower().str.strip()

    # Kullanıcının işlemlerini filtrele
    customer_data = df[(df['Customer_ID'] == customer_id) & (df['Transaction_Type'] == transaction_type_lower)]

    if customer_data.empty:
        return f"Sayın müşterimiz, '{transaction_type}' türünde herhangi bir işleminiz bulunmamaktadır."

    return customer_data[['Transaction_Date', 'Amount', 'Currency', 'Merchant', 'Category']].to_string(index=False)


@tool
def fetch_transaction_by_id(customer_id: str, transaction_id: str) -> str:
    """Belirli bir işlem numarası ile harcama detaylarını getirir."""
    df = load_transaction_data()

    # İşlem numarasını temizleyip string olarak al
    transaction_id = str(transaction_id).strip()

    # Kullanıcının işlem numarasını filtrele
    customer_data = df[(df['Customer_ID'] == customer_id) & (df['Transaction_ID'] == transaction_id)]

    if customer_data.empty:
        return f"Sayın müşterimiz, işlem numarası '{transaction_id}' ile eşleşen bir harcama kaydı bulunmamaktadır."

    return customer_data[
        ['Transaction_Date', 'Transaction_Type', 'Amount', 'Currency', 'Merchant', 'Category']].to_string(index=False)


@tool
def fetch_transactions_by_merchant(customer_id: str, merchant: str) -> str:
    """Müşterinin belirli bir satıcıdan (Merchant) yaptığı harcamaları getirir."""
    df = load_transaction_data()

    # Merchant adını küçük harfe çevirerek eşleşmeyi sağla
    merchant_lower = merchant.lower().strip()
    df['Merchant'] = df['Merchant'].str.lower().str.strip()

    # Kullanıcının belirttiği merchant ile eşleşen verileri filtrele
    customer_data = df[(df['Customer_ID'] == customer_id) & (df['Merchant'] == merchant_lower)]

    if customer_data.empty:
        return f"Sayın müşterimiz, {merchant} satıcısından herhangi bir harcama kaydınız bulunmamaktadır."

    # İşlem detaylarını döndür
    return customer_data[['Transaction_Date', 'Transaction_Type', 'Amount', 'Currency', 'Category']].to_string(
        index=False)
