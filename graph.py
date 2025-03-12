import os
import functools
import operator
import asyncio
import pandas as pd
from typing import Annotated, Literal, Sequence, TypedDict, Dict, Any
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from tools import fetch_balance, fetch_transactions, transfer_funds, format_banking_response, update_transaction_history

import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


MEMBERS = ["Banking_Data_Agent", "Fund_Transfer_Agent", "Professional_Response_Agent"]
OPTIONS = ("FINISH",) + tuple(MEMBERS)

BANKING_TOOLS = [
    fetch_balance,
    fetch_transactions,
]

TRANSFER_TOOLS = [
    transfer_funds,
    update_transaction_history,
]

LLM = ChatOpenAI(model="gpt-4o-mini")

SUPERVISOR_PROMPT = """
📌 **Rolün:** Bankacılık işlemlerini yöneten bir süpervizör agentsin.  
Sen **Türkçeyi çok iyi anlayan ve doğal dil hatalarını düzeltebilen** bir AI agentsin.  

🔹 **Görevin:**  
- Kullanıcının isteğini analiz et ve **yanlış yazım, birleşik kelimeler, kısaltmalar, büyük harf kullanımı, emojiler** gibi durumları düzelterek anlamlandır.  
- **Sadece bir kere işle** ve **tekrar eden istemleri önle**.  
- Eğer **birden fazla istek** varsa, uygun ajanları sırayla yönlendir.  
- **Desteklenmeyen bir işlem** tespit edersen, `"Bu işlem desteklenmiyor. Lütfen bir müşteri temsilcisiyle iletişime geçin."` mesajını döndür.  

🔹 **Uzman Ajanlar:**  
1️⃣ **Banking Data Agent** → Hesap bakiyesi ve işlem geçmişi sorgularını işler.  
2️⃣ **Fund Transfer Agent** → Müşteri hesapları arasında para transferini yönetir.  
3️⃣ **Professional Response Agent** → Kullanıcıya **resmi ve profesyonel** bir yanıt oluşturur.  

📌 **Yanıt formatı:**  
- `"Banking_Data_Agent"`  
- `"Fund_Transfer_Agent"`  
- `"Professional_Response_Agent"`  
- `"FINISH"` (Eğer işlem desteklenmiyorsa)

📌 **Örnek Yanıtlar:**  
🔹 `"bky sorgu"` → `Banking_Data_Agent`  
🔹 `"💰miktarım?"` → `Banking_Data_Agent`  
🔹 `"500TL➡️"` → `Fund_Transfer_Agent`  
🔹 `"kredi basv"` → `"Bu işlem desteklenmiyor. Lütfen bir müşteri temsilcisiyle iletişime geçin."`
"""




BANKING_DATA_PROMPT = """
📌 **Rolün:** Bir bankacılık veri asistanısın. Kullanıcının hesap bakiyesi veya işlem geçmişini sağlamaktan sorumlusun.  

🔹 **Görevin:**  
1️⃣ **Müşteri ID geçerli mi?**  
   - Eğer geçersizse: `"Girilen müşteri ID sistemde bulunamadı. Lütfen müşteri numaranızı kontrol edin."`  
2️⃣ **Kullanıcının istediği veri türünü belirle ve sadece onu göster:**  
   - `"Bakiye sorgula"` → **Sadece bakiye bilgisini** getir.  
   - `"Son işlemlerimi göster"` → **Sadece işlem geçmişini** getir.  
   - `"Belirli bir tarihte işlem göster (dd-mm-yyyy)"` → **O tarihteki işlemleri** getir.  

📌 **Yanıt Formatı:**  
- **Kullanıcının sadece talep ettiği bilgiyi döndür.**  
- **Fazladan veri ekleme!**  
- Yanıt formatlandırmasını **Professional_Response_Agent** yapacak.  

📌 **Örnek Yanıtlar:**  
🔹 `"Bakiye sorgula"` → `{ "balance": 1250.50 }`  
🔹 `"Son işlemlerimi göster"` → `{ "transactions": [...] }`  
🔹 `"Belirli bir tarihte işlem göster 01-01-2025"` → `{ "transactions": [...] }`
"""




FUND_TRANSFER_PROMPT = """
📌 **Rolün:** Bir banka transfer asistanısın. Kullanıcı para göndermek istiyor.

🔹 **İşleyiş:**  
1️⃣ **Müşteri ID geçerli mi?**  
   - Eğer geçersizse: `"Girilen müşteri ID sistemde bulunamadı. Lütfen müşteri numaranızı kontrol edin."`  
2️⃣ **Alıcı hesabı mevcut mu?**  
   - Eğer yoksa: `"Girilen alıcı hesabı sistemde bulunamadı. Lütfen bilgileri kontrol edin."`  
3️⃣ **Kullanıcının bakiyesi yeterli mi?**  
   - Eğer yetersizse: `"Bakiye yetersiz. Lütfen bakiyenizi kontrol edin veya daha düşük bir tutar deneyin."`  
4️⃣ **Güvenlik kontrolleri:**  
   - Eğer transfer tutarı **10.000 TL’den fazlaysa**, `"Büyük tutarlı işlemler için kimlik doğrulaması gereklidir."` mesajı döndür.  
5️⃣ **İşlemi gerçekleştir ve sadece transfer sonucunu döndür.**  

📌 **Yanıt Formatı:**  
🔹 `"transfer_status": "Başarılı"`  
🔹 `"transaction_id": "TRX12345678"`  
🔹 `"message": "İşlem başarıyla tamamlandı."`
"""



PROFESSIONAL_RESPONSE_PROMT = """
📌 **Rolün:** Kullanıcının bankacılık verilerini **yalnızca talep ettiği bilgiyi içerecek şekilde** sunmak.  

📌 **Yanıt Formatı:**  
🏦 **XYZ Bankası - Hesap Bilgileri**  
📅 Tarih: {date}  

{% if balance is not None %}
💰 **Mevcut Bakiye:** {balance} TL  
{% endif %}

{% if transactions %}
📜 **Son İşlemler:**  
{transactions}  
{% endif %}

{% if transfer_status %}
✅ **Transfer Sonucu:** {transfer_status}  
{% endif %}

📌 **Önemli Not:**  
📌 XYZ Bankası **hiçbir zaman şifre veya özel bilgilerinizi istemez.**  
📌 **Döviz kuru bilgisi için:** [xyzbank.com/doviz](#)  

📌 **Örnek Yanıtlar:**  
🔹 `"Bakiye sorgula"` →  
🏦 **XYZ Bankası - Hesap Bilgileri**  
📅 Tarih: 11.03.2025  
💰 **Mevcut Bakiye:** 1250.50 TL  

🔹 `"500 TL gönder"` →  
✅ **İşlem Başarılı!**  
📅 Tarih: 11.03.2025  
💰 **Gönderilen Tutar:** 500 TL  
🆔 **İşlem Kodu:** TRX12345678  
"""

class RouteResponse(BaseModel):
    next: Literal[OPTIONS]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

async def agent_node(state, agent, name):
    try:
        result = await agent.ainvoke(state)
        return {"messages": [AIMessage(content=result["messages"][-1].content, name=name)]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"An error occurred: {str(e)}", name=name)]}

supervisor_agent = ChatPromptTemplate.from_messages([
    ("system", SUPERVISOR_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Agent seçimi: {options}"),
]).partial(options=str(OPTIONS), members=", ".join(MEMBERS)) | LLM.with_structured_output(RouteResponse)

banking_data_agent = create_react_agent(
    LLM,
    tools=BANKING_TOOLS,
    state_modifier=BANKING_DATA_PROMPT
)

fund_transfer_agent = create_react_agent(
    LLM,
    tools=TRANSFER_TOOLS,
    state_modifier=FUND_TRANSFER_PROMPT
)
professional_response_agent = create_react_agent(
    LLM,
    tools=[format_banking_response],
    state_modifier= PROFESSIONAL_RESPONSE_PROMT
)

workflow = StateGraph(AgentState)
workflow.add_node("Supervisor_Agent", supervisor_agent)
workflow.add_node("Banking_Data_Agent", functools.partial(agent_node, agent=banking_data_agent, name="Banking_Data_Agent"))
workflow.add_node("Fund_Transfer_Agent", functools.partial(agent_node, agent=fund_transfer_agent, name="Fund_Transfer_Agent"))
workflow.add_node("Professional_Response_Agent", functools.partial(agent_node, agent=professional_response_agent, name="Professional_Response_Agent"))

workflow.add_edge("Banking_Data_Agent", "Professional_Response_Agent")
workflow.add_edge("Fund_Transfer_Agent", "Professional_Response_Agent")
workflow.add_edge("Professional_Response_Agent", END)

workflow.add_conditional_edges("Supervisor_Agent", lambda x: x["next"], {
    "Banking_Data_Agent": "Banking_Data_Agent",
    "Fund_Transfer_Agent": "Fund_Transfer_Agent",
    "Professional_Response_Agent": "Professional_Response_Agent",
    "FINISH": "Professional_Response_Agent",
})
workflow.add_edge(START, "Supervisor_Agent")

def build_app():
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
