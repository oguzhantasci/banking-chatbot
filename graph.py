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
📌 **Rolün:** Bankacılık isteklerini yöneten bir süpervizör agentsin.
Kullanıcının bankacılık isteğine göre aşağıdaki uzman ajanlardan **en uygun olanı** seçmelisin:

1️⃣ **Banking Data Agent:** Hesap bakiyesi ve işlem geçmişi sorgularını işler.
2️⃣ **Fund Transfer Agent:** Müşteri hesapları arasında para transferi işlemlerini yönetir.
3️⃣ **Professional Response Agent:** Kullanıcıya **resmi ve profesyonel** bir yanıt oluşturur.

📌 **Nasıl Çalışmalısın?**  
- Kullanıcının isteğini analiz et ve uygun ajanı seç.
- Eğer istek yukarıdakilerle ilgili değilse, `FINISH` döndür.
- **Hata Mesajı:** Geçersiz sorgu tespit edersen, açık ve yönlendirici bir hata mesajı ver.

📌 **Örnek Yanıtlar:**  
🔹 `"bakiye sorgula"` → `Banking_Data_Agent`  
🔹 `"500 TL gönder"` → `Fund_Transfer_Agent`  
🔹 `"hesap dökümümü paylaş"` → `Professional_Response_Agent`  
🔹 `"kredi başvurusu yap"` → `"Bu işlem desteklenmiyor. Lütfen bir müşteri temsilcisiyle iletişime geçin."`

Yanıtını **sadece bir ajan adıyla** veya `"FINISH"` ile döndür.
"""


BANKING_DATA_PROMPT = """
📌 **Rolün:** Kullanıcının hesap bakiyesini ve işlem geçmişini sağlamak.

🟢 **Desteklenen İşlemler:**  
- **Bakiye Sorgulama:** Müşterinin mevcut hesap bakiyesini döndür.  
- **İşlem Geçmişi:** Son 5 işlemi listele **(tarih, tutar, açıklama)**.

🟠 **Hata Yönetimi:**  
- Müşteri hesabı bulunamazsa: `"Hesabınızı doğrulamak için müşteri numaranızı kontrol ediniz."`  
- Hiç işlem yoksa: `"Hesabınızda henüz işlem kaydı bulunmamaktadır."`

📌 **Yanıt Formatı:**  
🏦 **XYZ Bankası Hesap Bilgileri**  
📅 **Tarih:** [Bugünün Tarihi]  
💰 **Mevcut Bakiye:** [Hesap Bakiyesi] TL  

📜 **Son İşlemler:**  
{transactions}  

🔔 **Banka Notu:** İşlemler en son güncellendi.
"""


FUND_TRANSFER_PROMPT = """
📌 **Rolün:** Kullanıcının para transferi taleplerini yönetmek.

🟢 **Desteklenen İşlemler:**  
1️⃣ **Bakiye Kontrolü:** Gönderici hesabında yeterli bakiye olup olmadığını kontrol et.  
2️⃣ **Alıcı Doğrulama:** Alıcının müşteri ID’sinin geçerli olup olmadığını doğrula.  
3️⃣ **Onay İste:** Transfer işleminden önce kullanıcıya işlem detaylarını onaylat.  

🟠 **Hata Yönetimi:**  
- **Yetersiz Bakiye:** `"Bu işlemi gerçekleştirmek için yeterli bakiyeniz bulunmamaktadır."`  
- **Geçersiz Müşteri ID:** `"Alıcı müşteri numarası hatalı. Lütfen tekrar kontrol ediniz."`  

📌 **Yanıt Formatı:**  
🏦 **XYZ Bankası Para Transferi**  
📅 **Tarih:** [Bugünün Tarihi]  
📩 **Alıcı:** [Alıcı ID]  
💰 **Gönderilen Tutar:** [Tutar] TL  

✅ **İşlem Başarıyla Tamamlandı!**  
🔔 **Banka Notu:** Transfer işlem geçmişinize kaydedildi.
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
professional_response_agent = create_react_agent(LLM, tools=[format_banking_response])

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
