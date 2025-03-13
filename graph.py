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
from tools import (
    fetch_cards, fetch_credit_limits, fetch_current_debt,
    fetch_statement_debt, fetch_card_settings, fetch_accounts, fetch_account_balance
)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

MEMBERS = ["Credit_Card_Agent", "Account_Agent", "Professional_Response_Agent"]
OPTIONS = ("FINISH",) + tuple(MEMBERS)



# AI Agent'lar
CREDIT_CARD_TOOLS = [
    fetch_cards,
    fetch_credit_limits,
    fetch_current_debt,
    fetch_statement_debt,
    fetch_card_settings
]

ACCOUNT_TOOLS = [
    fetch_accounts,
    fetch_account_balance
]

# AI Model
LLM = ChatOpenAI(model="gpt-4o-mini")

# Supervisor Agent (Kullanıcı İsteklerini Yönlendirir)
SUPERVISOR_PROMPT = """
📌 **Rolün:** Kullanıcının isteğini analiz eden bir AI yöneticisisin.
🔹 **Görevin:**
- Kullanıcının sorgusunu inceleyerek uygun AI Agent'ı seçmek.
- Eğer işlem desteklenmiyorsa "Bu işlem desteklenmiyor." mesajını döndürmek.

🔹 **Desteklenen AI Agent'lar:**
1️⃣ **Credit_Card_Agent** → Kredi kartı bilgilerini ve ayarlarını getirir.
2️⃣ **Account_Agent** → Banka hesap bilgilerini ve bakiyeleri sorgular.

📌 **Yanıt formatı:**
- `Credit_Card_Agent`
- `Account_Agent`
- `FINISH`

❌ **Desteklenmeyen bir işlem talep edilirse**, aşağıdaki mesajı ver:  
*"Üzgünüz, yalnızca aşağıdaki işlemleri gerçekleştirebilirsiniz:"*  
- **Bakiye sorgulama**
- **Limit bilgisi sorgulama**
- **Anlık borç sorgulama**
- **Ekstre borcu sorgulama**
- **Hesap bilgileri sorgulama**
- **Kredi Kartı bilgileri sorgulama**
- **Kredi Kartı ayarlarını sorgulama**
*"Size yardımcı olabileceğim başka bir konu var mı?"*  
"""

CREDIT_CARD_PROMPT = """
📌 **Rolün:** Bir kredi kartı bilgi asistanısın.
🔹 **Görevin:** Kullanıcının kartları, limitleri, borçları ve kart ayarlarını sağlamak.
- Eğer müşteri ID geçerli değilse: "Müşteri bulunamadı."
"""

ACCOUNT_PROMPT = """
📌 **Rolün:** Bir hesap bilgi asistanısın.
🔹 **Görevin:** Kullanıcının banka hesaplarını ve bakiyelerini göstermek. Eksik bilgi varsa, kullanıcıdan iste.
- Eğer müşteri ID geçerli değilse: "Müşteri bulunamadı."
"""

PROFESSIONAL_RESPONSE_PROMT = """
📌 **Rolün:** Resmi, kurumsal ve bankacılığa uygun bir üslupla müşteri taleplerine net, saygılı ve profesyonel yanıtlar veren bir bankacılık asistanısın.
✅ **Yanıtlarını açık, net ve saygılı bir dille ver.**  
✅ **Sadece aşağıdaki bankacılık işlemleri hakkında yanıt ver:** 

🏦 **Desteklenen İşlemler:**  
- **Bakiye sorgulama**
- **Limit bilgisi sorgulama**
- **Anlık borç sorgulama**
- **Ekstre borcu sorgulama**
- **Hesap bilgileri sorgulama**
- **Kredi Kartı bilgileri sorgulama**
- **Kredi Kartı ayarlarını sorgulama**

❌ **Desteklenmeyen bir işlem talep edilirse**, aşağıdaki mesajı ver:  
*"Üzgünüz, yalnızca aşağıdaki işlemleri gerçekleştirebilirsiniz:"*  
- **Bakiye sorgulama**
- **Limit bilgisi sorgulama**
- **Anlık borç sorgulama**
- **Ekstre borcu sorgulama**
- **Hesap bilgileri sorgulama**
- **Kredi Kartı bilgileri sorgulama**
- **Kredi Kartı ayarlarını sorgulama**
*"Size yardımcı olabileceğim başka bir konu var mı?"*  
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

credit_card_agent = create_react_agent(
    LLM,
    tools=CREDIT_CARD_TOOLS,
    state_modifier=CREDIT_CARD_PROMPT
)

account_agent = create_react_agent(
    LLM,
    tools=ACCOUNT_TOOLS,
    state_modifier=ACCOUNT_PROMPT
)

professional_response_agent = create_react_agent(
    LLM,
    tools=[],
    state_modifier= PROFESSIONAL_RESPONSE_PROMT
)

workflow = StateGraph(AgentState)
workflow.add_node("Supervisor_Agent", supervisor_agent)
workflow.add_node("Credit_Card_Agent", functools.partial(agent_node, agent=credit_card_agent, name="Credit_Card_Agent"))
workflow.add_node("Account_Agent", functools.partial(agent_node, agent=account_agent, name="Account_Agent"))
workflow.add_node("Professional_Response_Agent", functools.partial(agent_node, agent=professional_response_agent, name="Professional_Response_Agent"))


workflow.add_conditional_edges("Supervisor_Agent", lambda x: x["next"], {
    "Credit_Card_Agent": "Credit_Card_Agent",
    "Account_Agent": "Account_Agent",
    "Professional_Response_Agent": "Professional_Response_Agent",
    "FINISH": "Professional_Response_Agent",
})
workflow.add_edge(START, "Supervisor_Agent")

def build_app():
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)