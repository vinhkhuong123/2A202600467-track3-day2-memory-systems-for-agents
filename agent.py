import json
import tiktoken
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from memory_backends import ShortTermMemory, LongTermMemory, EpisodicMemory, SemanticMemory

class MemoryState(TypedDict):
    query: str
    intent: str
    active_backends: List[str]
    raw_memory: Dict[str, Any]
    merged_context: str
    response: str
    messages: list
    memory_budget: int

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize memories
short_term = ShortTermMemory()
long_term = LongTermMemory()
episodic = EpisodicMemory()
semantic = SemanticMemory()

def memory_router_node(state: MemoryState):
    """Memory Router: LLM-based intent classification & route to appropriate memory"""
    prompt = f"""Phân loại intent của câu hỏi sau để xác định cần gọi những bộ nhớ nào: '{state['query']}'. 
Trả lời dưới dạng JSON với 2 keys:
- 'intent': một trong [greeting, fact_recall, question_answering, general_chat, fact_update].
- 'active_backends': list các backend cần gọi từ [short_term, long_term, episodic, semantic].

Ví dụ:
{{"intent": "greeting", "active_backends": ["short_term", "long_term"]}}
{{"intent": "fact_recall", "active_backends": ["short_term", "long_term", "episodic"]}}
{{"intent": "question_answering", "active_backends": ["short_term", "semantic"]}}

Trả về đúng định dạng JSON:"""
    
    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        content = res.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        state['intent'] = data.get('intent', 'general_chat')
        state['active_backends'] = data.get('active_backends', ['short_term'])
    except Exception as e:
        print("Lỗi parse intent:", e)
        state['intent'] = 'general_chat'
        state['active_backends'] = ['short_term', 'long_term', 'episodic', 'semantic']
        
    return state

def count_tokens(text: str) -> int:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        return len(text.split())

def memory_backends_node(state: MemoryState):
    """Memory Backends: Retrieve from routed backends."""
    raw_memory = {
        "short_term": [],
        "long_term": {},
        "episodic": [],
        "semantic": []
    }
    
    backends = state.get('active_backends', [])
    
    if "short_term" in backends:
        raw_memory["short_term"] = short_term.get_context()
        
    if "long_term" in backends:
        raw_memory["long_term"] = long_term.get_profile()
        
    if "episodic" in backends:
        raw_memory["episodic"] = episodic.get_episodes(limit=3)
        
    if "semantic" in backends:
        raw_memory["semantic"] = semantic.search(state['query'], top_k=2)
        
    state['raw_memory'] = raw_memory
    return state

def context_manager_node(state: MemoryState):
    """Context Manager: Priority-based trim, 20% budget."""
    raw = state.get('raw_memory', {})
    budget = state.get('memory_budget', 400)
    current_tokens = 0
    
    final_profile = {}
    final_sem_hits = []
    final_eps = []
    
    # Priority 1: Profile (Long-term)
    if raw.get("long_term"):
        profile_str = json.dumps(raw["long_term"], ensure_ascii=False)
        profile_tokens = count_tokens(profile_str)
        if current_tokens + profile_tokens <= budget:
            final_profile = raw["long_term"]
            current_tokens += profile_tokens
            
    # Priority 2: Semantic Hits
    if raw.get("semantic"):
        for hit in raw["semantic"]:
            hit_tokens = count_tokens(hit)
            if current_tokens + hit_tokens <= budget:
                final_sem_hits.append(hit)
                current_tokens += hit_tokens
                
    # Priority 3: Episodes
    if raw.get("episodic"):
        for ep in reversed(raw["episodic"]): # most recent first
            ep_str = json.dumps(ep, ensure_ascii=False)
            ep_tokens = count_tokens(ep_str)
            if current_tokens + ep_tokens <= budget:
                final_eps.insert(0, ep)
                current_tokens += ep_tokens

    # Merge contexts
    context_str = ""
    if final_profile:
        context_str += f"\nUser Profile: {json.dumps(final_profile, ensure_ascii=False)}"
    if final_sem_hits:
        context_str += f"\nRelevant Knowledge: {final_sem_hits}"
    if final_eps:
        context_str += f"\nPast Episodes: {json.dumps(final_eps, ensure_ascii=False)}"
        
    state['merged_context'] = context_str
    return state

def generate_node(state: MemoryState):
    """LLM Generate: Inject into prompt."""
    sys_prompt = "Bạn là trợ lý AI thông minh có khả năng ghi nhớ."
    
    if state.get('merged_context'):
        sys_prompt += f"\n\nContext:\n{state['merged_context']}"
        
    messages = [SystemMessage(content=sys_prompt)]
    
    # Add short term history
    raw = state.get('raw_memory', {})
    short_term_hist = raw.get('short_term', short_term.get_context())
    
    for msg in short_term_hist:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
            
    # Add current query
    messages.append(HumanMessage(content=state['query']))
    
    response = llm.invoke(messages)
    state['response'] = response.content
    
    # Update short term buffer
    short_term.add_message("user", state['query'])
    short_term.add_message("assistant", response.content)
    
    return state

def save_memory_node(state: MemoryState):
    """Save Memory: LLM extract facts + conflict handling."""
    profile = state.get('raw_memory', {}).get('long_term', long_term.get_profile())
    prompt = f"""Dựa vào cuộc hội thoại sau:
User: {state['query']}
Assistant: {state['response']}

Hãy trích xuất các thông tin cá nhân của User (ví dụ: tên, sở thích, dị ứng, nghề nghiệp...) dưới định dạng JSON.
Nếu có cập nhật (ví dụ User nói "À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò"), hãy cập nhật thông tin mới nhất và ghi đè thông tin cũ.
Profile hiện tại của hệ thống: {json.dumps(profile, ensure_ascii=False)}

Chỉ trả về JSON format: {{"key": "value"}}. Trả về {{}} nếu không có thông tin cá nhân nào được nhắc tới hoặc cập nhật.
Tuyệt đối không dùng code block markdown."""
    
    try:
        extraction = llm.invoke([HumanMessage(content=prompt)])
        content = extraction.content.replace("```json", "").replace("```", "").strip()
        new_facts = json.loads(content)
        
        for k, v in new_facts.items():
            long_term.update_fact(k, str(v))
            
    except Exception as e:
        pass # Silently pass parsing errors
        
    # Save episodic
    episodic.save_episode({
        "query": state['query'],
        "response": state['response'],
        "intent": state['intent']
    })
    
    return state

# Build Graph
graph = StateGraph(MemoryState)
graph.add_node("memory_router", memory_router_node)
graph.add_node("memory_backends", memory_backends_node)
graph.add_node("context_manager", context_manager_node)
graph.add_node("llm_generate", generate_node)
graph.add_node("save_memory", save_memory_node)

graph.set_entry_point("memory_router")
graph.add_edge("memory_router", "memory_backends")
graph.add_edge("memory_backends", "context_manager")
graph.add_edge("context_manager", "llm_generate")
graph.add_edge("llm_generate", "save_memory")
graph.add_edge("save_memory", END)

app = graph.compile()

def process_query(query: str):
    initial_state = {
        "query": query,
        "intent": "",
        "active_backends": [],
        "raw_memory": {},
        "merged_context": "",
        "response": "",
        "messages": [],
        "memory_budget": 400
    }
    result = app.invoke(initial_state)
    return result["response"]
