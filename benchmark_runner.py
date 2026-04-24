import os
import json
from dotenv import load_dotenv

# Load env before importing agent
load_dotenv()

from agent import process_query, short_term, long_term, episodic, semantic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

raw_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def process_query_no_memory(query: str):
    return raw_llm.invoke([HumanMessage(content=query)]).content

def run_benchmark():
    print("--- Khởi tạo Semantic Memory ---")
    semantic.add_knowledge([
        "LangGraph là một thư viện để xây dựng stateful, multi-actor applications với LLMs.",
        "ChromaDB là một open-source vector database được thiết kế dành cho các ứng dụng AI."
    ])

    scenarios = [
        {
            "name": "1. Profile Recall (Tên)",
            "turns": [
                "Chào bạn, tôi tên là Linh.",
                "Hôm nay trời đẹp quá.",
                "Tên của tôi là gì?"
            ],
            "expected": "Linh"
        },
        {
            "name": "2. Allergy conflict update",
            "turns": [
                "Tôi bị dị ứng sữa bò.",
                "À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.",
                "Nếu bạn làm bánh cho tôi, bạn cần tránh nguyên liệu gì?"
            ],
            "expected": "đậu nành"
        },
        {
            "name": "3. Episodic Recall",
            "turns": [
                "Để debug lỗi connection refuse, tôi phải dùng docker service name.",
                "Cảm ơn bạn đã ghi nhận.",
                "Hôm trước tôi bảo bạn debug lỗi connection refuse như thế nào?"
            ],
            "expected": "dùng docker service name"
        },
        {
            "name": "4. Semantic Retrieval (FAQ)",
            "turns": [
                "LangGraph là gì?"
            ],
            "expected": "thư viện xây dựng stateful"
        },
        {
            "name": "5. Profile Update (Nghề nghiệp)",
            "turns": [
                "Tôi đang làm giáo viên.",
                "Tôi vừa nghỉ việc và chuyển sang làm lập trình viên.",
                "Nghề nghiệp hiện tại của tôi là gì?"
            ],
            "expected": "lập trình viên"
        },
        {
            "name": "6. Episodic + Profile (Món ăn)",
            "turns": [
                "Tôi rất thích ăn Phở.",
                "Trưa nay ăn gì cho ngon nhỉ?"
            ],
            "expected": "Gợi ý Phở"
        },
        {
            "name": "7. Semantic Retrieval 2",
            "turns": [
                "ChromaDB dùng để làm gì?"
            ],
            "expected": "vector database"
        },
        {
            "name": "8. Short-term Memory limit (Trim)",
            "turns": [
                "Quả táo màu đỏ.",
                "Quả cam màu cam.",
                "Quả chuối màu vàng.",
                "Tôi vừa kể tên quả gì đầu tiên?"
            ],
            "expected": "quả táo"
        },
        {
            "name": "9. Profile fact - Pets",
            "turns": [
                "Tôi có nuôi một con mèo tên là Tom.",
                "Thú cưng của tôi tên là gì?"
            ],
            "expected": "Tom"
        },
        {
            "name": "10. Context merging",
            "turns": [
                "Tôi tên là Minh.",
                "Tôi thích lập trình AI.",
                "LangGraph dùng làm gì và bạn có nghĩ tôi nên học nó không?"
            ],
            "expected": "Nhắc đến tên Minh, thích AI, và giải thích LangGraph"
        }
    ]

    print("Bắt đầu chạy benchmark so sánh No-Memory và With-Memory...\n")
    
    benchmark_results = []
    
    for i, s in enumerate(scenarios, 1):
        print(f"==================================================")
        print(f"Scenario {i}: {s['name']}")
        print(f"==================================================")
        
        # Clear short-term memory before each scenario
        short_term.clear() 
        
        scenario_data = {
            "scenario": s['name'],
            "expected": s['expected'],
            "turns": [],
            "no_memory_final_result": "",
            "with_memory_final_result": ""
        }
        
        for turn_idx, turn in enumerate(s['turns']):
            print(f"👤 User: {turn}")
            
            # Nếu là câu hỏi cuối cùng mang tính quyết định để kiểm tra memory
            if turn_idx == len(s['turns']) - 1:
                resp_no_mem = process_query_no_memory(turn)
                resp_with_mem = process_query(turn)
                print(f"\n🤖 [No-Memory Agent]: {resp_no_mem}")
                print(f"🧠 [With-Memory Agent]: {resp_with_mem}")
                print(f"🎯 [Expected]: {s['expected']}\n")
                
                scenario_data["turns"].append({"user": turn, "with_memory_agent": resp_with_mem})
                scenario_data["no_memory_final_result"] = resp_no_mem
                scenario_data["with_memory_final_result"] = resp_with_mem
            else:
                # Các turn trước đó chỉ cho Agent có memory xử lý để tích lũy context
                resp_with_mem = process_query(turn)
                print(f"🧠 [With-Memory Agent]: {resp_with_mem}\n")
                
                scenario_data["turns"].append({"user": turn, "with_memory_agent": resp_with_mem})
                
        benchmark_results.append(scenario_data)

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=4)
        
    print(f"\nĐã lưu kết quả benchmark vào file benchmark_results.json")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Vui lòng thiết lập OPENAI_API_KEY trong file .env trước khi chạy benchmark.")
    else:
        run_benchmark()
