from langgraph.graph import StateGraph,START,END
import argparse
import os 
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import time
import torch 
import numpy as np
import sys
import random
from typing import Dict,TypedDict,List
from utils import GetConversation,LoadAllData
from models import GetModels
from summarise_qa import SummariseContext
from full_context_qa import FullContext
from rag_qa import RAGQA
from hybrid_rag_qa import HybridRAGQA
from lightrag_qa import LightRAG
from helper_fns import Logger,set_device,print_star

device=set_device()
# device=torch.device("cpu")
original_stdout = sys.stdout

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def logging(s='log', model_name="A2"):
    global current_logger
    logs_dir=os.path.join("logs",model_name)
    os.makedirs(logs_dir,exist_ok=True)
    log_path = os.path.join(logs_dir,f'output_{s}.txt')
    current_logger = Logger(log_path)
    sys.stdout = current_logger
    print("Logging started at - ", log_path)
    print("Using device:", device)


class AgentState(TypedDict):
    context_idx: int
    context: GetConversation
    data_path: str
    task: str
    summary:str
    retriever:None

class ConstructGraph:

    def route_to_task(self,state: AgentState):
        """ this node will route to the task assigned."""

        if state["task"]=="0":
            return "Full_context"
        elif state['task']=="1":
            return "Summarised_context"
        elif state['task']=="2":
            return "Naive_RAG"
        elif state["task"]=="3":
            return "Hybrid_RAG"
        elif state["task"]=="4":
            return "Light_RAG"

    def __init__(self,model,plot=False):
        self.plot=plot
        full_context_nodes=FullContext(model=model,qa_batch=8,judge_batch=24)
        summarise_nodes=SummariseContext(model=model,summarise_batch=10,qa_batch=10,judge_batch=24)
        rag_nodes=RAGQA(model=model,qa_batch=128,judge_batch=128)
        hybrid_rag_nodes=HybridRAGQA(model=model,entity_batch=128,qa_batch=16,judge_batch=24)
        light_rag_nodes=LightRAG(model=model,entity_batch=64,summary_batch=32,keyword_batch=16,qa_batch=8,judge_batch=96,plot=False,merge_threshold=2,entity_top_k=6,relation_top_k=6,chunk_top_k=4)

        graph=StateGraph(AgentState)

        graph.add_node("router",lambda state:state)

        graph.add_node("fetch_conversation_0",full_context_nodes.fetch_conversation)
        graph.add_node("qa_0",full_context_nodes.question_answer)
        graph.add_node("llm_judge_0",full_context_nodes.llm_judge)

        graph.add_node("fetch_conversation_1",summarise_nodes.fetch_conversation)
        graph.add_node("summarise_1",summarise_nodes.llm_summariser)
        graph.add_node("qa_1",summarise_nodes.question_answer)
        graph.add_node("llm_judge_1",summarise_nodes.llm_judge)

        graph.add_node("fetch_conversation_2",rag_nodes.fetch_conversation)
        graph.add_node("build_rag_db_2",rag_nodes.build_rag_db)
        graph.add_node("rag_qa_2",rag_nodes.rag_qa)
        graph.add_node("llm_judge_2",rag_nodes.llm_judge)

        graph.add_node("fetch_conversation_3",hybrid_rag_nodes.fetch_conversation)
        graph.add_node("build_hybrid_rag_db_3",hybrid_rag_nodes.build_rag_db)
        graph.add_node("rag_qa_3",hybrid_rag_nodes.rag_qa)
        graph.add_node("llm_judge_3",hybrid_rag_nodes.llm_judge)


        graph.add_node("fetch_conversation_4", light_rag_nodes.fetch_conversation)
        graph.add_node("chunk_document_4", light_rag_nodes.chunk_document)
        graph.add_node("extract_graph_elements_4", light_rag_nodes.extract_graph_elements)
        graph.add_node("build_knowledge_graph_4", light_rag_nodes.build_knowledge_graph)
        graph.add_node("build_vector_indices_4", light_rag_nodes.build_vector_indices)
        graph.add_node("dual_level_qa_4", light_rag_nodes.dual_level_qa)
        graph.add_node("llm_judge_4", light_rag_nodes.llm_judge)


        graph.add_edge(START,"router")
        graph.add_conditional_edges(
            "router",
            self.route_to_task,
            {
                "Full_context":"fetch_conversation_0",
                "Summarised_context":"fetch_conversation_1",
                "Naive_RAG":"fetch_conversation_2",
                "Hybrid_RAG":"fetch_conversation_3",
                "Light_RAG":"fetch_conversation_4"
            }
        )

        graph.add_edge("fetch_conversation_0","qa_0")
        graph.add_edge("qa_0","llm_judge_0")
        graph.add_edge("llm_judge_0",END)

        graph.add_edge("fetch_conversation_1","summarise_1")
        graph.add_edge("summarise_1","qa_1")
        graph.add_edge("qa_1","llm_judge_1")
        graph.add_edge("llm_judge_1",END)

        graph.add_edge("fetch_conversation_2","build_rag_db_2")
        graph.add_edge("build_rag_db_2","rag_qa_2")
        graph.add_edge("rag_qa_2","llm_judge_2")
        graph.add_edge("llm_judge_2",END)

        graph.add_edge("fetch_conversation_3","build_hybrid_rag_db_3")
        graph.add_edge("build_hybrid_rag_db_3","rag_qa_3")
        graph.add_edge("rag_qa_3","llm_judge_3")
        graph.add_edge("llm_judge_3",END)


        graph.add_edge("fetch_conversation_4", "chunk_document_4")
        graph.add_edge("chunk_document_4", "extract_graph_elements_4")
        graph.add_edge("extract_graph_elements_4", "build_knowledge_graph_4")
        graph.add_edge("build_knowledge_graph_4", "build_vector_indices_4")
        graph.add_edge("build_vector_indices_4", "dual_level_qa_4")
        graph.add_edge("dual_level_qa_4", "llm_judge_4")
        graph.add_edge("llm_judge_4", END)


        self.app=graph.compile()

        if self.plot:
            png = self.app.get_graph().draw_mermaid_png()

            os.makedirs("images",exist_ok=True)
            with open("images/graph.png", "wb") as f:
                f.write(png)
            print("Saved as graph.png")

def parsing_fn():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Specify the model to use",choices=["Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-Small-3.1-24B-Instruct-2503","Qwen/Qwen3-8B","meta-llama/Llama-3.1-8B-Instruct","Qwen/Qwen2.5-1.5B-Instruct"])
    parser.add_argument("--task", type=str, default="0",help="specify the task which you want to perform",choices=["0","1","2","3","4"])
    args = parser.parse_args()
    return args


if __name__=="__main__":
    args=parsing_fn()

    logging(args.task,"testing")
    print_star()
    start_time=time.time()
    set_seed(42)

    all_models=["./local-Qwen3-14B","./local-Qwen3-14B-Instruct","OpenPipe/Qwen3-14B-Instruct","Qwen/Qwen2.5-1.5B-Instruct","mistralai/Ministral-8B-Instruct-2410","Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-Small-3.1-24B-Instruct-2503","meta-llama/Llama-3.1-8B-Instruct"]

    model=GetModels(model_name=all_models[1],load_type=True)
    graph=ConstructGraph(model=model,plot=False)
    conv=graph.app.invoke({
        "context_idx":0,
        "context":"",
        "data_path":"files/locomo10.json",
        "summary":"",
        "task":args.task
    })

    print_star()
    print(f"Total execution time: {(time.time() - start_time)/60.0:.2f} minutes")
    print_star()

    sys.stdout = original_stdout; current_logger.close()
