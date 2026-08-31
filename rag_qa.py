import os
import re
from typing import TypedDict, Any
from tqdm.auto import tqdm

# Langchain & Langgraph imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
# Custom imports
from utils import GetConversation, LoadAllData
from all_prompts import RAG_PROMPT, JUDGE_PROMPT
from models import GetModels

class JudgeResult(BaseModel):
    score: float = Field(description="Score: 0.0, 0.2, 0.4, 0.6, 0.8, or 1.0")

    @field_validator("score")
    @classmethod
    def validate_score(cls, v):
        valid = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
        if v not in valid:
            raise ValueError(f"Score must be one of {valid}")
        return v


judge_parser = PydanticOutputParser(pydantic_object=JudgeResult)


class AgentState(TypedDict):
    context_idx: int
    context: Any  
    data_path: str
    retriever: Any

class RAGQA:
    def __init__(self,model,qa_batch,judge_batch,plot=False):
        self.model=model
        self.qa_batch=qa_batch
        self.judge_batch=judge_batch
        self.plot=plot
        self.role_map = {"human": "user", "ai": "assistant", "system": "system"}
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.app = self._build_graph()

    def _format_chat(self, prompt_template,enable_thinking=False, **kwargs) -> str:
        """Invoke a ChatPromptTemplate and render it through the model's chat template."""
        messages = prompt_template.invoke(kwargs).to_messages()
        chat_messages = [
            {"role": self.role_map.get(m.type, m.type), "content": m.content}
            for m in messages
        ]
        return self.model.tokenizer.apply_chat_template(
            chat_messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

    def _batch_generate(self, formatted_prompts: list, batch_size: int) -> list:
        """Helper to generate text in batches."""
        if self.model.pipe is None:
            # Fallback for testing when no model is loaded
            return [f"Mock answer for prompt {i}" for i in range(len(formatted_prompts))]

        results = []
        for i in tqdm(range(0, len(formatted_prompts), batch_size), desc="Generating"):
            batch = formatted_prompts[i:i + batch_size]
            outputs = self.model.pipe(batch, batch_size=batch_size)
            
            for out in outputs:
                item = out[0] if isinstance(out, list) else out
                results.append(item["generated_text"].strip())
                
        return results

    def fetch_conversation(self, state: AgentState) -> AgentState:
        """Fetches the info of conversation."""
        all_data = LoadAllData(data_path=state["data_path"])
        sample = all_data.get_sample_conversation(idx=state["context_idx"])
        state["context"] = GetConversation(sample)
        
        print(f"\n[Conversation metadata: ID {state['context_idx']}]")
        print(f"Speakers: {state['context'].speakers}")
        print(f"Num Sessions: {len(state['context'].session_ids)} | Num QA's: {state['context'].num_qa}\n")
        return state

    def build_rag_db(self, state: AgentState) -> AgentState:
        """Builds the vector database given the context."""
        print("Building RAG Database...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", " "]
        )
        chunks = text_splitter.split_text(state["context"].get_full_conversation())
        print(f"Number of chunks created: {len(chunks)}")

        vector_store = FAISS.from_texts(texts=chunks, embedding=self.embeddings)
        state["retriever"] = vector_store.as_retriever(search_kwargs={"k": 4})
        return state

    def rag_qa(self, state: AgentState) -> AgentState:
        """Answers the questions using retrieved context."""
        print("Running RAG QA...")
        retriever = state["retriever"]

        questions = [state["context"].get_qa(i)[0] for i in range(state["context"].num_qa)]
        batch_docs = retriever.batch(questions)

        formatted_prompts = [
            self._format_chat(
                RAG_PROMPT,
                enable_thinking=False,
                context="\n---\n".join([doc.page_content for doc in docs]), 
                question=q
            )
            for q, docs in zip(questions, batch_docs)
        ]

        llm_answers = self._batch_generate(formatted_prompts, self.qa_batch)
        for i, llm_answer in enumerate(llm_answers):
            state["context"].qa[i]["llm_answer"] = llm_answer

        print("Finished RAG QA.\n" + "-"*50)
        return state
    



    def llm_judge(self, state: AgentState) -> AgentState:
        """Compares AI answers with true golden answers to generate a score."""
        print("Running LLM Judge...")
    
        formatted_prompts = [
            self._format_chat(
                JUDGE_PROMPT, 
                enable_thinking=False,
                llm_answer=state['context'].get_qla(i)[2], 
                true_answer=state['context'].get_qla(i)[1]
            )
            for i in range(state['context'].num_qa)
        ]
        all_scores = self._batch_generate(formatted_prompts, self.judge_batch)

        parsed_scores = []

        for output in all_scores:
            try:
                result = judge_parser.parse(output)
                parsed_scores.append(result.score)
            except Exception as e:
                print(f"INVALID JUDGE OUTPUT: {repr(output)}")
                print(f"Parse error: {e}")
                parsed_scores.append(0.0)
        print(parsed_scores)
        avg_score = sum(parsed_scores) / len(parsed_scores) if parsed_scores else 0.0
        print(f"Total Questions: {len(parsed_scores)} | Sum: {sum(parsed_scores):.2f} | Average: {avg_score:.2f}")
        print("Finished LLM Judge.\n" + "-"*50)
        return state
    
    def _build_graph(self):
        """Constructs and compiles the LangGraph workflow."""
        graph = StateGraph(AgentState)

        graph.add_node("fetch_conversation", self.fetch_conversation)
        graph.add_node("build_rag_db", self.build_rag_db)
        graph.add_node("rag_qa", self.rag_qa)
        graph.add_node("llm_judge", self.llm_judge)

        graph.add_edge(START, "fetch_conversation")
        graph.add_edge("fetch_conversation", "build_rag_db")
        graph.add_edge("build_rag_db", "rag_qa")
        graph.add_edge("rag_qa", "llm_judge")
        graph.add_edge("llm_judge", END)

        app = graph.compile()

        if self.plot:
            os.makedirs("images", exist_ok=True)
            with open("images/rag_qa.png", "wb") as f:
                f.write(app.get_graph().draw_mermaid_png())
            print("Graph image saved to images/rag_qa.png")
            
        return app

if __name__=="__main__":
    # model_name = "./local-Qwen3-14B-Instruct"
    # model_name="./gpt-oss-20b"
    model_name="./DeepSeek-R1-Distill-Qwen-14B"

    model = GetModels(model_name=model_name, load_type=True)

    pipeline = RAGQA(model=model, qa_batch=48, judge_batch=30, plot=False)
    
    conv = pipeline.app.invoke({
        "context_idx": 0,
        "context": "",
        "data_path": "files/locomo10.json",
        "retriever": None
    })