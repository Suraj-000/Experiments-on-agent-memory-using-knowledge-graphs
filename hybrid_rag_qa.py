from typing import Dict, TypedDict, List, Optional, Any
from utils import GetConversation, LoadAllData
from all_prompts import SUMMARY_PROMPT, QA_PROMPT, JUDGE_PROMPT, RAG_PROMPT, EXTRACT_ENTITIES_PROMPT
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langgraph.graph import StateGraph, START, END
import os
from models import GetModels
import re
from tqdm.auto import tqdm

class AgentState(TypedDict):
    context_idx: int
    context: GetConversation
    data_path: str
    task: str
    summary: str
    retriever: Optional[Any]
    entities: List[Dict[str, str]]
    chunk_entities: List[List[Dict[str, str]]]

class HybridRAGQA:
    def __init__(self,model,entity_batch,qa_batch,judge_batch,plot=False):
        self.model=model
        self.qa_batch=qa_batch
        self.entity_batch=entity_batch
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

    def _parse_entity_json(self, raw_text: str) -> List[Dict[str, str]]:
        """
        Robustly parse the model's JSON entity output. Handles markdown fences,
        stray leading/trailing tokens, and other common local-model quirks.
        Returns [] if parsing fails after all fallbacks (never raises).
        """
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                print(f"[extract_entities] Failed to parse JSON, no list found. Raw: {text[:200]!r}")
                return []
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                print(f"[extract_entities] Failed to parse JSON after fallback. Raw: {text[:200]!r}")
                return []

        if not isinstance(parsed, list):
            print(f"[extract_entities] Parsed JSON was not a list: {type(parsed)}. Raw: {text[:200]!r}")
            return []

        cleaned = []
        for item in parsed:
            # Defensive: model sometimes emits a flat list of strings like
            # ["Caroline", "Melanie"] instead of proper {"entity","type"} objects.
            if isinstance(item, dict) and "entity" in item and "type" in item:
                entity = str(item["entity"]).strip()
                etype = str(item["type"]).strip().upper()
                if entity:
                    cleaned.append({"entity": entity, "type": etype})
            elif isinstance(item, str):
                # salvage as MISC rather than silently dropping
                stripped = item.strip()
                if stripped:
                    print(f"[extract_entities] Got bare string entity (no type): {stripped!r} — tagging as MISC")
                    cleaned.append({"entity": stripped, "type": "MISC"})
            else:
                print(f"[extract_entities] Skipping malformed entity item: {item!r} (type={type(item)})")
        return cleaned

    def extract_entities(self, chunks: List[str]) -> List[Dict[str, str]]:
        """
        Extracts named entities from a list of context chunks using batched
        GPU generation, then dedupes across all chunks.
        """
        print(f"in extract entities: batching {len(chunks)} chunks")

        formatted_prompts=[
            self._format_chat(
                EXTRACT_ENTITIES_PROMPT,
                enable_thinking=True,
                context=chunk
            )
            for chunk in chunks
        ]
        outputs = self._batch_generate(formatted_prompts,self.entity_batch)
        print(f'len of outputs : {len(outputs)}')

        per_chunk_entities=[]
        for i, output in enumerate(outputs):
            parsed = self._parse_entity_json(output)
            # print(f"parsed : {parsed}")
            if not parsed:
                print(f"[extract_entities] chunk {i} produced no valid entities")
            per_chunk_entities.append(parsed)
        return per_chunk_entities

    def build_rag_db(self,state:AgentState)->AgentState:
        """ this node will build the vector database given the context"""
        print("\n Building RAG Database")
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=50,
                separators=["\n\n", "\n", " "]
            )

        chunks = text_splitter.split_text(state["context"].get_full_conversation())
        print(f"Number of chunks created: {len(chunks)}")

        per_chunk_entities = self.extract_entities(chunks)
        state["chunk_entities"] = per_chunk_entities

        seen = {}
        for chunk_idx, chunk_ents in enumerate(per_chunk_entities):
            if not isinstance(chunk_ents, list):
                print(f"[build_rag_db] chunk {chunk_idx}: expected list of entities, got {type(chunk_ents)} — skipping")
                continue
            for item in chunk_ents:
                if not isinstance(item, dict) or "entity" not in item or "type" not in item:
                    print(f"[build_rag_db] chunk {chunk_idx}: skipping malformed entity item: {item!r}")
                    continue
                key = (item["entity"].lower(), item["type"])
                if key not in seen:
                    seen[key] = item
        state["entities"] = list(seen.values())
        print(f"total unique entities extracted : {len(state["entities"])}")

        # Build Documents with entities attached as metadata (as a searchable
        # comma-joined string, since FAISS/BM25 metadata filtering works best
        # on flat scalar values rather than nested lists)
        documents = []
        for i, (chunk, ents) in enumerate(zip(chunks, per_chunk_entities)):
            if not isinstance(ents, list):
                print(f"[build_rag_db] chunk {i}: expected list for document build, got {type(ents)} — treating as empty")
                ents = []
            entity_str = ", ".join(e["entity"] for e in ents if isinstance(e, dict) and "entity" in e)
            entity_types = list({e["type"] for e in ents if isinstance(e, dict) and "type" in e})
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "chunk_id": i,
                        "entities": entity_str,
                        "entity_types": entity_types,
                    }
                )
            )

        print("Building dense (FAISS) retriever...")
        vector_store = FAISS.from_documents(documents=documents, embedding=self.embeddings)
        dense_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        print("Building sparse (BM25) retriever...")
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5

        print("Combining into ensemble hybrid retriever...")
        hybrid_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[0.2, 0.8]  # favor semantic match slightly; tune as needed
        )

        state["retriever"] = hybrid_retriever
        print("\n Done building rag db")
        return state

    def retrieve(self, state: AgentState, query: str) -> List[Document]:
        """
        Convenience node/method to run a query against the hybrid retriever.
        Not wired into the graph yet — call directly for testing, or add as
        a graph node once your QA/judge nodes are ready to consume it.
        """
        if state.get("retriever") is None:
            raise ValueError("state['retriever'] is not built yet — call build_rag_db first")

        results = state["retriever"].invoke(query)
        return results

    def rag_qa(self,state:AgentState)->AgentState:
        """this node will answer the question"""
        print("In Hybrid rag qa")
        retriever=state["retriever"]
        questions = [state["context"].get_qa(i)[0] for i in range(state["context"].num_qa)]
        batch_docs = retriever.batch(questions)
        formatted_prompts = [
            self._format_chat(
                RAG_PROMPT, 
                enable_thinking=True,
                context="\n---\n".join([doc.page_content for doc in docs]), 
                question=q
            )
            for q, docs in zip(questions, batch_docs)
        ]

        llm_answers = self._batch_generate(formatted_prompts, self.qa_batch)
        for i, llm_answer in enumerate(llm_answers):
            state["context"].qa[i]["llm_answer"] = llm_answer

        print("Finished QA.\n" + "-"*50)
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
        for score_str in all_scores:
            match = re.search(r"([0-9]*\.?[0-9]+)", score_str)
            parsed_scores.append(float(match.group(1)) if match else 0.0)

        avg_score = sum(parsed_scores) / len(parsed_scores) if parsed_scores else 0.0
        print(f"Total Questions: {len(parsed_scores)} | Sum: {sum(parsed_scores):.2f} | Average: {avg_score:.2f}")
        print("Finished LLM Judge.\n" + "-"*50)
        return state

    def _build_graph(self):
        graph=StateGraph(AgentState)

        graph.add_node("fetch_conversation",self.fetch_conversation)
        graph.add_node("build_hybrid_rag_db",self.build_rag_db)
        graph.add_node("rag_qa",self.rag_qa)
        graph.add_node("llm_judge",self.llm_judge)

        graph.add_edge(START,"fetch_conversation")
        graph.add_edge("fetch_conversation","build_hybrid_rag_db")
        graph.add_edge("build_hybrid_rag_db","rag_qa")
        graph.add_edge("rag_qa","llm_judge")
        graph.add_edge("llm_judge",END)

        app=graph.compile()

        if self.plot:
            png = app.get_graph().draw_mermaid_png()

            os.makedirs("images",exist_ok=True)
            with open("images/hybrid_rag_qa.png", "wb") as f:
                f.write(png)
            print("Saved as hybrid_rag_qa.png")

if __name__=="__main__":

    model_name="./local-Qwen3-14B-Instruct"
    model=GetModels(model_name=model_name,load_type=True)

    graph=HybridRAGQA(model=model,entity_batch=128,qa_batch=8,judge_batch=16,plot=True)
    conv=graph.app.invoke({
        "context_idx":0,
        "context":"",
        "data_path":"files/locomo10.json"
    })
    # state=AgentState(
    #     context_idx=0,
    #     data_path="files/locomo10.json"
    # )
    # s=graph.fetch_conversation(state)
    # s=graph.build_rag_db(s)

    # test_query = "What career is Caroline considering?"
    # results = graph.retrieve(s, test_query)
    # print(f"\n--- Hybrid retrieval results for query: {test_query!r} ---")
    # for i, doc in enumerate(results):
    #     print(f"\n[{i}] chunk_id={doc.metadata.get('chunk_id')} entities={doc.metadata.get('entities')}")
    #     print(doc.page_content)