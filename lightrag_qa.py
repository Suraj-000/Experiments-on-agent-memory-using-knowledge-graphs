"""
LightRAG-style pipeline, built on top of your existing HybridRAGQA scaffolding
(graph_construction.py), same conventions: AgentState TypedDict flowing through
LangGraph nodes, self.model.pipe for batched local HF generation, role_map +
tokenizer.apply_chat_template for prompt formatting.

Implements the core ideas from the LightRAG paper (Guo et al., 2024):

  1. Graph-based indexing: chunk the document, extract entities +
     relationships (+ chunk-level high-level keywords) per chunk with a
     single LLM call, then merge everything into one knowledge graph
     (networkx.Graph) - duplicate entities/relationships seen across chunks
     get their descriptions merged (and, once a node/edge has enough
     fragments, re-summarized with the LLM).
  2. Dual-level retrieval:
       - low-level  -> query's specific-entity keywords are matched against
                        an entity vector index, then expanded 1-hop over the
                        graph to pull in directly connected relationships and
                        their source chunks.
       - high-level -> query's thematic keywords are matched against a
                        relationship vector index, pulling in the
                        broader/global relationships and their source chunks.
     A supplementary direct chunk-similarity search is also mixed in, so a
     retrieval "miss" on the graph side still has a naive-RAG fallback.
  3. Context is assembled from three sections (entities / relationships /
     source excerpts) and handed to the LLM for final answer generation.

No internet is required at any point after model + embedding weights are
already on disk locally - graph storage is plain networkx in-memory, no
external graph DB.
"""

from typing import Dict, TypedDict, List, Optional, Any
import json
import os
import re
import time
from tqdm.auto import tqdm
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import networkx as nx
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END

from utils import GetConversation, LoadAllData
from models import GetModels
from all_prompts import JUDGE_PROMPT
from lightrag_prompts import (
    GRAPH_EXTRACTION_PROMPT,
    ENTITY_MERGE_SUMMARY_PROMPT,
    RELATION_MERGE_SUMMARY_PROMPT,
    KEYWORD_EXTRACTION_PROMPT,
    LIGHTRAG_ANSWER_PROMPT,
)


class AgentState(TypedDict):
    context_idx: int
    context: GetConversation
    data_path: str
    chunks: List[str]
    chunk_store: Dict[int, str]
    graph_elements: List[Dict[str, Any]]
    graph: Any                      # networkx.Graph
    entity_vdb: Optional[Any]       # FAISS
    relation_vdb: Optional[Any]     # FAISS
    chunk_vdb: Optional[Any]        # FAISS


class LightRAG:
    def __init__(
        self,
        model,
        entity_batch: int,
        summary_batch: int,
        keyword_batch: int,
        qa_batch: int,
        judge_batch: int,
        plot: bool = False,
        merge_threshold: int = 2,
        entity_top_k: int = 6,
        relation_top_k: int = 6,
        chunk_top_k: int = 4,
    ):
        self.model = model
        self.entity_batch = entity_batch
        self.summary_batch = summary_batch
        self.keyword_batch = keyword_batch
        self.qa_batch = qa_batch
        self.judge_batch = judge_batch
        self.plot = plot

        # how many merged description fragments a node/edge needs before we
        # bother spending an extra LLM call to re-summarize it
        self.merge_threshold = merge_threshold
        self.entity_top_k = entity_top_k
        self.relation_top_k = relation_top_k
        self.chunk_top_k = chunk_top_k

        self.role_map = {"human": "user", "ai": "assistant", "system": "system"}
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ------------------------------------------------------------------ #
    # small shared helpers
    # ------------------------------------------------------------------ #
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

    def _batch_generate(self, formatted_prompts: List[str], batch_size: int) -> List[str]:
        """Run a batch of already-templated prompts through self.model.pipe."""
        results = []
        for i in tqdm(range(0, len(formatted_prompts), batch_size), desc="Generating"):
            batch = formatted_prompts[i:i + batch_size]
            outputs = self.model.pipe(batch, batch_size=batch_size)
            
            for out in outputs:
                item = out[0] if isinstance(out, list) else out
                results.append(item["generated_text"].strip())
                
        return results

    @staticmethod
    def _robust_json_parse(raw_text: str, expected_type=dict, context_label: str = ""):
        """
        Same defensive pattern as your original _parse_entity_json: strip
        markdown fences, try json.loads, fall back to regex-extracting the
        first {...} or [...] blob, never raise.
        """
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        label = f" {context_label}" if context_label else ""

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pattern = r"\{.*\}" if expected_type is dict else r"\[.*\]"
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                print(f"[_robust_json_parse{label}] no {expected_type.__name__} found. Raw: {text[:200]!r}")
                return expected_type()
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                print(f"[_robust_json_parse{label}] failed after fallback. Raw: {text[:200]!r}")
                return expected_type()

        if not isinstance(parsed, expected_type):
            print(f"[_robust_json_parse{label}] wrong type {type(parsed)}. Raw: {text[:200]!r}")
            return expected_type()
        return parsed

    # ------------------------------------------------------------------ #
    # graph nodes
    # ------------------------------------------------------------------ #
    def fetch_conversation(self, state: AgentState) -> AgentState:
        all_data = LoadAllData(data_path=state["data_path"])
        sample = all_data.get_sample_conversation(idx=state["context_idx"])
        state["context"] = GetConversation(sample)
        print(f"\n Conversation meta data: ID {state['context_idx']} \n \
                Speakers: {state['context'].speakers} \n \
                Num Sessions: {len(state['context'].session_ids)}, Num QA's: {state['context'].num_qa} \n \
            ")
        return state

    def chunk_document(self, state: AgentState) -> AgentState:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " "],
        )
        chunks = text_splitter.split_text(state["context"].get_full_conversation())
        state["chunks"] = chunks
        state["chunk_store"] = {i: c for i, c in enumerate(chunks)}
        print(f"Number of chunks created: {len(chunks)}")
        return state

    def extract_graph_elements(self, state: AgentState) -> AgentState:
        """Batched single-pass entity + relationship + content-keyword extraction per chunk."""
        chunks = state["chunks"]
        print(f"Extracting graph elements from {len(chunks)} chunks")
        p="/home/scai/msr/aiy257586/scratch/aiy257586/graph/outputs/graph_elements.json"
        if os.path.exists(p):
            with open(p, 'r') as f:
                elements = json.load(f)
            print(f"Graph elements already exists\n")
            state["graph_elements"]=elements
            return state

        formatted_prompts = [
            self._format_chat(GRAPH_EXTRACTION_PROMPT,enable_thinking=True, context=chunk) for chunk in chunks
        ]

        if self.model.pipe is None:
            raw_outputs = ['{"entities":[],"relationships":[],"content_keywords":""}'] * len(chunks)
        else:
            raw_outputs = self._batch_generate(formatted_prompts, self.entity_batch)

        elements = []
        for i, raw in enumerate(raw_outputs):
            parsed = self._robust_json_parse(raw, expected_type=dict, context_label=f"chunk {i}")
            entities = [
                e for e in parsed.get("entities", [])
                if isinstance(e, dict) and str(e.get("name", "")).strip()
            ]
            relationships = [
                r for r in parsed.get("relationships", [])
                if isinstance(r, dict) and str(r.get("source", "")).strip() and str(r.get("target", "")).strip()
            ]
            elements.append({
                "entities": entities,
                "relationships": relationships,
                "content_keywords": parsed.get("content_keywords", ""),
            })
            if not entities and not relationships:
                print(f"[extract_graph_elements] chunk {i} produced no graph elements")

        os.makedirs("/home/scai/msr/aiy257586/scratch/aiy257586/graph/outputs",exist_ok=True)
        with open(p, "w") as f:
            json.dump(elements, f)

        state["graph_elements"] = elements
        return state

    def _summarize_descriptions(
        self,
        graph: "nx.Graph",
        node_fragments: Dict[str, List[str]],
        edge_fragments: Dict[Any, List[str]],
    ) -> None:
        """
        For nodes/edges seen in >= merge_threshold chunks, re-summarize their
        merged description fragments with a single batched LLM call each
        (nodes and edges are batched separately). Otherwise just dedupe-join.
        """
        node_keys_to_summarize, node_prompts = [], []
        for key, frags in node_fragments.items():
            frags = [f for f in frags if f]
            if len(frags) >= self.merge_threshold and self.model.pipe is not None:
                node_keys_to_summarize.append(key)
                node_prompts.append(self._format_chat(
                    ENTITY_MERGE_SUMMARY_PROMPT,
                    enable_thinking=True,
                    entity_name=graph.nodes[key]["display_name"],
                    entity_type=graph.nodes[key]["entity_type"],
                    descriptions="\n".join(f"- {f}" for f in frags),
                ))
            else:
                graph.nodes[key]["description"] = " ".join(dict.fromkeys(frags)) or "No description available."

        if node_prompts:
            summaries = self._batch_generate(node_prompts, self.summary_batch)
            for key, summary in zip(node_keys_to_summarize, summaries):
                graph.nodes[key]["description"] = summary

        edge_keys_to_summarize, edge_prompts = [], []
        for key, frags in edge_fragments.items():
            frags = [f for f in frags if f]
            if len(frags) >= self.merge_threshold and self.model.pipe is not None:
                edge_keys_to_summarize.append(key)
                a, b = key
                edge_prompts.append(self._format_chat(
                    RELATION_MERGE_SUMMARY_PROMPT,
                    enable_thinking=True,
                    source=graph.nodes[a]["display_name"],
                    target=graph.nodes[b]["display_name"],
                    descriptions="\n".join(f"- {f}" for f in frags),
                ))
            else:
                graph.edges[key]["description"] = " ".join(dict.fromkeys(frags)) or "No description available."

        if edge_prompts:
            summaries = self._batch_generate(edge_prompts, self.summary_batch)
            for key, summary in zip(edge_keys_to_summarize, summaries):
                graph.edges[key]["description"] = summary

    def save_graph(self, graph, filename="knowledge_graph.graphml"):
        G = graph.copy()

        for _, attrs in G.nodes(data=True):
            attrs["source_chunks"] = ",".join(map(str, attrs["source_chunks"]))

        for _, _, attrs in G.edges(data=True):
            attrs["keywords"] = ",".join(attrs["keywords"])
            attrs["source_chunks"] = ",".join(map(str, attrs["source_chunks"]))

        nx.write_graphml(G, filename)

    def build_knowledge_graph(self, state: AgentState) -> AgentState:
        """Merge per-chunk entities/relationships into a single deduplicated networkx graph."""
        start_time=time.time()
        print("\nBuilding knowledge graph")
        graph = nx.Graph()
        elements = state["graph_elements"]

        node_fragments: Dict[str, List[str]] = {}
        edge_fragments: Dict[Any, List[str]] = {}

        for chunk_id, elem in enumerate(elements):
            for ent in elem["entities"]:
                name = str(ent["name"]).strip()
                key = name.lower()
                if not graph.has_node(key):
                    graph.add_node(
                        key,
                        display_name=name,
                        entity_type=str(ent.get("type", "MISC")).strip().upper() or "MISC",
                        source_chunks=set(),
                    )
                graph.nodes[key]["source_chunks"].add(chunk_id)
                node_fragments.setdefault(key, []).append(str(ent.get("description", "")).strip())

            for rel in elem["relationships"]:
                src, tgt = str(rel["source"]).strip(), str(rel["target"]).strip()
                if not src or not tgt or src.lower() == tgt.lower():
                    continue
                src_key, tgt_key = src.lower(), tgt.lower()

                # relation-extraction sometimes names an endpoint the entity
                # list missed - create a placeholder node so the graph stays
                # connected rather than silently dropping the edge
                for key, display in ((src_key, src), (tgt_key, tgt)):
                    if not graph.has_node(key):
                        graph.add_node(key, display_name=display, entity_type="MISC", source_chunks=set())
                    graph.nodes[key]["source_chunks"].add(chunk_id)

                edge_key = tuple(sorted([src_key, tgt_key]))
                if not graph.has_edge(*edge_key):
                    graph.add_edge(*edge_key, keywords=set(), weight=0, source_chunks=set())

                graph.edges[edge_key]["keywords"].update(
                    k.strip() for k in str(rel.get("keywords", "")).split(",") if k.strip()
                )
                try:
                    graph.edges[edge_key]["weight"] += int(rel.get("strength", 1))
                except (ValueError, TypeError):
                    graph.edges[edge_key]["weight"] += 1
                graph.edges[edge_key]["source_chunks"].add(chunk_id)
                edge_fragments.setdefault(edge_key, []).append(str(rel.get("description", "")).strip())

        self._summarize_descriptions(graph, node_fragments, edge_fragments)
        
        self.save_graph(graph)
        state["graph"] = graph
        print(f"\n Knowledge graph built: {graph.number_of_nodes()} entities, "
              f"{graph.number_of_edges()} relationships")

        if self.plot:
            self._plot_graph(graph)

        print(f"Total execution time: {(time.time() - start_time)/60.0:.2f} minutes")

        return state

    def _plot_graph(self, graph: "nx.Graph") -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs("images", exist_ok=True)
            plt.figure(figsize=(14, 10))
            pos = nx.spring_layout(graph, seed=42, k=0.6)
            labels = {n: d["display_name"] for n, d in graph.nodes(data=True)}
            nx.draw_networkx_nodes(graph, pos, node_size=350, node_color="#89CFF0")
            nx.draw_networkx_edges(graph, pos, alpha=0.4)
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("images/lightrag_knowledge_graph.png", dpi=200)
            plt.close()
            print("Saved as images/lightrag_knowledge_graph.png")
        except ImportError:
            print("[plot] matplotlib not available, skipping graph plot")

    def build_vector_indices(self, state: AgentState) -> AgentState:
        """Build the three FAISS indices LightRAG's dual-level retrieval reads from."""
        graph = state["graph"]

        entity_docs = [
            Document(
                page_content=f"{attrs['display_name']} ({attrs['entity_type']}): {attrs.get('description', '')}",
                metadata={
                    "entity_key": key,
                    "display_name": attrs["display_name"],
                    "entity_type": attrs["entity_type"],
                    "source_chunks": ",".join(str(c) for c in attrs["source_chunks"]),
                },
            )
            for key, attrs in graph.nodes(data=True)
        ]

        relation_docs = [
            Document(
                page_content=(
                    f"{graph.nodes[a]['display_name']} <-> {graph.nodes[b]['display_name']} "
                    f"[{', '.join(sorted(attrs['keywords']))}]: {attrs.get('description', '')}"
                ),
                metadata={
                    "source_key": a,
                    "target_key": b,
                    "source_chunks": ",".join(str(c) for c in attrs["source_chunks"]),
                },
            )
            for a, b, attrs in graph.edges(data=True)
        ]

        chunk_docs = [
            Document(page_content=text, metadata={"chunk_id": cid})
            for cid, text in state["chunk_store"].items()
        ]

        print(f"Indexing {len(entity_docs)} entities, {len(relation_docs)} relationships, "
              f"{len(chunk_docs)} chunks...")

        state["entity_vdb"] = FAISS.from_documents(entity_docs, self.embeddings) if entity_docs else None
        state["relation_vdb"] = FAISS.from_documents(relation_docs, self.embeddings) if relation_docs else None
        state["chunk_vdb"] = FAISS.from_documents(chunk_docs, self.embeddings) if chunk_docs else None
        return state

    def _retrieve_dual_level(self, state: AgentState, low_kw: str, high_kw: str):
        """
        Core of LightRAG's retrieval: low-level keywords hit the entity index
        and expand 1-hop over the graph; high-level keywords hit the
        relationship index directly. A direct chunk-similarity search is
        mixed in as a naive-RAG safety net.
        """
        graph = state["graph"]
        chunk_store = state["chunk_store"]

        entity_hits: List[Document] = []
        relation_hits: List[Document] = []
        chunk_ids = set()
        imp_chunks=set()

        if low_kw.strip() and state["entity_vdb"] is not None:
            docs = state["entity_vdb"].similarity_search(low_kw, k=self.entity_top_k)
            for d in docs:
                entity_hits.append(d)
                key = d.metadata["entity_key"]
                chunk_ids.update(graph.nodes[key].get("source_chunks", []))
                for nbr in graph.neighbors(key):
                    edge_key = tuple(sorted([key, nbr]))
                    chunk_ids.update(graph.edges[edge_key].get("source_chunks", []))

        if high_kw.strip() and state["relation_vdb"] is not None:
            docs = state["relation_vdb"].similarity_search(high_kw, k=self.relation_top_k)
            for d in docs:
                relation_hits.append(d)
                src_chunks = d.metadata.get("source_chunks", "")
                chunk_ids.update(int(c) for c in src_chunks.split(",") if c)

        if state["chunk_vdb"] is not None:
            fallback_query = f"{low_kw} {high_kw}".strip()
            if fallback_query:
                docs = state["chunk_vdb"].similarity_search(fallback_query, k=self.chunk_top_k)
                for d in docs:
                    imp_chunks.add(d.metadata["chunk_id"])

        entities_context = "\n".join(dict.fromkeys(d.page_content for d in entity_hits)) or "None"
        relationships_context = "\n".join(dict.fromkeys(d.page_content for d in relation_hits)) or "None"
        # sources_context = "\n---\n".join(chunk_store[c] for c in sorted(chunk_ids)[:5]) or "None"
        imp_context = "\n---\n".join(chunk_store[c] for c in sorted(imp_chunks)) or "None"

        return entities_context, relationships_context, imp_context

    def dual_level_qa(self, state: AgentState) -> AgentState:
        print("In dual_level_qa")
        num_qa = state["context"].num_qa
        questions = []
        keyword_prompts = []
        for i in range(num_qa):
            question, _ = state["context"].get_qa(i)
            questions.append(question)
            keyword_prompts.append(self._format_chat(KEYWORD_EXTRACTION_PROMPT, enable_thinking=True,question=question))
        
        print("Extracting question keywords")
        if self.model.pipe is None:
            keyword_outputs = ['{"low_level_keywords":"","high_level_keywords":""}'] * len(questions)
        else:
            keyword_outputs = self._batch_generate(keyword_prompts, self.keyword_batch)
        print("Done with questions")
        print("Retrieving context for all questions")
        answer_prompts = []
        for question, raw_kw in zip(questions, keyword_outputs):
            parsed_kw = self._robust_json_parse(raw_kw, expected_type=dict, context_label="keywords")
            low_kw = str(parsed_kw.get("low_level_keywords", ""))
            high_kw = str(parsed_kw.get("high_level_keywords", ""))

            entities_ctx, relationships_ctx, sources_ctx = self._retrieve_dual_level(state, low_kw, high_kw)

            answer_prompts.append(self._format_chat(
                LIGHTRAG_ANSWER_PROMPT,
                enable_thinking=True,
                entities_context=entities_ctx,
                relationships_context=relationships_ctx,
                sources_context=sources_ctx,
                question=question,
            ))
        # print("checking ")
        # print(f"{answer_prompts[0]}\n")
        # print(f"{answer_prompts[1]}")
        # print("Done with Retrieving")
        print("Doing QA")
        if self.model.pipe is None:
            llm_answers = [f"llm answer {i}" for i in range(len(questions))]
        else:
            llm_answers = self._batch_generate(answer_prompts, self.qa_batch)

        for i, (question, answer) in enumerate(zip(questions, llm_answers)):
            state["context"].qa[i]["llm_answer"] = answer
            # print(f"\n Question {i}: {question}\n LLM Answer: {answer}\n")

            # input_tokens = len(self.model.tokenizer(answer_prompts[i])["input_ids"])
            # output_tokens = len(self.model.tokenizer(answer)["input_ids"])
            # print(f"IT: {input_tokens}, OT: {output_tokens}, TT: {input_tokens + output_tokens}")

        print("Done with dual-level QA.")
        return state

    def llm_judge(self, state: AgentState) -> AgentState:
        """Unchanged from your original pipeline - grades llm_answer against the gold answer."""
        final_answers = []
        formatted_prompts = []
        for i in range(state["context"].num_qa):
            _, true_answer, llm_answer = state["context"].get_qla(i)
            final_answers.append((true_answer, llm_answer))
            formatted_prompts.append(self._format_chat(
                JUDGE_PROMPT, enable_thinking=False, llm_answer=llm_answer, true_answer=true_answer
            ))

        if self.model.pipe is None:
            all_scores = [f" testing {i}" for i in range(len(final_answers))]
        else:
            all_scores = self._batch_generate(formatted_prompts, self.judge_batch)

        # for i, (final_answer, judge_score) in enumerate(zip(final_answers, all_scores)):
        #     print(f"\n{i + 1}: True answer: {final_answer[0]}\n    LLM Answer: {final_answer[1]}\n  Score: {judge_score}\n")

        t = [float(s) for s in all_scores]
        print(f"Total Questions: {len(t)}, sum_scores: {sum(t)}, average: {sum(t) / len(t) * 1.0}")
        return state

    # ------------------------------------------------------------------ #
    # graph wiring
    # ------------------------------------------------------------------ #
    def build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("fetch_conversation", self.fetch_conversation)
        graph.add_node("chunk_document", self.chunk_document)
        graph.add_node("extract_graph_elements", self.extract_graph_elements)
        graph.add_node("build_knowledge_graph", self.build_knowledge_graph)
        graph.add_node("build_vector_indices", self.build_vector_indices)
        graph.add_node("dual_level_qa", self.dual_level_qa)
        graph.add_node("llm_judge", self.llm_judge)

        graph.add_edge(START, "fetch_conversation")
        graph.add_edge("fetch_conversation", "chunk_document")
        graph.add_edge("chunk_document", "extract_graph_elements")
        graph.add_edge("extract_graph_elements", "build_knowledge_graph")
        graph.add_edge("build_knowledge_graph", "build_vector_indices")
        graph.add_edge("build_vector_indices", "dual_level_qa")
        graph.add_edge("dual_level_qa", "llm_judge")
        graph.add_edge("llm_judge", END)

        self.app = graph.compile()

        if self.plot:
            png = self.app.get_graph().draw_mermaid_png()
            os.makedirs("images", exist_ok=True)
            with open("images/lightrag_pipeline.png", "wb") as f:
                f.write(png)
            print("Saved as images/lightrag_pipeline.png")


if __name__ == "__main__":
    model_name = "./local-Qwen3-14B-Instruct"
    model = GetModels(model_name=model_name, load_type=True)

    pipeline = LightRAG(
        model=model,
        entity_batch=64,      # graph extraction prompts are long (schema + chunk), keep smaller than QA batch
        summary_batch=32,
        keyword_batch=16,
        qa_batch=8,
        judge_batch=96,
        plot=False,
        merge_threshold=2,    # re-summarize once an entity/relation is seen in >=2 chunks
        entity_top_k=6,
        relation_top_k=6,
        chunk_top_k=4,
    )
    pipeline.build_graph()
    conv = pipeline.app.invoke({
        "context_idx": 0,
        "context": "",
        "data_path": "files/locomo10.json",
    })