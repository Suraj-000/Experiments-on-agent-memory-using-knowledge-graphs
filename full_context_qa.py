import sys
from typing import Dict,TypedDict,List
from utils import GetConversation,LoadAllData
from all_prompts import QA_PROMPT, JUDGE_PROMPT
from langgraph.graph import StateGraph,START,END
import os 
from models import GetModels
from tqdm.auto import tqdm
import re

class AgentState(TypedDict):
    context_idx: int
    context: GetConversation
    data_path: str

class FullContext:
    def __init__(self,model,qa_batch,judge_batch,plot=False):
        self.model=model
        self.qa_batch=qa_batch
        self.judge_batch=judge_batch
        self.plot=plot
        self.role_map = {"human": "user", "ai": "assistant", "system": "system"}

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

    
    def question_answer(self, state:AgentState)->AgentState:
        """ this node will see the entire conversation and answer the questions asked."""
        print("Running Full Conversation QA...")

        questions = [state["context"].get_qa(i)[0] for i in range(state["context"].num_qa)]
        formatted_prompts=[
            self._format_chat(
                QA_PROMPT,
                enable_thinking=True,
                conversation=state["context"].get_full_conversation(),
                question=q
                )
                for q in questions
        ]

        llm_answers = self._batch_generate(formatted_prompts, self.qa_batch)
        for i, llm_answer in enumerate(llm_answers):
            state["context"].qa[i]["llm_answer"] = llm_answer

        print("Finished Full Context QA.\n" + "-"*50)
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
        graph.add_node("qa",self.question_answer)
        graph.add_node("llm_judge",self.llm_judge)

        graph.add_edge(START,"fetch_conversation")
        graph.add_edge("fetch_conversation","qa")
        graph.add_edge("qa","llm_judge")
        graph.add_edge("llm_judge",END)

        app=graph.compile()

        if self.plot:
            png = app.get_graph().draw_mermaid_png()

            os.makedirs("images",exist_ok=True)
            with open("images/full_context_qa.png", "wb") as f:
                f.write(png)
            print("Saved as full_context_qa.png")
        return app

if __name__=="__main__":

    # model_name="Qwen/Qwen2.5-1.5B-Instruct"
    model_name="./local-Qwen3-14B-Instruct"
    model=GetModels(model_name=model_name,load_type=True)

    graph=FullContext(model=model,qa_batch=8,judge_batch=128,plot=True)
    conv=graph.app.invoke({
        "context_idx":0,
        "context":"",
        "data_path":"files/locomo10.json",
    })
