

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class GetModels:
    def __init__(self,model_name,load_type=True):
        self.load_type=load_type
        self.llm = self.get_transformer(model_name)

    def get_transformer(self,model_name):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            fix_mistral_regex=True,
            local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side="left"

        if self.load_type==True:
            print("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True
            )
            model.config.pad_token_id = self.tokenizer.pad_token_id
            print("Creating pipeline...")
            self.pipe=pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                max_new_tokens=2048,
                max_length=None,
                pad_token_id=self.tokenizer.pad_token_id,
                clean_up_tokenization_spaces=False,
                return_full_text=False,
                # batch_size=24
                )

            print("Before HFPipeline")
            chat_model = HuggingFacePipeline(pipeline=self.pipe)
            llm = ChatHuggingFace(llm=chat_model,tokenizer=self.tokenizer)
            print("After HFPipeline")

            print(f"\n === Using model {model_name} === \n")

            print("*"*100)
            return llm
        self.pipe=None
        return None



if __name__=="__main__":
    all_models=["./local-Qwen3-14B-Instruct"]

    model=GetModels(model_name=all_models[0])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful math assistant. Keep your answers brief."),
        ("human", "What is {math_problem}?")
    ])
    chain=prompt | model.llm
    response=chain.invoke({"math_problem": "what is 2 + 2?"})
    print(response)



