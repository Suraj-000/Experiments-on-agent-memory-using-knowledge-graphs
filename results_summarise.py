import os 
from models import GetModels
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


path="/home/scai/msr/aiy257586/scratch/aiy257586/graph/logs/testing"

results=""
for i in os.listdir(path):
    with open(os.path.join(path,i), "r") as f:
        content = f.read()
    results+=content

summarise_results_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an ML experiment results analyzer.

Analyze the experiment logs provided by the user.

Your final answer MUST be a single valid JSON object.

Do not output Markdown.
Do not output ```json.
Do not output explanations.
Do not output analysis after the final JSON.

Required structure:

{
  "experiments": [
    {
      "Experiment_name": "string",
      "accuracy": null,
      "execution_time": null
    }
  ]
}

Rules:

- Include every distinct experiment/model found.
- Extract only values explicitly reported in the logs.
- Use null when a value is missing.
- Do not estimate missing values.
- Do not invent values.
- Avoid duplicate experiments.
- Use the final reported accuracy.
- Use the final reported execution time.
- Return the JSON object completely.
"""
    ),
    ("human", "{results}")
])




# model_name = "./local-Qwen3-14B-Instruct"
model_name2="./DeepSeek-R1-Distill-Qwen-14B"
model = GetModels(model_name=model_name2, load_type=True)

chain=summarise_results_prompt | model.llm | JsonOutputParser()
final_summary=chain.invoke({
    "results":results
})

print(final_summary)
print(type(final_summary))
