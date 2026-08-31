import sys
import json
import re
import os 

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

class LoadAllData:
    def __init__(self,data_path):
        self.data=read_json_file(data_path)
        self.num_samples=len(self.data)

    def get_sample_conversation(self,idx):
        if idx>self.num_samples-1:
            print(f"sample index out of range, nof samples = {self.num_samples} \n")
            return None
        return self.data[idx]
    
class GetConversation:
    def __init__(self,sample):
        self.qa=sample["qa"]
        self.conversation=sample["conversation"]
        self.num_qa=len(self.qa)
        self.session_ids=sorted(
            int(m.group(1)) for k in self.conversation.keys() if (m :=re.fullmatch(r"session_(\d+)",k))
        )
        self.num_sessions=len(self.session_ids)
        self.full_conversation,self.all_sessions=self.save_all_sessions()
        self.speakers=[]
        self.speakers.append(self.conversation["speaker_a"])
        self.speakers.append(self.conversation["speaker_b"])

    def save_all_sessions(self):
        all_sessions=[]
        for i in self.session_ids:
            session=[]
            s=self.conversation[f"session_{i}"]
            date_key=f"session_{i}_date_time"
            date_val=self.conversation.get(date_key, "date unknown")
            session.append(f"===== Session {i} : {date_val} =====")
            for ses in s:
                sentence=f"{ses["speaker"]}: {ses["text"]}"
                session.append(sentence)
            all_sessions.append(session)

        output=[]
        for session in all_sessions:
            output.append("\n".join(session))
            output.append("")

        full_context="\n".join(output)
        # with open("data/conversations.txt","w",encoding="utf-8") as f:
            # f.write(full_context)
        return full_context, all_sessions

    def get_qa(self,idx):
        if idx>self.num_qa-1:
            print(f"sample index out of range, nof samples = {self.num_qa} \n")
            return None
        question=self.qa[idx]["question"]
        answer = self.qa[idx].get("answer") or self.qa[idx].get("adversarial_answer")
        # evidence=self.qa[idx]["evidence"]
        # category=self.qa[idx]["category"]
        return question,answer

    def get_qla(self,idx):
        if idx>self.num_qa-1:
            print(f"sample index out of range, nof samples = {self.num_qa} \n")
            return None
        question=self.qa[idx]["question"]
        answer = self.qa[idx].get("answer") or self.qa[idx].get("adversarial_answer")
        llm_answer = self.qa[idx].get("llm_answer") 
        # evidence=self.qa[idx]["evidence"]
        # category=self.qa[idx]["category"]
        return question,answer,llm_answer
    
    def get_session(self,idx):
        if idx>self.num_sessions:
            print(f"session index out of range, nof sessions = {len(self.session_ids)} one indexed \n")
            return None
        return "\n".join(self.all_sessions[idx])
        
    def get_full_conversation(self):
        return self.full_conversation
    
if __name__=="__main__":
    # speakers, full_context,all_sessions,qa=agent_format_conversation("files/locomo10.json","files")
    d=LoadAllData("files/locomo10.json")
    sample=d.get_sample_conversation(0)
    k=GetConversation(sample)
    q,a=k.get_qa(198)
    print(q,a)
    session=k.get_session(1)
    # print(session)
    conv = k.get_full_conversation()
    print(conv)
    print(type(conv))