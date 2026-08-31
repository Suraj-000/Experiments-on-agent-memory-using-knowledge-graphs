from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator


SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
SYSTEM ROLE: You are an automated conversation archivist. Your job is to compress a
session conversation into a dense, fact-preserving record for a downstream
question-answering system. You are not writing prose for a human reader; you are
building a lossless-as-possible fact store.

CONVERSATION:
{session}

OUTPUT: One paragraph for this session starting with
"Session <N> (<date/time as stated>):".

RULES:
1. Never invent, infer, or resolve information that is not explicitly stated. If
   something is ambiguous, describe it as stated instead of resolving it yourself.
2. Always use explicit names instead of pronouns when the conversation lets you
   resolve who is meant. Write "(speaker unclear)" if you cannot.
3. Preserve exactly as stated, never rounded/paraphrased/generalized:
   - names of people, organizations, places
   - dates, times, and relative expressions ("next Friday" stays "next Friday")
   - locations and addresses
   - decisions and plans, including who decided and why
   - stated relationships between people
   - opinions/preferences, attributed to the specific speaker
   - medical info (conditions, symptoms, medications, appointments)
   - financial info (amounts, currencies, transactions, debts)
   - any other numbers (ages, quantities, durations, measurements)
4. Attribute facts to whoever said them.
5. If a plan or fact changes later in the conversation, keep both versions and
   state which one is more recent/final.
6. Drop greetings, filler, and repeated small talk that carries no fact.
7. Do not compress specific details into vague generalities. Write "they agreed
   to meet at Cafe Luna at 6 PM on Friday", not "they planned to meet up".

Return only the session paragraph. No headers, no preamble, no explanation.
""")


QA_PROMPT = ChatPromptTemplate.from_template("""
SYSTEM ROLE: You are a precision QA engine operating strictly on the memory
context below. You answer only from what is explicitly present.

CONTEXT:
{conversation}

QUESTIONS:
Each line is: Category: (category), Q: (question)
{question}

PER-CATEGORY INSTRUCTIONS (apply based on each question's stated Category):
- temporal: Resolve relative dates ("yesterday", "next Friday") using session
  dates/timestamps in the context. State the resolved date/duration explicitly.
- multi-hop / relationship: Connect facts that appear in different, possibly
  distant, parts of the context before answering. For "both/all" comparison
  questions, state the shared fact rather than listing each person's details.
- single-hop / factual: Return only the exact requested fact (name, number,
  place, method). No extra explanation.
- open-domain / adversarial: Base the answer only on what is stated. If the
  premise of the question is not supported by the context, or the fact is
  absent, that is a signal to answer "na" rather than guess.
- financial / numerical / medical: Copy numbers, units, and terms exactly as
  stated. Do not round or convert.

GENERAL RULES:
1. Use only information explicitly present in the context. Never use outside
   knowledge and never guess.
2. Resolve pronouns via context before answering ("where does she live" ->
   identify who "she" is first, then answer with that person's stated location).
3. If multiple versions of a fact exist (e.g. a plan that changed), use the
   most recent/final one, unless the question explicitly asks about an earlier
   point in time.
4. If the context does not contain enough information to answer, the answer
   must be exactly: na
5. Keep answers short: a phrase or short sentence. No reasoning, no
   justification, no repeating the question.


OUTPUT FORMAT (strict String, nothing else — no markdown fences, no commentary):
   (the answer)

""")



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


JUDGE_PROMPT = ChatPromptTemplate.from_template("""
You are an expert evaluator.

Compare the AI answer with the gold answer based on semantic meaning.

Scoring:
1.0 = exact / fully correct
0.8 = substantially correct
0.6 = partially correct
0.4 = related but misses the core answer
0.2 = mostly wrong
0.0 = contradictory, hallucinated, unrelated, or incorrectly says unknown

Rules:
- Judge meaning, not wording.
- Accept synonyms and different formatting.
- Accept relative dates/times when they resolve to the same point.
- Gold unknown + AI unknown = 1.0.
- Gold unknown + hallucinated AI answer = 0.0.
- Ignore omitted evidence.

AI ANSWER:
{llm_answer}

GOLD ANSWER:
{true_answer}

Return ONLY valid JSON.
No explanation.
No markdown.
No additional fields.

{format_instructions}
""").partial(
    format_instructions=judge_parser.get_format_instructions()
)




RAG_PROMPT = ChatPromptTemplate.from_template("""
SYSTEM ROLE: You are an advanced Retrieval-Augmented Generation (RAG) synthesis engine. 
Your objective is to extract and synthesize precise answers based strictly on the provided 
retrieved context blocks. You do not possess external knowledge.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

EXTRACTION & SYNTHESIS DIRECTIVES:
1. STRICT GROUNDING: Base your answer EXCLUSIVELY on the text provided in the RETRIEVED CONTEXT. 
   Do not incorporate outside knowledge, logical leaps, or assumptions. 
2. MULTI-CHUNK SYNTHESIS: The context may be fragmented across different blocks. Connect related 
   pieces of information across the chunks to form a complete answer.
3. CONFLICT RESOLUTION: If context blocks contain conflicting information (e.g., evolving facts 
   or differing accounts), explicitly state the discrepancy or rely on the most definitive/recent 
   evidence provided.
4. MISSING INFORMATION: If the provided context does not contain sufficient information to answer 
   the question, you must not guess or deduce. Output exactly and only: INSUFFICIENT_CONTEXT
5. NO FILLER: Never use phrases like "Based on the provided context...", "According to the transcript...", 
   or "Here is the answer." Start immediately with the facts.

OUTPUT FORMAT:
Provide the synthesized answer concisely as plain text. Do not use conversational preambles, markdown 
formatting, or reasoning traces.
""")


EXTRACT_ENTITIES_PROMPT = ChatPromptTemplate.from_template("""
SYSTEM ROLE: You are an entity and topic extraction engine specialized in analyzing 
multi-turn dialogue transcripts for a retrieval-augmented generation (RAG) system. 
Your extractions are used to index conversation chunks so they can be retrieved later 
when a user asks a question about them. Extract both named entities AND salient topics 
so the chunk is retrievable even when the query doesn't mention a proper noun.

CONTEXT:
{context}

ENTITY & TOPIC TAXONOMY:
- PERSON: named individuals (not pronouns or generic references like "the user")
- ORG: companies, institutions, teams, named groups (e.g. "LGBTQ support group")
- LOCATION: cities, countries, landmarks, addresses, named venues
- DATE: absolute or relative calendar references (e.g. "March 3rd", "last Tuesday")
- TIME: clock times or time-of-day references (e.g. "9am", "this evening")
- EVENT: named or clearly identifiable occurrences (e.g. "the wedding", "Q3 review")
- TOPIC: the subject matter being discussed, even without a proper noun (e.g. "mental 
  health", "gender identity", "career change", "counseling"). Use short noun phrases 
  (2-4 words) that someone might search for later.
- ACTIVITY: an action, plan, or intention someone expresses (e.g. "considering a 
  counseling career", "painting a lake sunrise", "attending a support group meeting"). 
  Keep it to a short phrase, not a full sentence.
- MISC: other proper nouns not covered above (products, titles, named objects)

EXTRACTION DIRECTIVES:
1. STRICT GROUNDING: Extract only entities/topics explicitly present or directly stated 
   in the CONTEXT. Do not infer information that is implied but not stated.
2. DEDUPLICATION: If something is mentioned multiple times (including by alias, nickname, 
   or rephrasing), list it once using its clearest canonical form.
3. NO PRONOUNS, NO BARE PRONOUNS AS SUBJECTS: Never extract pronouns or unnamed role 
   references as PERSON/ORG/LOCATION entities (e.g. "she", "my manager"). This restriction 
   does NOT apply to TOPIC or ACTIVITY — those may reference generic concepts as long as 
   they are specific enough to be useful for retrieval (avoid vague single words like 
   "conversation" or "talk").
4. AMBIGUITY: If a term could fit multiple categories, choose the most useful one for 
   retrieval rather than omitting it.
5. NO CONTENT: If nothing qualifies, output an empty JSON array: []
6. LIMIT: Extract at most 8 TOPIC/ACTIVITY items per chunk — prioritize the most 
   distinctive, query-worthy ones over exhaustive coverage.

OUTPUT FORMAT:
Return ONLY valid JSON — a list of objects, each with "entity" and "type" keys. 
No markdown code fences, no commentary, no preamble.

Example:
[{{"entity": "Caroline", "type": "PERSON"}}, {{"entity": "LGBTQ support group", "type": "ORG"}}, 
{{"entity": "gender identity and acceptance", "type": "TOPIC"}}, 
{{"entity": "considering a counseling career", "type": "ACTIVITY"}}]

EXTRACTED ENTITIES:
""")