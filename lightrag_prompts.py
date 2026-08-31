"""
Prompts for the LightRAG-style pipeline (lightrag_pipeline.py).

Mirrors the structure of your existing all_prompts.py (ChatPromptTemplate,
system+human messages) but adds the prompts LightRAG needs on top of plain
entity extraction:

  1. GRAPH_EXTRACTION_PROMPT       - entities + relationships + chunk-level
                                      high-level keywords, in one pass.
  2. ENTITY_MERGE_SUMMARY_PROMPT   - collapse N descriptions of the same
                                      entity (seen across chunks) into one.
  3. RELATION_MERGE_SUMMARY_PROMPT - same, for a relationship/edge.
  4. KEYWORD_EXTRACTION_PROMPT     - splits a user question into low-level
                                      (specific entities) vs high-level
                                      (themes) keywords, per the paper's
                                      dual-level retrieval design.
  5. LIGHTRAG_ANSWER_PROMPT        - final answer generation grounded in
                                      entities + relationships + raw chunks.
"""

from langchain_core.prompts import ChatPromptTemplate

GRAPH_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert at extracting structured knowledge graphs from text. "
     "Given a chunk of conversation, identify all entities and the relationships between them.\n\n"
     "Return ONLY valid JSON (no markdown fences, no preamble) with this exact schema:\n"
     "{{\n"
     '  "entities": [\n'
     '    {{"name": "<entity name>", "type": "<PERSON|ORG|LOCATION|EVENT|CONCEPT|MISC>", '
     '"description": "<1-2 sentence description grounded in the text>"}}\n'
     "  ],\n"
     '  "relationships": [\n'
     '    {{"source": "<entity name>", "target": "<entity name>", '
     '"description": "<how they relate, grounded in the text>", '
     '"keywords": "<comma-separated high-level keywords summarizing the relationship type>", '
     '"strength": <integer 1-10, how central this relationship is>}}\n'
     "  ],\n"
     '  "content_keywords": "<comma-separated high-level themes/topics of this whole chunk>"\n'
     "}}\n\n"
     "Rules:\n"
     "- Entity names must use the exact same surface form across the entities and relationships lists.\n"
     "- Only extract relationships between entities that also appear in 'entities'.\n"
     "- If nothing is found, return empty lists but still return valid JSON.\n"
     "- Do not invent facts that are not supported by the text."
     ),
    ("human", "Text chunk:\n{context}\n\nExtract the knowledge graph as JSON.")
])

ENTITY_MERGE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You merge multiple descriptions of the same entity, collected from different parts of a "
     "conversation, into a single coherent description of at most 5 sentences. Preserve every "
     "distinct fact; remove redundancy. Return ONLY the merged description text - no preamble, "
     "no quotes, no bullet points."
     ),
    ("human", "Entity: {entity_name} ({entity_type})\n\nDescriptions to merge:\n{descriptions}")
])

RELATION_MERGE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You merge multiple descriptions of the same relationship between two entities, collected "
     "from different parts of a conversation, into a single coherent description of at most 2 "
     "sentences. Return ONLY the merged description text - no preamble, no quotes."
     ),
    ("human", "Relationship: {source} <-> {target}\n\nDescriptions to merge:\n{descriptions}")
])

KEYWORD_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract search keywords from a user question for a dual-level retrieval system.\n"
     "Return ONLY valid JSON (no markdown fences, no preamble) with this exact schema:\n"
     "{{\n"
     '  "low_level_keywords": "<comma-separated specific entities, names, or concrete details '
     'mentioned or implied in the question>",\n'
     '  "high_level_keywords": "<comma-separated broader themes, topics, or concepts the '
     'question is really about>"\n'
     "}}"
     ),
    ("human", "Question: {question}")
])

LIGHTRAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a question-answering assistant. Answer the question using ONLY the knowledge graph "
     "context and source excerpts provided below. Be concise and answer directly. If the context "
     "does not contain the answer, say you don't know rather than guessing.\n\n"
     "--- Relevant Entities ---\n{entities_context}\n\n"
     "--- Relevant Relationships ---\n{relationships_context}\n\n"
     "--- Source Excerpts ---\n{sources_context}"
     ),
    ("human", "Question: {question}")
])