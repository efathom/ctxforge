"""
Query Rewrite Prompt Template.

LLM prompt for transforming ambiguous queries into explicit,
self-contained queries using conversation context.
"""

QUERY_REWRITE_PROMPT = """# Task Objective
Rewrite a user query to make it self-contained and explicit by resolving
references and ambiguities using the conversation history.

# Workflow
1. Review the **Conversation History** to identify relevant entities and context.
2. Analyze the **Current Query** for:
   - Pronouns (they, it, their, his, her)
   - References (that, those, the same)
   - Implicit context (what about..., and also...)
   - Incomplete information
3. If rewriting is needed:
   - Replace pronouns with specific entities
   - Add necessary background from history
   - Make implicit references explicit
   - Ensure the query is understandable standalone
4. If the query is already clear, keep it unchanged.

# Rules
- Preserve the original intent
- Only use information from the conversation history
- Do not introduce assumptions or external knowledge
- Keep the rewritten query concise but explicit
- If you cannot determine the referent, keep the original

# Output Format
Respond in this exact XML format:

<rewrite_response>
  <analysis>Brief analysis of whether rewriting is needed and why.</analysis>
  <rewritten_query>The rewritten query (or original if no change needed).</rewritten_query>
  <reason>pronoun | reference | implicit | ellipsis | no_change</reason>
  <confidence>0.0 to 1.0 confidence score</confidence>
  <resolved_entities>Comma-separated list of entities resolved (empty if none).</resolved_entities>
</rewrite_response>

# Input

## Conversation History
{conversation_history}

## Session Context
{session_context}

## Current Query
{query}
"""

# Shorter prompt for simple cases
QUERY_REWRITE_SIMPLE_PROMPT = """Rewrite this query to be self-contained using the context.

Context: {context}

Query: {query}

If the query is already clear, return it unchanged.
Return ONLY the rewritten query, nothing else.
"""
