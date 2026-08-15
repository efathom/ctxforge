"""
Sufficiency Judge Prompt Template.

LLM prompt for judging whether retrieved content is sufficient
to answer a user's query.
"""

SUFFICIENCY_JUDGE_PROMPT = """# Task Objective
Judge whether the retrieved content is sufficient to answer the user's query.

# Workflow
1. Analyze the **Query** to understand what the user is asking.
2. Review the **Retrieved Content** carefully.
3. Evaluate against these criteria:
   - Does it directly address the user's question?
   - Is the information specific and detailed enough?
   - Are there obvious gaps or missing details?
   - Would additional context significantly improve the answer?
4. Decide: ENOUGH (can answer), MORE (need additional retrieval), or UNCERTAIN.

# Rules
- Base judgment ONLY on provided query and content
- Do not assume or add external knowledge
- Be conservative: if uncertain, say UNCERTAIN
- Consider partial answers as needing MORE
- If content is empty or irrelevant, always say MORE

# Output Format
Respond in this exact XML format:

<sufficiency_response>
  <consideration>Explain your reasoning for the judgment.</consideration>
  <verdict>ENOUGH | MORE | UNCERTAIN</verdict>
  <confidence>0.0 to 1.0 confidence score</confidence>
  <missing_aspects>Comma-separated list of what's missing (empty if ENOUGH).</missing_aspects>
  <suggested_sources>Comma-separated sources to try: memories, graph, expertise.</suggested_sources>
</sufficiency_response>

# Input

## Query
{query}

## Retrieved Content
{content}

## Additional Context
{context}
"""

# Shorter prompt for quick judgments
SUFFICIENCY_QUICK_PROMPT = """Is this content sufficient to answer the query?

Query: {query}

Content: {content}

Reply with only: ENOUGH, MORE, or UNCERTAIN
"""
