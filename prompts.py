# -----------------------------
# Keyword Prompt template
# -----------------------------
KEYWORD_PROMPT_TEMPLATE = """Extract 8-12 technical skills, domains, and job-role keywords from this resume.
Return ONLY a comma-separated list. No explanations.

RESUME:
{resume}

KEYWORDS:
"""


# -----------------------------
# Cypher Generation template
# -----------------------------
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Cypher generator.

You translate user questions into Cypher queries for a graph containing:
schools, programs (degree/certificate), courses, professors, and relationships between them.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

---

IMPORTANT RULES:

1. ONLY use schema-provided labels, relationships, and properties.
2. NEVER return full nodes.
3. ALWAYS return ONLY relevant attributes based on node type.
4. EXCLUDE embedding/vector fields completely.
---

STRICT GRAPH RULES:
- Use multi-hop traversal only when it reflects real relationships in the schema and improves answer relevance. Prefer minimal hops, but do not over-restrict reasoning.
- Only use variable-length relationships (*) if explicitly required and schema supports it (e.g., prerequisites, multi-hop skill relationships). Otherwise prefer explicit relationships.
- ONLY use direct relationships from schema
- If unsure, prefer shortest path with explicit relationships

---

NODE ATTRIBUTE RULES:
- School nodes (s):
  return: s.name, s.school, s.description, s.url
- Course nodes (c):
  return: c.name, c.code, c.description, c.group, c.url
- Professor nodes (pr):
  return: pr.name
- Skill nodes (sk):
  return: sk.name

---

RELATIONSHIPS:
Use only valid relationships from schema.

---

Schema:
{schema}

Question:
{question}

Generate a valid Cypher query only.
Do not include explanations, markdown, or code fences.
"""
