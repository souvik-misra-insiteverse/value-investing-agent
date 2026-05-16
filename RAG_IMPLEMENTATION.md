# Dynamic Few-Shot RAG Implementation

This document details the implementation of the Retrieval-Augmented Generation (RAG) few-shot system within the `value_investing_agent`.

## Objective
The agent relies on a highly structured set of rules to evaluate companies and format reports. While a static zero-shot prompt provides good baseline instructions, LLMs heavily benefit from perfectly formatted past examples (few-shot prompting). 

Rather than hard-coding static examples into the prompt, this RAG implementation dynamically saves "Golden Examples" of past successes and retrieves the most contextually relevant ones during runtime to inject into the LLM's context window.

## How It Works

The architecture is divided into two distinct pathways: the **Write Path** (saving high-quality examples) and the **Read Path** (retrieving examples during report generation).

### 1. The Write Path (Saving Golden Examples)
Located in `src/value_agent/cli.py`.

Every time the agent generates a report, it is graded by the internal **LLM Judge** (`judge_report`).
If the LLM Judge determines the report is structurally flawless and factually correct (Score > 0.97 and `format_valid` == True), the system automatically promotes this run to a "Golden Example".

1. The underlying mathematical JSON data (`context_pack`) and the final markdown text (`report`) are packaged together.
2. The package is inserted into LangGraph's `AsyncPostgresStore` vector database under the namespace `("golden_examples",)`.
3. The store automatically generates vector embeddings for the JSON context using the local HuggingFace embedding model (`all-MiniLM-L6-v2`), making the financial profile semantically searchable.

### 2. The Read Path (Dynamic Retrieval)
Located in `src/value_agent/graph.py` inside the `synthesize` node.

When a user requests a screen for a new ticker (e.g., `MSFT`), the agent does the following *before* generating the report:

1. It compiles the current ticker's financial data and SEC metrics into the standard JSON `context_pack`.
2. It executes an `asearch` query against the `golden_examples` vector store using the *current* `context_pack` as the search query.
3. The vector database returns the top 2 historical "Golden Examples" that had the most mathematically and semantically similar financial footprints.

### 3. The Few-Shot Prompt Construction
Also located in the `synthesize` node.

Once the top 2 relevant past runs are retrieved, they are injected directly into the active prompt payload as conversational turns:

```python
# 1. Base Rules
messages = [SystemMessage(content=system_prompt)]

# 2. Retrieved Past Examples
for ex in few_shot_examples:
    messages.append(HumanMessage(content=f"Example Input Context:\n{ex['context']}"))
    messages.append(SystemMessage(content=f"Example Output Report:\n{ex['report']}"))

# 3. Current Task
messages.append(HumanMessage(content=state["context_pack"]))
```

By framing the retrieved examples as past interaction turns (`HumanMessage` -> `SystemMessage`), the active LLM immediately recognizes the precise structure, tone, and logic required to solve the current task, resulting in near-perfect adherence to formatting rules and score logic without modifying the base instructions.

## Inspecting the Data
To view the stored Golden Examples directly in Postgres, run the following SQL query:

```sql
SELECT 
    prefix AS namespace,
    key AS ticker,
    value->>'report' AS saved_golden_report,
    updated_at
FROM public.store
WHERE prefix::text LIKE '%golden_examples%'
ORDER BY updated_at DESC;
```
