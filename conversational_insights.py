"""
Conversational Insights over BigQuery.
Pulls conversation text from BigQuery, computes sentiment, discovers topics,
summarizes each topic with Claude (Anthropic), and writes everything back to BigQuery.

Setup:
    pip install google-cloud-bigquery transformers bertopic sentence-transformers torch pandas anthropic python-dotenv

Auth (once):
    gcloud auth application-default login

LLM config (set ONE of these):
  Anthropic API:
    setx ANTHROPIC_API_KEY "sk-ant-..."
    (optional) setx ANTHROPIC_MODEL "claude-sonnet-5-5"   # default: claude-haiku-5-5
  Claude on Vertex AI (uses your gcloud ADC credentials, no API key):
    setx ANTHROPIC_VERTEX_PROJECT_ID "<your-gcp-project>"
    setx CLOUD_ML_REGION "us-east5"                        # a region where Claude is enabled
    (optional) setx ANTHROPIC_MODEL "<vertex model id>"
  (restart the terminal after setx)
"""
import os
import time
from google.cloud import bigquery
from transformers import pipeline
from bertopic import BERTopic
from anthropic import Anthropic, AnthropicVertex
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
# ---------------- config — change these ----------------
PROJECT      = "project-819150c4-7f48-40eb-b26"
DATASET      = "analytics"
SOURCE_TABLE = "conversations"
TEXT_COLUMN  = "transcript"
ID_COLUMN    = "conversation_id"
DEST_TABLE   = "conversation_insights"
MAX_ROWS     = 5000
DOCS_PER_TOPIC = 5                 # representative convos sent to the LLM
# -------------------------------------------------------

# ---------- LLM client (Vertex if configured, else Anthropic API) ----------
if os.getenv("ANTHROPIC_VERTEX_PROJECT_ID"):
    llm = AnthropicVertex(
        project_id=os.environ["ANTHROPIC_VERTEX_PROJECT_ID"],
        region=os.getenv("CLOUD_ML_REGION", "us-east5"),
    )
else:
    llm = Anthropic()                                   # reads ANTHROPIC_API_KEY
LLM_MODEL = os.getenv("ANTHROPIC_API_KEY", "claude-haiku-5-5")


def summarize_topic(keywords, examples, retries=3):
    """One-two sentence summary of what a topic cluster is about."""
    sample = "\n\n---\n\n".join(e[:1200] for e in examples)
    system = ("You analyze customer support conversations. Given a topic's "
              "keywords and sample conversations from that topic, write a "
              "1-2 sentence summary (max 40 words) of the common customer "
              "issue or theme. Be specific, no preamble.")
    user = f"Keywords: {', '.join(keywords)}\n\nSample conversations:\n\n{sample}"
    for attempt in range(retries):
        try:
            r = llm.messages.create(
                model=LLM_MODEL,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=100,
            )
            return "".join(b.text for b in r.content if b.type == "text").strip()
        except Exception as e:
            print(f"  LLM call failed (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(2 ** attempt)
    return None


client = bigquery.Client(project=PROJECT)

# 1. Pull conversations from BigQuery
df = client.query(f"""
    SELECT {ID_COLUMN}, {TEXT_COLUMN}
    FROM `{PROJECT}.{DATASET}.{SOURCE_TABLE}`
    WHERE {TEXT_COLUMN} IS NOT NULL AND LENGTH({TEXT_COLUMN}) > 20
    LIMIT {MAX_ROWS}
""").to_dataframe()
print(f"Fetched {len(df)} conversations")
docs = df[TEXT_COLUMN].tolist()

# 2. Sentiment — POSITIVE / NEGATIVE + score
sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
)
results = sentiment(docs, batch_size=32)
df["sentiment_label"] = [r["label"] for r in results]
df["sentiment_score"] = [round(r["score"], 3) for r in results]

# 3. Topic discovery
topic_model = BERTopic(verbose=False)
topics, _ = topic_model.fit_transform(docs)
df["topic_id"] = topics
topic_info = topic_model.get_topic_info()
print(topic_info[["Topic", "Count", "Name"]].head(20).to_string(index=False))

# 4. Summarize each topic with Claude
summaries = {}
topic_ids = [t for t in sorted(set(topics)) if t != -1]   # -1 = outliers
for i, tid in enumerate(topic_ids, 1):
    keywords = [w for w, _ in topic_model.get_topic(tid)[:8]]
    examples = topic_model.get_representative_docs(tid)[:DOCS_PER_TOPIC]
    s = summarize_topic(keywords, examples)
    failed = s is None
    if failed:                                             # LLM failed; fall back
        s = examples[0][:200]
    summaries[tid] = s
    print(f"[{i}/{len(topic_ids)}] topic {tid} {'FELL BACK to raw text' if failed else 'done'}", flush=True)
summaries[-1] = "Uncategorized / outlier conversations"
df["topic_summary"] = df["topic_id"].map(summaries)

# 5. Write enriched results back to BigQuery
dest = f"{PROJECT}.{DATASET}.{DEST_TABLE}"
job = client.load_table_from_dataframe(
    df[[ID_COLUMN, TEXT_COLUMN, "sentiment_label",
        "sentiment_score", "topic_id", "topic_summary"]],
    dest,
    job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
)
job.result()
print(f"Wrote {len(df)} enriched rows to {dest}")

# 6. Console report
print("\n== Sentiment split ==")
print(df["sentiment_label"].value_counts().to_string())
print("\n== Topic summaries ==")
for tid in topic_ids:
    name = topic_info.loc[topic_info.Topic == tid, "Name"].values[0]
    n = (df["topic_id"] == tid).sum()
    print(f"[Topic {tid} | {n} convos | {name}] {summaries[tid]}")