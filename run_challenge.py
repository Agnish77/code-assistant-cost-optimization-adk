import os, warnings, json, time
warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"
import logging
logging.getLogger("google_adk").setLevel(logging.CRITICAL)

PROJECT_ID = "qwiklabs-gcp-01-1e21e1d85ea3"
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
LOCATION = "global"
EVAL_LOCATION = "us-central1"

os.environ.update(GOOGLE_GENAI_USE_VERTEXAI="TRUE",
                  GOOGLE_CLOUD_PROJECT=PROJECT, GOOGLE_CLOUD_LOCATION=LOCATION)

MODELS = {"coordinator": "gemini-3.1-pro-preview", "docs": "gemini-3.5-flash",
          "community": "gemini-3.5-flash", "embedding": "gemini-embedding-001"}
THINKING = "HIGH"

from code_assistant.agents import build_code_assistant, answer_query
from code_assistant.retrieval import build_index, embed_texts
from code_assistant.ledger import UsageLedger
from code_assistant.data import load_jsonl
from code_assistant.quality import run_quality_eval
from code_assistant import vais
from google.cloud import storage

PRICING = {
    "gemini-3.1-pro-preview": {"input": 2.00, "cached": 0.20, "output": 12.00},
    "gemini-3.5-flash":       {"input": 1.50, "cached": 0.15, "output":  9.00},
    "gemini-3.1-flash-lite":  {"input": 0.25, "cached": 0.025, "output": 1.50},
    "gemini-2.5-pro":         {"input": 1.25, "cached": 0.13, "output": 10.00},
    "gemini-2.5-flash":       {"input": 0.30, "cached": 0.03, "output":  2.50},
    "embedding":  {"per_1m_tokens": 0.15},
    "vais_query": {"per_1k_queries": 1.50},
}

def cost_of(record):
    ct = record.get("call_type")
    if ct == "model":
        p = PRICING[record["model"]]
        out = record.get("output_tokens", 0) + record.get("thought_tokens", 0)
        return round((record.get("input_tokens", 0) * p["input"]
                      + record.get("cached_tokens", 0) * p["cached"]
                      + out * p["output"]) / 1e6, 12)
    if ct == "embedding":
        return round(record.get("embed_tokens", 0) * PRICING["embedding"]["per_1m_tokens"] / 1e6, 12)
    if ct == "retrieval":
        return round(record.get("retrieval_queries", 0) * PRICING["vais_query"]["per_1k_queries"] / 1e3, 12)
    return 0.0

def measure(coordinator, items, qid_holder, ledger, project, location, label):
    ledger.reset()
    answers = []
    for item in items:
        for attempt in (1, 2):
            try:
                ans = answer_query(coordinator, item["query"], str(item["id"]), qid_holder)
                if not ans.strip():
                    raise RuntimeError("agent returned an empty answer")
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"query {item['id']} failed twice ({e}); re-run this cell") from e
                print(f"[run warning] query {item['id']}: {str(e).splitlines()[0]}; retrying once")
                time.sleep(5)
        print(f"Completed query {item['id']}", flush=True)
        answers.append({"query": item["query"], "answer": ans, "reference": item.get("reference", "")})
    quality = run_quality_eval(answers, project, location)
    return {"configuration": label, "cost_per_query": ledger.cost_per_query(),
            "quality": quality, "breakdown": ledger.cost_breakdown()}, answers

VAIS_LOCATION, DS_ID, ENGINE_ID = "global", "code_assistant-docs", "code_assistant-docs-engine"
DOCS_SC = vais.ensure(PROJECT, VAIS_LOCATION, DS_ID, ENGINE_ID, load_jsonl("assets/docs.jsonl"))
so      = load_jsonl("assets/stackoverflow.jsonl")
holdout = load_jsonl("assets/eval_holdout.jsonl")
community = build_index(so, MODELS["embedding"], PROJECT, LOCATION, name="community")
qid = {"v": "_"}

def build(models, thinking_level=THINKING, max_output_tokens=None, top_k=3):
    ledger = UsageLedger(cost_of)
    coord = build_code_assistant(models=models, ledger=ledger, qid_holder=qid,
                            project=PROJECT, location=LOCATION,
                            collections={"community": community}, docs_serving_config=DOCS_SC,
                            top_k=top_k, max_output_tokens=max_output_tokens,
                            thinking_level=thinking_level)
    return coord, ledger

BUDGET = 0.0135
QUALITY_FLOOR = 0.90

def check_challenge_pass(cost_per_query, quality, *, budget, quality_floor):
    reasons = []
    if cost_per_query > budget:
        reasons.append("cost/query %.6f exceeds budget %.6f" % (cost_per_query, budget))
    if quality < quality_floor:
        reasons.append("quality %.3f below floor %.3f" % (quality, quality_floor))
    return {"passed": not reasons, "cost_per_query": cost_per_query, "quality": quality,
            "budget": budget, "quality_floor": quality_floor, "reasons": reasons}

MY_MODELS = {"coordinator": "gemini-3.5-flash",
             "docs":        "gemini-3.1-flash-lite",
             "community":   "gemini-3.1-flash-lite",
             "embedding":   "gemini-embedding-001"}
MY_THINKING = "LOW"
MY_TOPK     = 2
MY_MAXOUT   = None

print("Running final challenge evaluation...", flush=True)
coord, ledger = build(MY_MODELS, thinking_level=MY_THINKING, max_output_tokens=MY_MAXOUT, top_k=MY_TOPK)
cap, _ = measure(coordinator=coord, items=holdout, qid_holder=qid, ledger=ledger,
                 project=PROJECT, location=EVAL_LOCATION, label="final challenge submission")
res = check_challenge_pass(cap["cost_per_query"], cap["quality"], budget=BUDGET, quality_floor=QUALITY_FLOOR)
print("Result:", json.dumps(res, indent=2), flush=True)
print("Breakdown:", ledger.to_df().groupby("agent")[["input_tokens", "thought_tokens", "output_tokens", "cost_usd"]].sum(), flush=True)

if res["passed"]:
    RESULT_BUCKET = f"{PROJECT_ID}-metrics-result"
    results = {
        "cost_per_query": cap['cost_per_query'],
        "quality": cap['quality'],
        "budget": BUDGET,
        "quality_floor": QUALITY_FLOOR,
        "passed": res['passed']
    }
    with open("/home/jupyter/final_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(RESULT_BUCKET)
    blob = bucket.blob("results/final_metrics.json")
    blob.upload_from_filename("/home/jupyter/final_metrics.json")
    print(f"Uploaded to gs://{RESULT_BUCKET}/results/final_metrics.json", flush=True)
