# ================================================
# app.py — Multi-Agent Ticket Management System (CrewAI + Groq + Streamlit)
# ================================================

import os, io, re, json, datetime as dt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, Optional, List, Tuple
from crewai import Agent, Task, Crew, LLM
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# ---------- STREAMLIT CONFIG ----------
st.set_page_config(page_title="Feedback AI — Local Agentic", layout="wide")
st.title("🤖 Feedback AI — Multi Agent System (CrewAI + GROQ + Knowledge Base + Heuristics)")

load_dotenv()
os.makedirs("data", exist_ok=True)

# ---------- PATHS ----------
CONFIG_PATH = "data/config.json"
TICKETS_CSV = "data/generated_tickets.csv"
LOG_CSV = "data/processing_log.csv"
METRICS_CSV = "data/metrics.csv"
KB_CSV = "data/knowledge_base.csv"

# ---------- LLM ----------
crew_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
    use_litellm=True
)

# ---------- SAFE CSV READ/WRITE ----------
def read_csv_safe(path: str) -> pd.DataFrame:
    """Read local CSV safely."""
    if not os.path.exists(path):
        return pd.DataFrame()
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"⚠ Warning reading {path} with {enc}: {e}")
    return pd.DataFrame()

def read_csv_flexible(uploaded_file):
    """Read uploaded CSV safely using byte resets."""
    uploaded_bytes = uploaded_file.read()
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis", "latin1"]

    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(uploaded_bytes), encoding=enc, on_bad_lines="skip")
            if df is not None and not df.empty and len(df.columns) > 0:
                print(f"✅ Successfully read CSV using encoding: {enc}")

                # 🧹 CLEANUP: remove empty or blank rows
                df = df.dropna(how="all")
                df = df.loc[~(df == '').all(axis=1)]
                df = df.reset_index(drop=True)

                return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"⚠ Warning reading uploaded CSV with {enc}: {e}")
            continue

    raise Exception("❌ Unable to read the file — unsupported or corrupt encoding.")

def write_csv_safe(df: pd.DataFrame, path: str):
    """Write CSV safely and force flush even on OneDrive."""
    try:
        temp_path = path + ".tmp"
        df.to_csv(temp_path, index=False, encoding="utf-8-sig")
        os.replace(temp_path, path)  # atomic overwrite
        os.sync() if hasattr(os, "sync") else None
        print(f"✅ CSV successfully written to {os.path.abspath(path)}")
    except Exception as e:
        print(f"❌ Failed to write {path}: {e}")


# ---------- CONFIG ----------
DEFAULT_CONFIG = {
    "classification_threshold": 0.65,
    "retention_days": 90,
    "use_crewai_summary": True
}

def load_config() -> Dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(cfg: Dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

# ---------- KNOWLEDGE BASE ----------
def ensure_kb_exists():
    if not os.path.exists(KB_CSV):
        demo = pd.DataFrame([
            ["login crash", "Bug", "High", "High", "Known login issue on Android."],
            ["payment declined", "Bug", "High", "High", "Intermittent payment failure."],
            ["dark mode", "Feature Request", "", "Medium", "Popular UI request."],
            ["slow performance", "Complaint", "", "Medium", "Performance regression report."],
            ["thank you", "Praise", "", "Low", "Positive feedback."]
        ], columns=["pattern","category","severity","priority","kb_notes"])
        write_csv_safe(demo, KB_CSV)

def load_kb() -> pd.DataFrame:
    ensure_kb_exists()
    kb = read_csv_safe(KB_CSV)
    for col in ["pattern", "category", "severity", "priority", "kb_notes"]:
        if col not in kb.columns:
            kb[col] = ""
    kb["pattern"] = kb["pattern"].astype(str).str.strip().str.lower()

    # 🧹 Remove duplicates and empty patterns
    kb = kb.drop_duplicates(subset=["pattern"], keep="last")
    kb = kb[kb["pattern"].notna() & (kb["pattern"].str.strip() != "")]
    kb = kb.reset_index(drop=True)

    return kb.fillna("")

# ---------- AGENTS ----------
classifier_agent = Agent(
    name="Classifier",
    role="Classify feedback into Bug, Feature Request, Complaint, Praise, or Spam.",
    goal="Return JSON with {category, confidence, severity?/impact?}",
    backstory="An expert model trained to interpret user feedback and categorize it precisely.",
    llm=crew_llm,
)
arbiter_agent = Agent(
    name="Arbiter",
    role="Validate and correct classification JSON.",
    goal="Ensure all required fields are present and consistent across outputs.",
    backstory="A QA validator agent ensuring consistency of classification fields.",
    llm=crew_llm,
)
summarizer_agent = Agent(
    name="Summarizer",
    role="Summarize batches of tickets.",
    goal="Return 3–6 concise markdown bullet points summarizing trends.",
    backstory="A senior analyst agent that summarizes feedback into insights.",
    llm=crew_llm,
)

# ---------- DETECTION ----------
def detect_text_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["review_text","body","feedback","comment","message","text","content"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if any(k in c.lower() for k in ["review","comment","body","message","text","mail","feedback","content"]):
            return c
    for c in df.columns:
        sample = " ".join(df[c].astype(str).head(5))
        if len(sample.split()) > 10:
            return c
    return None

# ---------- CREWAI PIPELINE ----------
def heuristic_classify(text: str) -> Tuple[str, float, Dict]:
    t = text.lower()
    if "crash" in t or "error" in t or "fail" in t:
        return ("Bug", 0.8, {"severity": "High"})
    if "please add" in t or "feature" in t:
        return ("Feature Request", 0.75, {"impact": "Medium"})
    if "love" in t or "great" in t or "thanks" in t:
        return ("Praise", 0.85, {})
    if "slow" in t or "bad" in t or "terrible" in t:
        return ("Complaint", 0.7, {})
    return ("Complaint", 0.55, {})

def escalate_with_crewai(text: str) -> Dict:
    desc = f"""Classify this feedback into one of: Bug, Feature Request, Complaint, Praise, Spam.
If Bug: include severity (High/Medium/Low).
If Feature Request: include impact.
Respond in JSON."""
    t1 = Task(description=desc, agent=classifier_agent, expected_output="JSON with category, confidence, severity/impact")
    t2 = Task(description="Validate classification JSON.", agent=arbiter_agent, expected_output="Clean JSON")
    crew = Crew(agents=[classifier_agent, arbiter_agent], tasks=[t1, t2])
    result = crew.kickoff()
    try:
        m = re.search(r"\{.*\}", str(result), re.S)
        if m: return json.loads(m.group(0))
    except Exception as e:
        print("CrewAI parse error:", e)
    return {}

# ---------- TICKET PIPELINE ----------
def process_feedbacks(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    tcol = detect_text_col(df)
    if not tcol:
        raise ValueError("No text column found.")
    
    canon = pd.DataFrame({
        "source_id": df.index.astype(str),
        "source_type": ["upload"] * len(df),
        "text": df[tcol].astype(str)
    })

    kb = load_kb()
    tickets = []
    for _, r in canon.iterrows():
        text = r["text"]
        cat, conf, extra = heuristic_classify(text)
        sev, imp = extra.get("severity", ""), extra.get("impact", "")
        if conf < cfg["classification_threshold"]:
            escal = escalate_with_crewai(text)
            if escal:
                cat = escal.get("category", cat)
                conf = escal.get("confidence", conf)
                sev = escal.get("severity", sev)
                imp = escal.get("impact", imp)
        tickets.append({
            "ticket_id": f"T-{r['source_id']}-{int(dt.datetime.now().timestamp())}",  # ensure unique ID
            "title": f"[{cat}] #{r['source_id']}",
            "category": cat,
            "priority": "High" if sev == "High" else "Medium",
            "severity": sev,
            "impact": imp,
            "source_id": r["source_id"],
            "text": text,
            "status": "Pending",
            "created_at": dt.date.today().isoformat(),
            "agent": "Hybrid"
        })

    tdf = pd.DataFrame(tickets)

    # 🧩 Append to existing CSV instead of overwriting
    existing = read_csv_safe(TICKETS_CSV)
    if not existing.empty:
        combined = pd.concat([existing, tdf], ignore_index=True)
        combined = combined.drop_duplicates(subset=["text"], keep="last")  # avoid exact duplicate feedbacks
    else:
        combined = tdf

    write_csv_safe(combined, TICKETS_CSV)
    return combined


# ---------- SUMMARY ----------
def crew_summary_for_run(tdf: pd.DataFrame) -> str:
    if tdf.empty: return "No tickets generated."
    sample = tdf[["ticket_id","category","priority","severity","impact","status"]].head(20)
    desc = f"Summarize this ticket batch (3–6 bullets, markdown):\n{sample.to_markdown(index=False)}"
    t = Task(description=desc, agent=summarizer_agent, expected_output="Markdown summary")
    c = Crew(agents=[summarizer_agent], tasks=[t])
    try:
        return str(c.kickoff())
    except Exception:
        return "Summary unavailable."

# ---------- STREAMLIT UI ----------
tabs = st.tabs(["📊 Dashboard","🎫 Tickets","📈 Analytics","⚙️ Config","🧠 Knowledge Base"])

# --- DASHBOARD ---
with tabs[0]:
    st.subheader("Upload & Process Locally")
    up = st.file_uploader("Upload feedback CSV", type=["csv"])
    # Ensure no duplicate runs across reruns
    if "last_uploaded_name" not in st.session_state:
        st.session_state["last_uploaded_name"] = None

    if up:
        df = read_csv_flexible(up)
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Run Agents", key="run_agents"):
            # Prevent re-processing same file on rerun
            if st.session_state["last_uploaded_name"] != up.name:
                st.session_state["last_uploaded_name"] = up.name
                cfg = load_config()
                with st.spinner("Running local hybrid pipeline..."):
                    tdf = process_feedbacks(df, cfg)
                    summary = crew_summary_for_run(tdf)
                st.success(f"✅ {len(tdf)} tickets saved to {TICKETS_CSV}")
                st.markdown("**Run Summary:**")
                st.markdown(summary)
            else:
                st.warning("File already processed — upload a new one to rerun.")
    st.markdown("💡 You can test the system using sample files in the `sample_data/` folder.")

# --- TICKETS ---
with tabs[1]:
    df = read_csv_safe(TICKETS_CSV)
    if not df.empty:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(True)
        gb.configure_selection('multiple', use_checkbox=True)
        gb.configure_default_column(editable=True, filter=True, sortable=True)
        gb.configure_column("status", editable=True, cellEditor='agSelectCellEditor',
                            cellEditorParams={'values':['Pending','Completed']})
        grid = AgGrid(df, gridOptions=gb.build(),
                      update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
                      theme="balham", height=420)
        selected = grid["selected_rows"]
        updated = pd.DataFrame(grid["data"])
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Mark Completed"):
                if selected is not None and len(selected) > 0:
                    # Handle both DataFrame and list cases safely
                    if isinstance(selected, pd.DataFrame):
                        ids = selected["ticket_id"].tolist()
                    else:
                        ids = [r["ticket_id"] for r in selected if isinstance(r, dict) and "ticket_id" in r]

                    updated.loc[updated["ticket_id"].isin(ids), "status"] = "Completed"
                    write_csv_safe(updated, TICKETS_CSV)
                    st.success(f"{len(ids)} tickets marked as completed.")
                    st.rerun()
                else:
                    st.warning("No rows selected.")

        with c2:
            if st.button("💾 Save Changes"):
                write_csv_safe(updated, TICKETS_CSV)
                st.success("Saved.")
    else:
        st.info("No tickets yet.")

# --- ANALYTICS ---
with tabs[2]:
    st.subheader("Analytics & Trends")
    tdf = read_csv_safe(TICKETS_CSV)
    if not tdf.empty:
        tdf["created_at"] = pd.to_datetime(tdf["created_at"], errors="coerce").dt.date
        last7 = tdf[tdf["created_at"] >= (dt.date.today() - dt.timedelta(days=7))]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Ticket Status")
            st.bar_chart(tdf["status"].value_counts())
        with c2:
            st.markdown("### Tickets Processed (Last 7 Days)")
            st.line_chart(last7.groupby("created_at").size())
        with c3:
            st.markdown("### Category Distribution")
            st.bar_chart(tdf["category"].value_counts())
    else:
        st.info("No analytics yet.")

# --- CONFIG ---
with tabs[3]:
    cfg = load_config()
    st.subheader("Configuration")
    cfg["classification_threshold"] = st.slider("Classification Threshold", 0.0, 1.0, cfg["classification_threshold"], 0.01)
    cfg["use_crewai_summary"] = st.checkbox("Use CrewAI for summary", cfg.get("use_crewai_summary", True))
    if st.button("💾 Save Config"):
        save_config(cfg)
        st.success("Saved.")
    if st.button("🗑️ Clear Data"):
        for f in [TICKETS_CSV, LOG_CSV, METRICS_CSV]:
            if os.path.exists(f): os.remove(f)
        st.warning("All old data cleared.")

# --- KNOWLEDGE BASE ---
with tabs[4]:
    st.subheader("Static Knowledge Base")
    kb = load_kb()
    st.data_editor(kb, use_container_width=True, num_rows="dynamic")
    upkb = st.file_uploader("Upload KB CSV", type=["csv"])
    if upkb:
        new = read_csv_flexible(upkb)
        merged = pd.concat([kb, new]).drop_duplicates(subset=["pattern"], keep="last")
        write_csv_safe(merged, KB_CSV)
        write_csv_safe(merged, KB_CSV)
        st.success(f"Knowledge base updated and saved to {os.path.abspath(KB_CSV)}")


