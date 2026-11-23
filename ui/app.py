"""Streamlit UI for the Local Model Eval Harness (talks to the FastAPI app)."""

from __future__ import annotations

import html
import json
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API = os.getenv("AGENT_API_URL", "http://127.0.0.1:8000")
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f2328", "#6b7280"

st.set_page_config(page_title="Local Model Eval", page_icon="🧪", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.2rem;}
.out {font-family: ui-monospace, Consolas, monospace; font-size: 0.8rem; white-space: pre-wrap; border: 1px solid #e6e6e6;
      border-radius: 6px; padding: 8px 10px; background: #fafafa; min-height: 3rem;}
.ok {color: #116329;} .bad {color: #a01c1c;}
</style>""", unsafe_allow_html=True)


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=60, **kw)
    except requests.RequestException as exc:
        st.error(f"API not reachable at {API}: {exc}")
        st.stop()
    if r.status_code >= 400:
        try:
            return {"api_error": r.json().get("detail"), "status": r.status_code}
        except ValueError:
            return {"api_error": r.text, "status": r.status_code}
    return r.json() if r.content else {}


def pct(v):
    return "–" if v is None else f"{v:.0%}"


def num(v, unit=""):
    return "–" if v is None else (f"{v / 1000:.1f} s" if unit == "ms" and v >= 1000 else f"{v:,.0f}{unit and ' ' + unit}")


def short(model: str) -> str:
    return model.split("/", 1)[1]


# ----------------------------------------------------------------------------- sidebar
runtimes = api("GET", "/runtimes")
with st.sidebar:
    st.title("🧪 Local Model Eval")
    st.caption("Accuracy, JSON & tool-call reliability and speed of local models")
    st.markdown("**Runtimes**")
    for rt in runtimes:
        dot = "🟢" if rt["online"] else "⚪"
        st.markdown(f"{dot} **{rt['name']}** · {rt['kind']} · "
                    + (f"{len(rt['models'])} models" if rt["online"] else "offline"))

tab_board, tab_drill, tab_new, tab_suites = st.tabs(["🏆 Leaderboard", "🔍 Drill-down", "▶️ New run", "📚 Suites"])

runs = api("GET", "/runs")
labels = {r["id"]: f"{r['name'] or 'run'} · {len(r['models'])} models · {r['status']} · {r['created_at'][11:16]}"
          for r in runs} if isinstance(runs, list) else {}


def pick_run(key: str) -> str | None:
    if not labels:
        st.info("No runs yet: start one in the ▶️ New run tab.")
        return None
    pending = st.session_state.get("started_run")
    if pending in labels and st.session_state.get(f"{key}_seen") != pending:
        st.session_state[key] = pending          # show a run that was just started (before the widget exists)
        st.session_state[f"{key}_seen"] = pending
    return st.selectbox("Run", list(labels), format_func=labels.get, key=key)


@st.fragment(run_every=1.5)
def watch(run_id: str):
    run = api("GET", f"/runs/{run_id}")
    if run["status"] in ("running", "queued"):
        pr = run["progress"]
        total = pr.get("total") or 1
        st.progress(min(pr.get("done", 0) / total, 1.0),
                    text=f"{pr.get('done', 0)}/{total} examples · {pr.get('stage', 'starting')}"
                         + (f" · **{pr['model']}**" if pr.get("model") else "")
                         + (f" · {pr['suite']}" if pr.get("suite") else ""))
    else:
        st.rerun()


# ----------------------------------------------------------------------------- leaderboard
with tab_board:
    run_id = pick_run("board_run")
    run = api("GET", f"/runs/{run_id}") if run_id else None
    if run and run["status"] in ("running", "queued"):
        watch(run_id)
    elif run and run["status"] == "failed" and not run.get("summary"):
        st.error(run["error"])
    elif run and run.get("summary"):
        s = run["summary"]
        board = s["leaderboard"]
        for m, why in (s.get("unavailable") or {}).items():
            st.warning(f"Skipped {m}: {why}")
        best = board[0]
        fastest = max(board, key=lambda b: b["tokens_per_sec"] or 0)
        jsonish = [b for b in board if b["json_valid_rate"] is not None]
        def tile(col, label, value, note):
            col.markdown(f"<div style='color:{MUTED};font-size:.85rem'>{label}</div>"
                         f"<div style='font-size:1.35rem;font-weight:600;color:{INK};margin:.1rem 0'>{html.escape(value)}</div>"
                         f"<div style='color:{MUTED};font-size:.8rem'>{note}</div>", unsafe_allow_html=True)
        k = st.columns(4)
        tile(k[0], "Most accurate", short(best["model"]), f"{best['accuracy']:.0%} average across {len(s['suites'])} suites")
        tile(k[1], "Fastest", short(fastest["model"]), f"{fastest['tokens_per_sec']:.0f} tokens/s decode")
        if jsonish:
            rel = max(jsonish, key=lambda b: (b["json_valid_rate"], b["accuracy"]))
            tile(k[2], "Most reliable JSON", short(rel["model"]), f"{rel['json_valid_rate']:.0%} valid, schema-conformant")
        tile(k[3], "Examples graded", str(sum(b["n"] for b in board)), f"{sum(b['errors'] for b in board)} errors")

        st.subheader("Leaderboard")
        rows = [{"model": b["model"], "accuracy": round(100 * b["accuracy"]),
                 "pass (95% CI)": f"{b['pass_rate']:.0%} ({b['pass_ci'][0]:.0%}–{b['pass_ci'][1]:.0%})",
                 "JSON ok": pct(b["json_valid_rate"]), "tool calls ok": pct(b["tool_valid_rate"]),
                 "tokens/s": round(b["tokens_per_sec"] or 0, 1), "TTFT p50": num(b["ttft_p50_ms"], "ms"),
                 "latency p50": num(b["latency_p50_ms"], "ms"), "load": num(b["load_ms"], "ms"),
                 "RAM": num((b["resources"] or {}).get("model_mb"), "MB")} for b in board]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "accuracy": st.column_config.ProgressColumn("accuracy", format="%d%%", min_value=0, max_value=100)})

        left, right = st.columns([1.15, 1])
        with left:
            st.subheader("Accuracy by suite")
            cells = pd.DataFrame([{"model": short(m), "suite": su, "accuracy": s["cells"][f"{m}|{su}"]["accuracy"]}
                                  for m in [b["model"] for b in board] for su in s["suites"]
                                  if f"{m}|{su}" in s["cells"]])
            order = [short(b["model"]) for b in board]
            base = alt.Chart(cells).encode(
                x=alt.X("suite:N", title=None, axis=alt.Axis(labelAngle=-25, labelLimit=220, labelOverlap=False,
                                                             labelColor=MUTED)),
                y=alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, labelColor=INK)))
            heat = base.mark_rect(stroke="white", strokeWidth=2, cornerRadius=3).encode(
                color=alt.Color("accuracy:Q", scale=alt.Scale(domain=[0, 1], range=["#eef4fc", "#0b4ea2"]),
                                legend=alt.Legend(format="%", title="accuracy")),
                tooltip=["model", "suite", alt.Tooltip("accuracy:Q", format=".0%")])
            text = base.mark_text(fontSize=12).encode(
                text=alt.Text("accuracy:Q", format=".0%"),
                color=alt.condition("datum.accuracy > 0.6", alt.value("white"), alt.value(INK)))
            st.altair_chart((heat + text).properties(height=46 * len(order) + 40).configure_view(strokeWidth=0),
                            width="stretch")
        with right:
            st.subheader("Quality vs speed")
            pts = pd.DataFrame([{"model": short(b["model"]), "runtime": b["model"].split("/")[0],
                                 "accuracy": b["accuracy"], "tokens/s": b["tokens_per_sec"] or 0} for b in board])
            # put a label on the left when another point sits just to its right, so names don't collide
            pts["left"] = [any(0 < o["tokens/s"] - r["tokens/s"] < 15 and abs(o["accuracy"] - r["accuracy"]) < .12
                               for _, o in pts.iterrows()) for _, r in pts.iterrows()]
            rts = sorted(pts["runtime"].unique())
            color = alt.Color("runtime:N", scale=alt.Scale(domain=rts, range=[BLUE, ORANGE][:len(rts)] or [BLUE]),
                              legend=alt.Legend(orient="bottom", title=None))
            enc = dict(x=alt.X("tokens/s:Q", title="decode speed (tokens/s) →", scale=alt.Scale(zero=True)),
                       y=alt.Y("accuracy:Q", title="accuracy ↑", axis=alt.Axis(format="%"),
                               scale=alt.Scale(domain=[0, 1])))
            dots = alt.Chart(pts).mark_circle(size=140, opacity=1, stroke="white", strokeWidth=2).encode(
                **enc, color=color, tooltip=["model", "runtime", alt.Tooltip("accuracy:Q", format=".0%"), "tokens/s"])
            names = (alt.Chart(pts[~pts["left"]]).mark_text(align="left", dx=9, dy=-8, fontSize=11, color=INK)
                     .encode(**enc, text="model")
                     + alt.Chart(pts[pts["left"]]).mark_text(align="right", dx=-9, dy=-8, fontSize=11, color=INK)
                     .encode(**enc, text="model"))
            st.altair_chart((dots + names).properties(height=300).configure_view(strokeWidth=0)
                            .configure_axis(gridColor="#eee", labelColor=MUTED, titleColor=MUTED),
                            width="stretch")

# ----------------------------------------------------------------------------- drill-down
with tab_drill:
    run_id = pick_run("drill_run")
    run = api("GET", f"/runs/{run_id}") if run_id else None
    if run and run.get("summary"):
        s = run["summary"]
        suite = st.selectbox("Suite", s["suites"], key="drill_suite")
        rows = api("GET", f"/runs/{run_id}/results", params={"suite": suite})
        models = [b["model"] for b in s["leaderboard"] if any(r["model"] == b["model"] for r in rows)]
        by = {(r["model"], r["example_id"]): r for r in rows}
        ids = list(dict.fromkeys(r["example_id"] for r in rows))
        grid = pd.DataFrame([{"example": i, **{short(m): ("✓" if by[(m, i)]["passed"] else "✗") if (m, i) in by else ""
                                              for m in models}} for i in ids])
        st.caption("✓ passed · ✗ failed. Pick an example to compare every model's output side by side.")
        st.dataframe(grid, hide_index=True, width="stretch", height=min(38 * len(ids) + 40, 420))
        ex = st.selectbox("Example", ids, key="drill_ex")
        first = next(r for r in rows if r["example_id"] == ex)
        with st.expander("Prompt and expected answer", expanded=False):
            st.code(first["prompt"], language=None)
            st.markdown(f"**Expected:** `{json.dumps(first['expected'], ensure_ascii=False)}`")
        cols = st.columns(len(models))
        for col, m in zip(cols, models):
            r = by.get((m, ex))
            if not r:
                continue
            with col:
                mark = "<span class='ok'>✓ pass</span>" if r["passed"] else "<span class='bad'>✗ fail</span>"
                st.markdown(f"**{short(m)}** {mark} · score {r['score']:.2f}", unsafe_allow_html=True)
                body = r["output"] or ""
                if r["tool_calls"]:
                    body += ("\n" if body else "") + "\n".join(
                        f"→ {c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in r["tool_calls"])
                st.markdown(f"<div class='out'>{html.escape(body or '(empty)')}</div>", unsafe_allow_html=True)
                st.caption(f"{r['detail']} · {num(r['latency_ms'], 'ms')} · {r['output_tokens']} tokens"
                           + (" (est.)" if r["tokens_estimated"] else ""))

# ----------------------------------------------------------------------------- new run
with tab_new:
    refs = [m["ref"] for rt in runtimes if rt["online"] for m in rt["models"]]
    suites = api("GET", "/suites")
    if not refs:
        st.warning("No runtime is online. Start Ollama (`ollama serve`) or an OpenAI-compatible server, "
                   "and list it in runtimes.yaml.")
    with st.form("new"):
        chosen = st.multiselect("Models", refs, default=refs[:4])
        picked = st.multiselect("Suites", [s["name"] for s in suites], default=[s["name"] for s in suites])
        c = st.columns(3)
        limit = c[0].number_input("Examples per suite (0 = all)", 0, 1000, 0)
        json_mode = c[1].selectbox("JSON mode", ["per suite", "force on", "force off"])
        name = c[2].text_input("Run name", "")
        if st.form_submit_button("Start evaluation", type="primary", disabled=not refs):
            res = api("POST", "/runs", json={"models": chosen, "suites": picked or None, "limit": limit or None,
                                             "json_mode": {"per suite": None, "force on": True, "force off": False}[json_mode],
                                             "name": name or None})
            if "api_error" in res:
                st.error(res["api_error"])
            else:
                st.session_state["started_run"] = res["id"]
                st.success("Started: follow it in the 🏆 Leaderboard tab.")
                st.rerun()

# ----------------------------------------------------------------------------- suites
with tab_suites:
    for s in api("GET", "/suites"):
        with st.expander(f"**{s['name']}** · {s['category']} · {s['grader']} grader · {s['examples']} examples"):
            st.write(s["description"])
            detail = api("GET", f"/suites/{s['name']}")
            st.code(detail["template"], language=None)
            if s["tools"]:
                st.caption("Tools: " + ", ".join(s["tools"]))
            st.dataframe(pd.DataFrame(detail["examples"][:5]).astype(str), hide_index=True, width="stretch")
