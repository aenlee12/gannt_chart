import io, re
from datetime import timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Gantt", layout="wide")

# ----------------- utils -----------------
def find_col(cols_map, subs):
    for orig, low in cols_map.items():
        for s in subs:
            if s in low:
                return orig
    return None

def coerce_int(x):
    try:
        return int(str(x).strip())
    except:
        return np.nan

def parse_bool01(x):
    if pd.isna(x): return 0
    s = str(x).strip().lower()
    if s in {"1","да","true","истина","y","yes"}: return 1
    if s in {"0","нет","false","ложь","n","no",""}: return 0
    try:
        return 1 if float(s) != 0 else 0
    except:
        return 0

def parse_preds(x):
    if pd.isna(x): return []
    s = str(x)
    parts = re.split(r"[;,/]+", s)
    ids = []
    for p in parts:
        p = p.strip()
        if not p: continue
        if p.isdigit():
            ids.append(int(p))
        else:
            ids += [int(n) for n in re.findall(r"\d+", p)]
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            out.append(i); seen.add(i)
    return out

def norm_status(x):
    if pd.isna(x): return ""
    s = str(x).strip().lower()
    if "нов" in s: return "новый"
    if "рабоч" in s: return "в работе"
    if "сделан" in s: return "сделано"
    if "заплан" in s: return "запланировано"
    return s

def shorten(s, n=70):
    s = str(s).strip()
    return s if len(s) <= n else s[:n-1] + "…"

# ----------------- core -----------------
def build_df(src: pd.DataFrame):
    cols = {c: str(c).strip().lower() for c in src.columns}

    col_id    = find_col(cols, ["id"])
    col_name  = find_col(cols, ["назван", "задач", "task"])
    col_s     = find_col(cols, ["дата от","начал","start","старт"])
    col_e     = find_col(cols, ["дата до","окон","end","finish","финиш"])
    col_dep   = find_col(cols, ["предшествен"])
    col_stat  = find_col(cols, ["состояние","статус","status"])
    col_head  = find_col(cols, ["головн"])
    col_sub   = find_col(cols, ["подголовн","subhead"])

    required = [col_id, col_name, col_s, col_e]
    if any(c is None for c in required):
        raise ValueError("Нужны колонки: ID, Название, Дата От, Дата До. Остальное опционально.")

    df = src[[c for c in [col_id,col_name,col_s,col_e,col_dep,col_stat,col_head,col_sub] if c is not None]].copy()
    df["Start"] = pd.to_datetime(df[col_s], errors="coerce", dayfirst=True)
    df["End"]   = pd.to_datetime(df[col_e], errors="coerce", dayfirst=True)
    bad = df["End"].notna() & df["Start"].notna() & (df["End"] <= df["Start"])
    df.loc[bad, "End"] = df.loc[bad, "Start"] + pd.to_timedelta(1, unit="D")

    df["ID"] = df[col_id].apply(coerce_int)
    df["Predecessors"] = df[col_dep].apply(parse_preds) if col_dep else [[] for _ in range(len(df))]
    df["Status"] = df[col_stat].apply(norm_status) if col_stat else ""

    df["__order"] = np.arange(len(df))
    df = df[df[col_name].notna() & df["Start"].notna() & df["End"].notna()].copy()
    df = df[df["Status"] != "новый"].copy()

    if col_head or col_sub:
        df["IsHeadFlag"]    = df[col_head].apply(parse_bool01) if col_head else 0
        df["IsSubheadFlag"] = df[col_sub].apply(parse_bool01)  if col_sub else 0
        df["Role"] = np.select(
            [df["IsHeadFlag"].eq(1), df["IsSubheadFlag"].eq(1)],
            ["head","subhead"],
            default="task"
        )
    else:
        df["Role"] = np.where(df["Predecessors"].apply(len).eq(0), "head", "task")
        df["IsHeadFlag"] = (df["Role"]=="head").astype(int)
        df["IsSubheadFlag"] = 0

    valid_ids = set(df["ID"].dropna().astype(int))
    id_to_role = {int(i): r for i, r in zip(df["ID"], df["Role"]) if not pd.isna(i)}

    parent = []
    cur_head, cur_sub = None, None
    for _, row in df.sort_values("__order").iterrows():
        pid = None
        preds = [p for p in row["Predecessors"] if p in valid_ids]
        for p in preds[::-1]:
            if id_to_role.get(p) in {"subhead","head"}:
                pid = p; break
        if pid is None:
            if row["Role"] == "head":
                cur_head, cur_sub = int(row["ID"]), None
                pid = None
            elif row["Role"] == "subhead":
                pid = cur_head
                cur_sub = int(row["ID"])
            else:
                pid = cur_sub if cur_sub is not None else cur_head
        parent.append(pid)
    df["ParentID"] = parent

    children = {}
    for tid, p in zip(df["ID"], df["ParentID"]):
        if pd.isna(tid): continue
        if p is not None:
            children.setdefault(int(p), []).append(int(tid))

    def child_tasks(ids):
        if not ids: return []
        sub = df[df["ID"].isin(ids)]
        return sub[sub["Role"]=="task"]["ID"].astype(int).tolist()

    sub_progress = {}
    for _, r in df[df["Role"]=="subhead"].iterrows():
        kids = child_tasks(children.get(int(r["ID"]), []))
        if not kids:
            sub_progress[int(r["ID"])] = 0.0
        else:
            rows = df[df["ID"].isin(kids)]
            total = len(rows); done = (rows["Status"]=="сделано").sum()
            sub_progress[int(r["ID"])] = 100.0*done/total

    head_progress = {}
    for _, r in df[df["Role"]=="head"].iterrows():
        hid = int(r["ID"])
        kids = children.get(hid, [])
        if not kids:
            head_progress[hid] = 0.0
            continue
        subs = [k for k in kids if id_to_role.get(k)=="subhead"]
        direct_tasks = [k for k in kids if id_to_role.get(k)=="task"]

        weights, values = [], []
        for sid in subs:
            leafs = child_tasks(children.get(sid, []))
            w = max(1, len(leafs))
            v = sub_progress.get(sid, 0.0)
            weights.append(w); values.append(v)

        if direct_tasks:
            rows = df[df["ID"].isin(direct_tasks)]
            total = len(rows); done = (rows["Status"]=="сделано").sum()
            weights.append(total); values.append(100.0*done/total if total else 0.0)

        head_progress[hid] = float(np.average(values, weights=weights)) if weights else 0.0

    prog = []
    for _, r in df.iterrows():
        rid = int(r["ID"]) if not pd.isna(r["ID"]) else None
        if r["Role"] == "head":
            prog.append(head_progress.get(rid, 0.0))
        elif r["Role"] == "subhead":
            prog.append(sub_progress.get(rid, 0.0))
        else:
            prog.append(np.nan)
    df["ProgressPct"] = prog

    out_rows, seen = [], set()
    base = df.sort_values("__order")
    by_id = {int(i): row for i, row in zip(df["ID"], df.itertuples(index=False)) if not pd.isna(i)}

    def push_row(row_tuple):
        out_rows.append(pd.Series(row_tuple._asdict()))

    for row in base.itertuples(index=False):
        rid = int(row.ID) if not pd.isna(row.ID) else None
        if rid is None or rid in seen: continue
        if row.Role == "head":
            push_row(row); seen.add(rid)
            for sid in children.get(rid, []):
                if by_id.get(sid) and by_id[sid].Role == "subhead":
                    push_row(by_id[sid]); seen.add(sid)
                    for tid in children.get(sid, []):
                        if by_id.get(tid) and by_id[tid].Role == "task" and tid not in seen:
                            push_row(by_id[tid]); seen.add(tid)

    for row in base.itertuples(index=False):
        rid = int(row.ID) if not pd.isna(row.ID) else None
        if rid is not None and rid not in seen:
            push_row(row); seen.add(rid)

    out = pd.DataFrame(out_rows)

    labels = []
    for name, role in zip(out[col_name], out["Role"]):
        nm = shorten(name, 80)
        if role == "head":
            labels.append(f"<b>{nm}</b>")
        elif role == "subhead":
            labels.append(f"<b>• {nm}</b>")
        else:
            labels.append(f"• {nm}")
    out["Label"] = labels
    return out, col_name

# ----------------- plotting -----------------
def make_fig(data, scale="Месяц", show_subheads=True):
    color_by_status = {"в работе":"#9ecae1", "сделано":"#a1d99b", "запланировано":"#f3e79b"}
    role_to_pattern = {"head":"", "subhead":"/", "task":""}
    role_to_opacity = {"head":1.0, "subhead":0.75, "task":0.95}

    d = data.copy()

    # скрыть подголовные визуально, оставив их задачи под головной
    if not show_subheads:
        id_to_role = {int(i): r for i, r in zip(d["ID"], d["Role"]) if not pd.isna(i)}
        parent_role = d["ParentID"].map(lambda p: id_to_role.get(int(p)) if p is not None and not pd.isna(p) else None)
        # удваиваем маркер у задач, чьим родителем была подголовная
        def relabel(row, pr):
            lbl = row["Label"]
            if row["Role"] == "task" and pr == "subhead":
                return lbl.replace("• ", "• • ", 1)
            return lbl
        d["Label"] = [relabel(r, pr) for r, pr in zip(d.to_dict("records"), parent_role)]
        d = d[d["Role"] != "subhead"].copy()

    d["Dur"] = (pd.to_datetime(d["End"]) - pd.to_datetime(d["Start"])).dt.days.clip(lower=1)

    base = go.Bar(
        x=d["Dur"], y=d["Label"], base=d["Start"], orientation="h",
        marker=dict(
            color=[color_by_status.get(s, "#d9d9d9") for s in d["Status"]],
            opacity=[role_to_opacity.get(r, 0.95) for r in d["Role"]],
            pattern=dict(shape=[role_to_pattern.get(r, "") for r in d["Role"]]),
            line=dict(color="#666", width=[0.8 if r in ("head","subhead") else 0.3 for r in d["Role"]])
        ),
        hovertemplate="<b>%{y}</b><br>%{base|%d.%m.%Y} · %{x:.0f} дн<extra></extra>",
        name=""
    )

    prog_rows = d[d["Role"].isin(["head","subhead"])].copy()
    prog_rows["ProgDur"] = prog_rows["Dur"] * (prog_rows["ProgressPct"].fillna(0)/100.0)
    prog = go.Bar(
        x=prog_rows["ProgDur"], y=prog_rows["Label"], base=prog_rows["Start"], orientation="h",
        marker=dict(color=[color_by_status.get(s, "#9ecae1") for s in prog_rows["Status"]], opacity=1.0),
        hovertemplate="<b>%{y}</b><br>прогресс: %{x:.0f} дн<extra></extra>",
        showlegend=False
    )

    fig = go.Figure([base, prog])

    today = pd.Timestamp.today().normalize()
    fig.add_vline(x=today, line_width=1, line_dash="dot", line_color="#444")

    max_end = pd.to_datetime(d["End"]).max()
    min_start = pd.to_datetime(d["Start"]).min()
    pad = timedelta(days=max(10, int((max_end - min_start).days*0.18)))
    x_text = max_end + pad
    for lbl, role, pct in zip(d["Label"], d["Role"], d["ProgressPct"]):
        if role in ("head","subhead"):
            fig.add_annotation(x=x_text, y=lbl, text=f"{(0 if pd.isna(pct) else pct):.0f} %",
                               showarrow=False, xanchor="left", font=dict(size=11))

    if scale == "День":
        fig.update_xaxes(dtick="D1", tickformat="%d.%m.%Y")
    elif scale == "Неделя":
        fig.update_xaxes(dtick="D7", tickformat="%d.%m")
    elif scale == "Месяц":
        fig.update_xaxes(dtick="M1", tickformat="%b %Y")
    else:
        fig.update_xaxes(dtick="M3", tickformat="%b %Y")

    fig.update_yaxes(categoryorder="array", categoryarray=d["Label"].iloc[::-1].tolist())
    fig.update_layout(
        barmode="overlay", bargap=0.25, title="План-график проекта",
        height=max(650, int(36*len(d))),
        margin=dict(l=320, r=220, t=50, b=40),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

# ----------------- UI -----------------
st.sidebar.markdown("### Файл")
up = st.sidebar.file_uploader("Excel (.xlsx)", type=["xlsx"])

scale = st.sidebar.radio("Масштаб", ["День","Неделя","Месяц","Квартал"], index=2)
show_subheads = st.sidebar.checkbox("Показывать подголовные", value=True)

if not up:
    st.info("Загрузи Excel с колонками: ID, Название, Дата От, Дата До. Опционально: Предшественники, Состояние, Головная задача (1/0), Подголовная задача (1/0).")
    st.stop()

try:
    raw = pd.read_excel(up, sheet_name=0, engine="openpyxl")
    data, col_name = build_df(raw)
except Exception as e:
    st.error(f"Ошибка парсинга: {e}")
    st.stop()

col_chart, col_table = st.columns([3, 2], gap="small")

with col_chart:
    fig = make_fig(data, scale=scale, show_subheads=show_subheads)
    st.plotly_chart(fig, use_container_width=True)
    try:
        png = fig.to_image(format="png", scale=2)  # требует kaleido
        st.download_button("Скачать PNG", data=png, file_name="gantt.png", mime="image/png")
    except Exception:
        st.caption("Для выгрузки PNG установи зависимость kaleido.")

with col_table:
    st.markdown("#### Расчёты")
    show = data[["ID","Label","Start","End","Status","Role","ProgressPct","ParentID","Predecessors"]].copy()
    show = show.rename(columns={
        "Label":"Задача","Start":"Начало","End":"Окончание","Status":"Статус",
        "Role":"Роль","ProgressPct":"Прогресс, %","ParentID":"Родитель","Predecessors":"Предшественники"
    })
    st.dataframe(show, use_container_width=True, height=560)
