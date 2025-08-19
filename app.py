import re
from datetime import timedelta, date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Gantt", layout="wide")

# ---------- helpers ----------
def find_col(cols_map, subs):
    for orig, low in cols_map.items():
        for s in subs:
            if s in low:
                return orig
    return None

def safe_int(x):
    try:
        if x is None: return None
        s = str(x).strip()
        if s == "" or s.lower() == "nan": return None
        return int(float(s))
    except:
        return None

def parse_bool01(x):
    if x is None: return 0
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
        if re.fullmatch(r"\d+", p):
            ids.append(int(p))
        else:
            ids += [int(n) for n in re.findall(r"\d+", p)]
    # уникальные, без нулей
    seen, out = set(), []
    for i in ids:
        if i and i not in seen:
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

def shorten(s, n=80):
    s = str(s).strip()
    return s if len(s) <= n else s[:n-1] + "…"

def quarter_ticks(min_start: pd.Timestamp, max_end: pd.Timestamp):
    start_q = pd.Timestamp(min_start).to_period("Q").start_time.normalize()
    end_q   = pd.Timestamp(max_end).to_period("Q").end_time.normalize()
    rng = pd.period_range(start=start_q, end=end_q, freq="Q")
    vals = [p.start_time for p in rng]
    text = [f"Q{p.quarter} {p.start_time.year}" for p in rng]
    return vals, text

# ---------- core parsing ----------
def build_df(src: pd.DataFrame):
    cols = {c: str(c).strip().lower() for c in src.columns}

    col_id    = find_col(cols, ["id"])
    col_name  = find_col(cols, ["назван","задач","task"])
    col_s     = find_col(cols, ["дата от","начал","start","старт"])
    col_e     = find_col(cols, ["дата до","окон","end","finish","финиш"])
    col_dep   = find_col(cols, ["предшествен"])
    col_stat  = find_col(cols, ["состояние","статус","status"])
    col_head  = find_col(cols, ["головн"])
    col_sub   = find_col(cols, ["подголовн","subhead"])

    need = [col_id, col_name, col_s, col_e]
    if any(c is None for c in need):
        raise ValueError("Нужны столбцы: ID, Название, Дата От, Дата До.")

    df = src[[c for c in [col_id,col_name,col_s,col_e,col_dep,col_stat,col_head,col_sub] if c is not None]].copy()
    df["Start"] = pd.to_datetime(df[col_s], errors="coerce", dayfirst=True)
    df["End"]   = pd.to_datetime(df[col_e], errors="coerce", dayfirst=True)
    bad = df["End"].notna() & df["Start"].notna() & (df["End"] <= df["Start"])
    df.loc[bad, "End"] = df.loc[bad, "Start"] + pd.to_timedelta(1, unit="D")

    df["ID"] = df[col_id].apply(safe_int)
    df["Predecessors"] = df[col_dep].apply(parse_preds) if col_dep else [[] for _ in range(len(df))]
    # убрать нули и самоссылки
    df["Predecessors"] = [
        [p for p in L if (p and p != (i if i is not None else -1))]
        for L, i in zip(df["Predecessors"], df["ID"])
    ]
    df["Status"] = df[col_stat].apply(norm_status) if col_stat else ""

    df["__order"] = np.arange(len(df))
    df = df[df[col_name].notna() & df["Start"].notna() & df["End"].notna()].copy()
    df = df[df["Status"] != "новый"].copy()

    # роли
    if col_head or col_sub:
        df["IsHeadFlag"]    = df[col_head].apply(parse_bool01) if col_head else 0
        df["IsSubheadFlag"] = df[col_sub].apply(parse_bool01)  if col_sub else 0
        df["Role"] = np.select(
            [df["IsHeadFlag"].eq(1), df["IsSubheadFlag"].eq(1)],
            ["head","subhead"],
            default="task"
        )
    else:
        df["Role"] = "task"

    valid_ids = {i for i in df["ID"].dropna().tolist()}
    id_to_role = {i: r for i, r in zip(df["ID"], df["Role"]) if i is not None}

    # родитель по предшественникам/контексту
    parent = []
    cur_head, cur_sub = None, None
    for _, row in df.sort_values("__order").iterrows():
        pid = None
        preds = [p for p in row["Predecessors"] if p in valid_ids]
        for p in reversed(preds):
            if id_to_role.get(p, "") in {"subhead","head"}:
                pid = p; break
        if pid is None:
            rid = row["ID"]; role = row["Role"]
            if role == "head":
                cur_head, cur_sub = rid, None
                pid = None
            elif role == "subhead":
                pid = cur_head
                cur_sub = rid
            else:
                pid = cur_sub if cur_sub is not None else cur_head
        parent.append(pid)
    df["ParentID"] = parent

    # если флагов нет, авто subhead по наличию детей у задач, чьим родителем head
    children = {}
    for tid, p in zip(df["ID"], df["ParentID"]):
        if tid is not None and p is not None:
            children.setdefault(p, []).append(tid)
    if (col_head is None and col_sub is None) or (df.get("IsSubheadFlag", pd.Series(dtype=int)).sum() == 0):
        df["Role"] = np.where(df["ParentID"].isna(), "head", df["Role"])
        has_children = df["ID"].map(lambda i: len(children.get(i, [])) if i is not None else 0)
        parent_role = df["ParentID"].map(lambda p: "head" if p is None else id_to_role.get(p, "task"))
        df.loc[(has_children > 0) & (parent_role == "head") & (df["Role"] != "head"), "Role"] = "subhead"

    id_to_role = {i: r for i, r in zip(df["ID"], df["Role"]) if i is not None}

    # прогресс
    def leaf_tasks(ids):
        if not ids: return []
        sub = df[df["ID"].isin(ids)]
        return sub[sub["Role"]=="task"]["ID"].tolist()

    sub_progress = {}
    for _, r in df[df["Role"]=="subhead"].iterrows():
        rid = r["ID"]
        kids = leaf_tasks(children.get(rid, []))
        if not kids:
            sub_progress[rid] = 0.0
        else:
            rows = df[df["ID"].isin(kids)]
            total = len(rows); done = (rows["Status"]=="сделано").sum()
            sub_progress[rid] = 100.0*done/total

    head_progress = {}
    for _, r in df[df["Role"]=="head"].iterrows():
        hid = r["ID"]
        kids = children.get(hid, [])
        subs = [k for k in kids if id_to_role.get(k)=="subhead"]
        direct_tasks = [k for k in kids if id_to_role.get(k)=="task"]
        weights, values = [], []
        for sid in subs:
            leafs = leaf_tasks(children.get(sid, []))
            w = max(1, len(leafs))
            v = sub_progress.get(sid, 0.0)
            weights.append(w); values.append(v)
        if direct_tasks:
            rows = df[df["ID"].isin(direct_tasks)]
            total = len(rows); done = (rows["Status"]=="сделано").sum()
            weights.append(total); values.append(100.0*done/total if total else 0.0)
        head_progress[hid] = float(np.average(values, weights=weights)) if weights else 0.0

    df["ProgressPct"] = np.where(df["Role"]=="head",
                                 df["ID"].map(head_progress).fillna(0.0),
                        np.where(df["Role"]=="subhead",
                                 df["ID"].map(sub_progress).fillna(0.0),
                                 np.nan))

    # верхняя головная для каждого ряда — пригодится для фильтра/подсветки
    id_to_parent = {i: p for i, p in zip(df["ID"], df["ParentID"])}
    def top_head(i):
        seen = set()
        cur = i
        while cur is not None and cur in id_to_parent and cur not in seen:
            seen.add(cur)
            p = id_to_parent[cur]
            if p is None:  # это и есть head
                return cur
            cur = p
        return None
    df["TopHeadID"] = [top_head(i) if i is not None else None for i in df["ID"]]
    id_to_name = {i: n for i, n in zip(df["ID"], df[col_name]) if i is not None}
    df["TopHeadName"] = df["TopHeadID"].map(lambda i: id_to_name.get(i))

    # SubheadID: у subhead свой, у task — ID родителя если parent=subhead
    df["SubheadID"] = np.where(df["Role"]=="subhead", df["ID"],
                         np.where(df["Role"]=="task",
                                  df["ParentID"], None))

    # подписи: plain для оси, html для слева
    plain, html = [], []
    for name, role in zip(df[col_name], df["Role"]):
        nm = shorten(name, 80)
        if role == "head":
            plain.append(f"  {nm}")
            html.append(f"<b>{nm}</b>")
        elif role == "subhead":
            plain.append(f"    • {nm}")
            html.append(f"<b>• {nm}</b>")
        else:
            plain.append(f"      • {nm}")
            html.append(f"• {nm}")
    df["LabelPlain"] = plain
    df["LabelHTML"]  = html

    return df

# ---------- plotting ----------
PALETTE = [
    "#b3cde3", "#ccebc5", "#decbe4", "#fbb4ae", "#fed9a6",
    "#e5d8bd", "#fddaec", "#cbd5e8", "#f2f2f2"
]

def make_fig(data, scale="Месяц", show_subheads=True, tint_groups=True):
    d = data.copy()

    # скрыть subhead строки, оставив их задачи
    if not show_subheads:
        keep_mask = d["Role"] != "subhead"
        d = d[keep_mask].copy()

    # период по данным
    min_start_all = pd.to_datetime(d["Start"]).min()
    max_end_all   = pd.to_datetime(d["End"]).max()

    d["Dur"] = (pd.to_datetime(d["End"]) - pd.to_datetime(d["Start"])).dt.days.clip(lower=1)
    color_by_status = {"в работе":"#9ecae1", "сделано":"#a1d99b", "запланировано":"#f3e79b"}

    base = go.Bar(
        x=d["Dur"], y=d["LabelPlain"], base=d["Start"], orientation="h",
        marker=dict(
            color=[color_by_status.get(s, "#d9d9d9") for s in d["Status"]],
            line=dict(color="#68707a",
                      width=[1.0 if r=="head" else (0.8 if r=="subhead" else 0.4) for r in d["Role"]]),
            pattern=dict(shape=["" if r=="head" else ("/" if r=="subhead" else "") for r in d["Role"]],
                         fillmode="overlay", size=6, solidity=0.06)
        ),
        hovertemplate="<b>%{y}</b><br>%{base|%d.%m.%Y} · %{x:.0f} дн<extra></extra>",
        name=""
    )

    prog_rows = d[d["Role"].isin(["head","subhead"])].copy()
    prog_rows["ProgDur"] = prog_rows["Dur"] * (prog_rows["ProgressPct"].fillna(0)/100.0)
    prog = go.Bar(
        x=prog_rows["ProgDur"], y=prog_rows["LabelPlain"], base=prog_rows["Start"], orientation="h",
        marker=dict(color=[color_by_status.get(s, "#9ecae1") for s in prog_rows["Status"]],
                    opacity=0.95),
        hovertemplate="<b>%{y}</b><br>прогресс: %{x:.0f} дн<extra></extra>",
        showlegend=False
    )

    fig = go.Figure([base, prog])

    # тики по масштабу
    if scale == "День":
        fig.update_xaxes(dtick="D1", tickformat="%d.%m.%Y")
    elif scale == "Неделя":
        fig.update_xaxes(dtick="D7", tickformat="%d.%m")
    elif scale == "Месяц":
        fig.update_xaxes(dtick="M1", tickformat="%b %Y")
    else:  # Квартал
        vals, text = quarter_ticks(min_start_all, max_end_all)
        fig.update_xaxes(tickmode="array", tickvals=vals, ticktext=text)

    # диапазон оси X с запасом слева/справа, чтобы подписи не прилипали
    span_days = max(30, (max_end_all - min_start_all).days)
    pad_left = timedelta(days=max(10, int(span_days*0.20)))
    pad_right = timedelta(days=max(10, int(span_days*0.16)))
    fig.update_xaxes(range=[min_start_all - pad_left, max_end_all + pad_right])

    # ось Y: тики скрываем, рисуем свои подписи слева
    fig.update_yaxes(showticklabels=False,
                     categoryorder="array",
                     categoryarray=d["LabelPlain"].iloc[::-1])

    x_left = min_start_all - pad_left*0.98
    for plain, html in zip(d["LabelPlain"], d["LabelHTML"]):
        fig.add_annotation(x=x_left, y=plain, xref="x", yref="y",
                           text=html, showarrow=False, xanchor="right", align="right")

    # проценты справа
    x_right = max_end_all + pad_right*0.35
    for plain, role, pct in zip(d["LabelPlain"], d["Role"], d["ProgressPct"]):
        if role in ("head","subhead"):
            val = 0.0 if pd.isna(pct) else float(pct)
            fig.add_annotation(x=x_right, y=plain, xref="x", yref="y",
                               text=f"{val:.0f} %", showarrow=False, xanchor="left")

    # мягкая подсветка блоков по подголовным
    if tint_groups:
        pal_i = 0
        present = set(d["LabelPlain"])
        # группы: subhead -> его label + labels задач
        sub_groups = {}
        for _, r in data[data["Role"]=="subhead"].iterrows():
            labs = [r["LabelPlain"]]
            labs += data.loc[data["ParentID"].eq(r["ID"]) & data["Role"].eq("task"), "LabelPlain"].tolist()
            labs = [lab for lab in labs if lab in present]
            if labs:
                sub_groups[r["ID"]] = labs
        # рисуем
        for sub_id, labels in sub_groups.items():
            y0, y1 = labels[0], labels[-1]
            color = PALETTE[pal_i % len(PALETTE)]; pal_i += 1
            fig.add_hrect(y0=y0, y1=y1, line_width=0, fillcolor=color, opacity=0.08, layer="below")

    # линия сегодня
    today = pd.Timestamp.today().normalize()
    fig.add_vline(x=today, line_width=1, line_dash="dot", line_color="#444")

    fig.update_layout(
        barmode="overlay", bargap=0.25, template="plotly_white",
        height=max(720, int(34*len(d))),
        margin=dict(l=340, r=260, t=60, b=40),
        title="План-график проекта",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

# ---------- UI ----------
st.sidebar.markdown("### Файл")
up = st.sidebar.file_uploader("Excel (.xlsx)", type=["xlsx"])

scale = st.sidebar.radio("Масштаб", ["День","Неделя","Месяц","Квартал"], index=3)
show_subheads = st.sidebar.checkbox("Показывать подголовные", value=True)
tint_groups = st.sidebar.checkbox("Мягкая подсветка блоков", value=True)

# слайдер периода, чтобы обрезать вид
st.sidebar.markdown("### Период")
placeholder = [date.today().replace(month=1, day=1), date.today()]
date_from = st.sidebar.date_input("c", value=placeholder[0])
date_to   = st.sidebar.date_input("по", value=placeholder[1])

if not up:
    st.info("Загрузи Excel: ID, Название, Дата От, Дата До. Опционально: Предшественники, Состояние, Головная задача (1/0), Подголовная задача (1/0).")
    st.stop()

try:
    raw = pd.read_excel(up, sheet_name=0, engine="openpyxl")
    data = build_df(raw)
except Exception as e:
    st.error(f"Ошибка парсинга: {e}")
    st.stop()

# применяем выбранный период к данным
try:
    d_from = pd.to_datetime(date_from)
    d_to   = pd.to_datetime(date_to) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask_period = (data["End"] >= d_from) & (data["Start"] <= d_to)
    data = data[mask_period].copy()
except Exception:
    pass

col_chart, col_table = st.columns([3, 2], gap="small")

with col_chart:
    fig = make_fig(data, scale=scale, show_subheads=show_subheads, tint_groups=tint_groups)
    st.plotly_chart(fig, use_container_width=True)
    try:
        png = fig.to_image(format="png", scale=2)  # kaleido
        st.download_button("Скачать PNG", data=png, file_name="gantt.png", mime="image/png")
    except Exception:
        st.caption("Для выгрузки PNG установи зависимость kaleido.")

with col_table:
    st.markdown("#### Расчёты")
    show = data[["ID","LabelHTML","Start","End","Status","Role","ProgressPct","ParentID","Predecessors","TopHeadName","SubheadID"]].copy()
    show = show.rename(columns={
        "LabelHTML":"Задача","Start":"Начало","End":"Окончание","Status":"Статус",
        "Role":"Роль","ProgressPct":"Прогресс, %","ParentID":"Родитель",
        "Predecessors":"Предшественники","TopHeadName":"Головная","SubheadID":"Подголовная ID"
    })
    st.dataframe(show, use_container_width=True, height=650)
