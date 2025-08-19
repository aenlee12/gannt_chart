import re
from datetime import timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Gantt", layout="wide")

# ---------- utils ----------
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
        if s == "" or s.lower() == "nan":
            return None
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

# ---------- core ----------
def build_df(src: pd.DataFrame):
    cols = {c: str(c).strip().lower() for c in src.columns}

    col_id    = find_col(cols, ["id"])
    col_name  = find_col(cols, ["назван", "задач", "task"])
    col_s     = find_col(cols, ["дата от","начал","start","старт"])
    col_e     = find_col(cols, ["дата до","окон","end","finish","финиш"])
    col_dep   = find_col(cols, ["предшествен"])
    col_stat  = find_col(cols, ["состояние","статус","status"])
    col_head  = find_col(cols, ["головн"])             # Головная задача
    col_sub   = find_col(cols, ["подголовн","subhead"]) # Подголовная задача

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
    # выкинуть 0 и самоссылки
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
        df["Role"] = "task"  # временно, дальше уточним

    valid_ids = {i for i in df["ID"].dropna().tolist()}
    id_to_role = {i: r for i, r in zip(df["ID"], df["Role"]) if i is not None}

    # Родитель по предшественникам/контексту
    parent = []
    cur_head, cur_sub = None, None
    for _, row in df.sort_values("__order").iterrows():
        pid = None
        preds = [p for p in row["Predecessors"] if p in valid_ids]
        for p in reversed(preds):
            if id_to_role.get(p, "") in {"subhead","head"}:
                pid = p; break
        if pid is None:
            rid = row["ID"]
            role = row["Role"]
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

    # если флагов нет — автоопределяем роли:
    if (col_head is None and col_sub is None) or (df["Role"].nunique() == 1):
        # «head» — у кого нет ParentID; «subhead» — у кого есть дети; остальное — task
        df["Role"] = "task"
        df.loc[df["ParentID"].isna(), "Role"] = "head"
        # после построения children дадим статус subhead
    # дети
    children = {}
    for tid, p in zip(df["ID"], df["ParentID"]):
        if tid is None: 
            continue
        if p is not None:
            children.setdefault(p, []).append(tid)

    if (col_head is None and col_sub is None) or (df["IsSubheadFlag"].sum() == 0):
        has_children = df["ID"].map(lambda i: len(children.get(i, [])) if i is not None else 0)
        # subhead — у кого есть дети и есть родитель head
        parent_role = df["ParentID"].map(lambda p: "head" if p is None else id_to_role.get(p, "task"))
        df.loc[(has_children > 0) & (parent_role == "head") & (df["Role"] != "head"), "Role"] = "subhead"

    # пересчитать map (мог поменяться Role)
    id_to_role = {i: r for i, r in zip(df["ID"], df["Role"]) if i is not None}

    def leaf_tasks(ids):
        if not ids: return []
        sub = df[df["ID"].isin(ids)]
        return sub[sub["Role"]=="task"]["ID"].tolist()

    # прогресс подголовных
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

    # прогресс головных (взвешенное среднее подголовных + прямые задачи)
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

    prog = []
    for _, r in df.iterrows():
        if r["Role"] == "head":
            prog.append(head_progress.get(r["ID"], 0.0))
        elif r["Role"] == "subhead":
            prog.append(sub_progress.get(r["ID"], 0.0))
        else:
            prog.append(np.nan)
    df["ProgressPct"] = prog

    # порядок: head -> subhead -> tasks
    out_rows, seen = [], set()
    base = df.sort_values("__order")
    by_id = {i: row for i, row in zip(df["ID"], df.itertuples(index=False)) if i is not None}

    def push_row(row_tuple):
        out_rows.append(pd.Series(row_tuple._asdict()))

    for row in base.itertuples(index=False):
        rid = row.ID
        if rid in seen: continue
        if row.Role == "head":
            push_row(row); seen.add(rid)
            for sid in children.get(rid, []):
                if sid in by_id and by_id[sid].Role == "subhead" and sid not in seen:
                    push_row(by_id[sid]); seen.add(sid)
                    for tid in children.get(sid, []):
                        if tid in by_id and by_id[tid].Role == "task" and tid not in seen:
                            push_row(by_id[tid]); seen.add(tid)

    for row in base.itertuples(index=False):
        rid = row.ID
        if rid is not None and rid not in seen:
            push_row(row); seen.add(rid)

    out = pd.DataFrame(out_rows)

    # подписи (plain для категорий и html для аннотаций)
    plain_labels = []
    html_labels  = []
    for name, role in zip(out[col_name], out["Role"]):
        nm = shorten(name, 80)
        if role == "head":
            plain_labels.append(f"  {nm}")
            html_labels.append(f"<b>{nm}</b>")
        elif role == "subhead":
            plain_labels.append(f"    • {nm}")
            html_labels.append(f"<b>• {nm}</b>")
        else:
            plain_labels.append(f"      • {nm}")
            html_labels.append(f"• {nm}")
    out["LabelPlain"] = plain_labels
    out["LabelHTML"]  = html_labels
    return out

# ---------- plotting ----------
PALETTE = [
    "#b3cde3", "#ccebc5", "#decbe4", "#fbb4ae", "#fed9a6",
    "#e5d8bd", "#fddaec", "#cbd5e8", "#f2f2f2"
]

def make_fig(data, scale="Месяц", show_subheads=True, tint_groups=True):
    d = data.copy()
    if not show_subheads:
        # скрываем subhead строки, но их задачи оставляем
        keep_mask = d["Role"] != "subhead"
        d = d[keep_mask].copy()

    d["Dur"] = (pd.to_datetime(d["End"]) - pd.to_datetime(d["Start"])).dt.days.clip(lower=1)

    color_by_status = {"в работе":"#9ecae1", "сделано":"#a1d99b", "запланировано":"#f3e79b"}

    # Бары
    base = go.Bar(
        x=d["Dur"], y=d["LabelPlain"], base=d["Start"], orientation="h",
        marker=dict(
            color=[color_by_status.get(s, "#d9d9d9") for s in d["Status"]],
            line=dict(color="#666", width=[0.9 if r in ("head","subhead") else 0.4 for r in d["Role"]]),
            pattern=dict(shape=["" if r=="head" else ("/" if r=="subhead" else "") for r in d["Role"]],
                         fillmode="overlay", size=6, solidity=0.05)
        ),
        hovertemplate="<b>%{y}</b><br>%{base|%d.%m.%Y} · %{x:.0f} дн<extra></extra>",
        name=""
    )

    # Прогресс поверх для head/subhead
    prog_rows = d[d["Role"].isin(["head","subhead"])].copy()
    prog_rows["ProgDur"] = prog_rows["Dur"] * (prog_rows["ProgressPct"].fillna(0)/100.0)
    prog = go.Bar(
        x=prog_rows["ProgDur"], y=prog_rows["LabelPlain"], base=prog_rows["Start"], orientation="h",
        marker=dict(color=[color_by_status.get(s, "#9ecae1") for s in prog_rows["Status"]], opacity=0.95),
        hovertemplate="<b>%{y}</b><br>прогресс: %{x:.0f} дн<extra></extra>",
        showlegend=False
    )

    fig = go.Figure([base, prog])

    # Диапазон времени + место слева под аннотации
    min_start = pd.to_datetime(d["Start"]).min()
    max_end = pd.to_datetime(d["End"]).max()
    span_days = max(30, (max_end - min_start).days)
    pad_left = timedelta(days=max(15, int(span_days*0.25)))
    pad_right = timedelta(days=max(15, int(span_days*0.20)))
    fig.update_xaxes(range=[min_start - pad_left, max_end + pad_right])

    # Ось Y — прячем подписи и рисуем свои жирные
    fig.update_yaxes(showticklabels=False, categoryorder="array", categoryarray=d["LabelPlain"].iloc[::-1])

    # Аннотации слева с HTML
    x_left = min_start - pad_left*0.98
    for plain, html in zip(d["LabelPlain"], d["LabelHTML"]):
        fig.add_annotation(x=x_left, y=plain, xref="x", yref="y",
                           text=html, showarrow=False, xanchor="right", align="right")

    # Проценты справа для head/subhead
    x_right = max_end + pad_right*0.35
    for plain, role, pct in zip(d["LabelPlain"], d["Role"], d["ProgressPct"]):
        if role in ("head","subhead"):
            val = 0.0 if pd.isna(pct) else float(pct)
            fig.add_annotation(x=x_right, y=plain, xref="x", yref="y",
                               text=f"{val:.0f} %", showarrow=False, xanchor="left")

    # Мягкая подсветка блоков подголовных
    if tint_groups:
        # строим группы по текущему набору строк (с учётом show_subheads)
        # берём subhead из исходного data, но ограничиваемся теми, чьи задачи видимы
        # найдём диапазон y для каждой «подголовная + её задачи»
        palette_idx = 0
        # подготовим быстрые карты
        id_to_role = {i: r for i, r in zip(data["ID"], data["Role"]) if i is not None}
        # карта: subhead_id -> список labelPlain в d
        # для этого из исходного набора соберём ParentID связи
        child_map = {}
        for _, r in data.iterrows():
            pid = r.get("ParentID")
            if (pid is not None) and id_to_role.get(pid) == "subhead":
                child_map.setdefault(pid, []).append(r["LabelPlain"])
        # добавим сам subhead label
        for _, r in data[data["Role"]=="subhead"].iterrows():
            labels = []
            # сам subhead может быть скрыт (если show_subheads=False), тогда берём только задачи
            if show_subheads:
                labels.append(r["LabelPlain"])
            labels += child_map.get(r["ID"], [])
            # фильтруем теми, кто действительно на графике d
            labels = [lab for lab in labels if lab in set(d["LabelPlain"])]
            if not labels: 
                continue
            y0 = labels[0]
            y1 = labels[-1]
            col = PALETTE[palette_idx % len(PALETTE)]; palette_idx += 1
            # очень бледный
            fig.add_shape(type="rect", xref="paper", yref="y",
                          x0=0, x1=1, y0=y0, y1=y1,
                          fillcolor=col, opacity=0.08, line_width=0, layer="below")

    # линия «сегодня»
    today = pd.Timestamp.today().normalize()
    fig.add_vline(x=today, line_width=1, line_dash="dot", line_color="#444")

    # Масштаб
    if scale == "День":
        fig.update_xaxes(dtick="D1", tickformat="%d.%m.%Y")
    elif scale == "Неделя":
        fig.update_xaxes(dtick="D7", tickformat="%d.%m")
    elif scale == "Месяц":
        fig.update_xaxes(dtick="M1", tickformat="%b %Y")
    else:
        fig.update_xaxes(dtick="M3", tickformat="%b %Y")

    fig.update_layout(
        barmode="overlay", bargap=0.25, template="plotly_white",
        height=max(700, int(36*len(d))),
        margin=dict(l=320, r=240, t=60, b=40),
        title="План-график проекта",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

# ---------- UI ----------
st.sidebar.markdown("### Файл")
up = st.sidebar.file_uploader("Excel (.xlsx)", type=["xlsx"])
scale = st.sidebar.radio("Масштаб", ["День","Неделя","Месяц","Квартал"], index=2)
show_subheads = st.sidebar.checkbox("Показывать подголовные", value=True)
tint_groups = st.sidebar.checkbox("Мягкая подсветка блоков", value=True)

if not up:
    st.info("Загрузи Excel: ID, Название, Дата От, Дата До. Опционально: Предшественники, Состояние, Головная задача (1/0), Подголовная задача (1/0).")
    st.stop()

try:
    raw = pd.read_excel(up, sheet_name=0, engine="openpyxl")
    data = build_df(raw)
except Exception as e:
    st.error(f"Ошибка парсинга: {e}")
    st.stop()

col_chart, col_table = st.columns([3, 2], gap="small")

with col_chart:
    fig = make_fig(data, scale=scale, show_subheads=show_subheads, tint_groups=tint_groups)
    st.plotly_chart(fig, use_container_width=True)
    try:
        png = fig.to_image(format="png", scale=2)  # требует kaleido
        st.download_button("Скачать PNG", data=png, file_name="gantt.png", mime="image/png")
    except Exception:
        st.caption("Для выгрузки PNG установи зависимость kaleido.")

with col_table:
    st.markdown("#### Расчёты")
    show = data[["ID","LabelHTML","Start","End","Status","Role","ProgressPct","ParentID","Predecessors"]].copy()
    show = show.rename(columns={
        "LabelHTML":"Задача","Start":"Начало","End":"Окончание","Status":"Статус",
        "Role":"Роль","ProgressPct":"Прогресс, %","ParentID":"Родитель","Predecessors":"Предшественники"
    })
    st.dataframe(show, use_container_width=True, height=600)
