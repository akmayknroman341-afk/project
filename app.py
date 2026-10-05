import os
import re
import json
import time
import sqlite3
import streamlit as st
import pandas as pd
from datetime import datetime
from groq import Groq
from json_repair import repair_json


# ============================================================
# 1. НАСТРОЙКА СТРАНИЦЫ + СКРЫТИЕ ЭЛЕМЕНТОВ АВТОРА
# ============================================================

st.set_page_config(
    page_title="Тема за 7 дней",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    /* Скрыть меню (три точки) */
    #MainMenu {visibility: hidden;}
    
    /* Скрыть футер "Made with Streamlit" */
    footer {visibility: hidden;}
    
    /* Скрыть тулбар (Deploy, Record) */
    [data-testid="stToolbar"] {display: none !important;}
    
    /* Скрыть цветную полоску сверху */
    [data-testid="stDecoration"] {display: none !important;}
    
    /* Скрыть индикатор "Running" */
    [data-testid="stStatusWidget"] {display: none !important;}
    
    /* Скрыть кнопку "Manage app" */
    [data-testid="manage-app-button"] {display: none !important;}
    
    /* ПРИНУДИТЕЛЬНО ПОКАЗАТЬ кнопку сайдбара (несколько вариантов селекторов) */
    button[data-testid="stBaseButton-headerNoPadding"],
    button[data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed !important;
        top: 10px !important;
        left: 10px !important;
        z-index: 999999 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 2. ПОЛУЧЕНИЕ КЛЮЧЕЙ
# ============================================================

try:
    api_key = st.secrets["GROQ_API_KEY"]
    model_name = st.secrets.get("GROQ_MODEL", "openai/gpt-oss-20b")
except (KeyError, FileNotFoundError):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not api_key:
    st.error(
        "❌ Не найден GROQ_API_KEY.\n\n"
        "**На Streamlit Cloud:** добавь его в Manage app → Settings → Secrets.\n\n"
        "**Локально:** создай файл `.env` с ключом."
    )
    st.stop()

client = Groq(api_key=api_key)
MODEL = model_name
DB = "progress.db"


# ============================================================
# 3. БАЗА ДАННЫХ (SQLite)
# ============================================================

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student TEXT,
            topic TEXT,
            plan_json TEXT,
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student TEXT,
            topic TEXT,
            day INTEGER,
            question TEXT,
            correct_answer TEXT,
            student_answer TEXT,
            is_correct INTEGER,
            ai_feedback TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_course(student, topic, plan):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO courses (student, topic, plan_json, created_at) VALUES (?,?,?,?)",
        (student, topic, json.dumps(plan, ensure_ascii=False), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def save_attempt(student, topic, day, question, correct, given, is_correct, feedback=""):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        """INSERT INTO attempts
           (student, topic, day, question, correct_answer, student_answer,
            is_correct, ai_feedback, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (student, topic, day, question, correct, given,
         int(is_correct), feedback, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_progress(student, topic):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        """SELECT day, COUNT(*), SUM(is_correct)
           FROM attempts WHERE student=? AND topic=?
           GROUP BY day ORDER BY day""",
        (student, topic),
    )
    rows = c.fetchall()
    conn.close()
    return rows


# ============================================================
# 4. AI-ФУНКЦИИ (Groq + Prefilling + json-repair)
# ============================================================

def ai_json(prompt: str, system: str = "Ты — опытный школьный учитель.",
            retries: int = 3) -> dict:
    """Запрос к Groq с Prefilling и ремонтом JSON."""
    last_err = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "```json\n{"},
                ],
                temperature=0.1,
                max_tokens=950,
                reasoning_effort="low",
                timeout=45,
            )

            if response.choices[0].finish_reason == "length":
                raise ValueError("Ответ оборвался (max_tokens).")

            content = response.choices[0].message.content
            content = "{" + content

            content = re.sub(r'^```(?:json)?\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            content = content.strip()
            content = re.sub(r'</?function[^>]*>', '', content).strip()

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass

            try:
                repaired = repair_json(content)
                return json.loads(repaired)
            except Exception as repair_err:
                last_err = repair_err

            fixed = content.replace("'", '"')
            fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

            match = re.search(r'\{.*\}', fixed, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

            raise ValueError(f"Не удалось распарсить JSON: {content[:200]}")

        except Exception as e:
            last_err = e
            time.sleep(2 + attempt)
            continue

    raise RuntimeError(f"LLM не вернул валидный JSON после {retries} попыток: {last_err}")


def build_outline(topic: str, grade: int, days: int = 7) -> dict:
    """Шаг 1: оглавление курса."""
    prompt = f"""Составь оглавление учебного курса по теме «{topic}» для ученика {grade} класса.
Курс на {days} дней, по 20-30 минут в день.

Формат ответа — строго JSON:
{{
  "topic": "{topic}",
  "grade": {grade},
  "prerequisites": ["тема1", "тема2"],
  "days": [
    {{"day": 1, "title": "Название дня", "goal": "Что освоит ученик"}}
  ]
}}

Ровно {days} дней. Только заголовки и цели, без теории и задач.
Ответь ТОЛЬКО валидным JSON."""
    return ai_json(prompt)


def build_day_theory(topic: str, grade: int, day_info: dict) -> dict:
    """Шаг 2а: теория и пример для одного дня."""
    prompt = f"""Тема: «{topic}» ({grade} класс).
День {day_info['day']}: {day_info['title']}
Цель: {day_info['goal']}

Сформулируй теорию и пример.

Формат ответа — строго JSON:
{{
  "theory": "Объяснение на 3-5 предложений простым языком",
  "example": "Разобранный пример"
}}

Ответь ТОЛЬКО валидным JSON."""
    return ai_json(prompt)


def build_day_tasks(topic: str, grade: int, day_info: dict) -> dict:
    """Шаг 2б: 3 задачи для одного дня."""
    prompt = f"""Тема: «{topic}» ({grade} класс).
День {day_info['day']}: {day_info['title']}
Цель: {day_info['goal']}

Составь 3 задачи по этой теме.

Формат ответа — строго JSON:
{{
  "tasks": [
    {{"question": "Задача 1", "answer": "Ответ", "hint": "Подсказка"}},
    {{"question": "Задача 2", "answer": "Ответ", "hint": "Подсказка"}},
    {{"question": "Задача 3", "answer": "Ответ", "hint": "Подсказка"}}
  ]
}}

Ровно 3 задачи. Все ответы точные.
Ответь ТОЛЬКО валидным JSON."""
    return ai_json(prompt)


def check_answer_ai(topic: str, question: str, correct: str, given: str) -> dict:
    """Проверка ответа ученика."""
    prompt = f"""Тема: {topic}
Вопрос: {question}
Правильный ответ: {correct}
Ответ ученика: {given}

Проверь, верен ли ответ ученика по смыслу.

Формат ответа — строго JSON:
{{"is_correct": true, "feedback": "Комментарий 1-2 предложения", "mistake_type": null}}

Ответь ТОЛЬКО валидным JSON."""
    return ai_json(prompt)


def check_explanation(topic: str, student_text: str) -> dict:
    """Оценка понимания через объяснение."""
    prompt = f"""Тема: {topic}
Ученик объясняет своими словами: «{student_text}»

Оцени понимание от 0 до 10.

Формат ответа — строго JSON:
{{"score": 7, "good": "что понял", "gaps": "чего не хватает", "advice": "совет"}}

Ответь ТОЛЬКО валидным JSON."""
    return ai_json(prompt)


# ============================================================
# 5. ИНИЦИАЛИЗАЦИЯ
# ============================================================

init_db()

st.title("🧠 Тема за 7 дней")
st.caption("Введи школьную тему — нейросеть составит персональный курс")


# ============================================================
# 6. БОКОВАЯ ПАНЕЛЬ
# ============================================================

with st.sidebar:
    st.header("Настройки")
    student = st.text_input("Имя ученика", value="Ученик")
    grade = st.selectbox("Класс", list(range(1, 12)), index=7)

    quick_topics = [
        "Квадратные уравнения",
        "Производная функции",
        "Закон Ома",
        "Причастие и деепричастие",
        "Строение атома",
        "Теорема Пифагора",
    ]
    picked = st.selectbox("Готовая тема:", ["— своя тема —"] + quick_topics)
    topic = st.text_input("Своя тема:", value="" if picked == "— своя тема —" else picked)

    days = st.slider("Сколько дней в курсе", 3, 7, 3)

    if st.button("🚀 Составить курс", type="primary"):
        if not topic.strip():
            st.error("Введи тему.")
        else:
            progress = st.progress(0, text="📋 Готовим оглавление курса...")
            try:
                outline = build_outline(topic, grade, days)
                progress.progress(15, text="📋 Оглавление готово. Заполняем дни...")

                full_days = []
                day_list = outline.get("days", [])
                total = max(len(day_list), 1)

                for i, day_info in enumerate(day_list):
                    pct = 15 + int(80 * (i + 1) / total)
                    progress.progress(
                        pct,
                        text=f"📝 День {day_info['day']}: {day_info['title']}"
                    )

                    try:
                        theory_data = build_day_theory(topic, grade, day_info)
                        theory = theory_data.get("theory", "")
                        example = theory_data.get("example", "")
                    except Exception as e:
                        theory = f"(не удалось: {e})"
                        example = ""

                    time.sleep(0.3)

                    try:
                        tasks_data = build_day_tasks(topic, grade, day_info)
                        tasks = tasks_data.get("tasks", [])
                    except Exception as e:
                        tasks = []

                    time.sleep(0.3)

                    full_days.append({
                        "day": day_info["day"],
                        "title": day_info["title"],
                        "goal": day_info["goal"],
                        "theory": theory,
                        "example": example,
                        "tasks": tasks,
                    })

                outline["days"] = full_days
                progress.progress(100, text="✅ Курс готов!")

                save_course(student, topic, outline)
                st.session_state.plan = outline
                st.session_state.topic = topic
                st.session_state.mistakes = []
                time.sleep(0.5)
                st.rerun()

            except Exception as e:
                st.error(f"Ошибка генерации: {e}")

    if st.button("📊 Моя статистика"):
        st.session_state.show_stats = True


# ============================================================
# 7. СТАТИСТИКА
# ============================================================

if st.session_state.get("show_stats"):
    st.subheader("📊 Ваш прогресс")
    if "topic" in st.session_state:
        rows = get_progress(student, st.session_state.topic)
        if rows:
            df = pd.DataFrame(rows, columns=["День", "Всего", "Правильно"])
            df["Процент"] = (df["Правильно"] / df["Всего"] * 100).round(1)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Пока нет решённых задач.")


# ============================================================
# 8. ОСНОВНОЙ ЭКРАН
# ============================================================

if "plan" not in st.session_state:
    st.info("👈 Введи тему в боковой панели и нажми «Составить курс»")
    st.stop()

plan = st.session_state.plan
topic = st.session_state.topic

st.header(f"📚 {plan['topic']} · {plan['grade']} класс")

with st.expander("🔍 Что стоит вспомнить"):
    for p in plan.get("prerequisites", []):
        st.write(f"• {p}")

day_titles = [f"День {d['day']}: {d['title']}" for d in plan["days"]]
tabs = st.tabs(day_titles)

for idx, (tab, day) in enumerate(zip(tabs, plan["days"])):
    with tab:
        st.subheader(day["title"])
        st.caption(f"🎯 Цель: {day['goal']}")

        with st.expander("📖 Теория", expanded=True):
            st.write(day["theory"])
            st.markdown("**Пример:**")
            st.write(day["example"])

        st.markdown("### ✍️ Задачи")

        for i, task in enumerate(day["tasks"]):
            key = f"t_{idx}_{i}"
            st.markdown(f"**Задача {i+1}.** {task['question']}")

            col1, col2 = st.columns([3, 1])
            with col1:
                answer = st.text_input("Ответ:", key=key)
            with col2:
                if st.button("Проверить", key=f"btn_{key}"):
                    if not answer.strip():
                        st.warning("Введи ответ.")
                    else:
                        with st.spinner("AI проверяет..."):
                            try:
                                result = check_answer_ai(
                                    topic, task["question"],
                                    task["answer"], answer
                                )
                                save_attempt(
                                    student, topic, day["day"],
                                    task["question"], task["answer"],
                                    answer, result["is_correct"],
                                    result.get("feedback", "")
                                )
                                if result["is_correct"]:
                                    st.success(f"✅ {result['feedback']}")
                                else:
                                    st.error(f"❌ {result['feedback']}")
                                    st.session_state.mistakes.append(task["question"])
                            except Exception as e:
                                st.error(f"Ошибка: {e}")

            if st.button("💡 Подсказка", key=f"hint_{key}"):
                st.info(task["hint"])

            st.divider()

        st.markdown("### 🧠 Проверь понимание")
        explanation = st.text_area(
            "Объясни тему своими словами:",
            key=f"expl_{idx}"
        )
        if st.button("Оценить", key=f"eval_{idx}"):
            if len(explanation.strip()) < 10:
                st.warning("Напиши чуть больше.")
            else:
                with st.spinner("AI оценивает..."):
                    try:
                        ev = check_explanation(topic, explanation)
                        st.metric("Понимание", f"{ev['score']}/10")
                        st.write(f"✅ **Понял:** {ev['good']}")
                        st.write(f"⚠️ **Пробелы:** {ev['gaps']}")
                        st.write(f"💡 **Совет:** {ev['advice']}")
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
