import os
import json
import sqlite3
import streamlit as st
import pandas as pd
from datetime import datetime
from openai import OpenAI

# ============================================================
# 1. ПОЛУЧЕНИЕ КЛЮЧЕЙ (работает и локально, и на Cloud)
# ============================================================

# На Streamlit Cloud ключи лежат в st.secrets
# Локально — в переменных окружения из .env
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
except (KeyError, FileNotFoundError):
    # Локальный запуск: ключи из .env через python-dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not api_key:
    st.error(
        "❌ Не найден OPENAI_API_KEY.\n\n"
        "**На Streamlit Cloud:** добавь его в Manage app → Settings → Secrets.\n\n"
        "**Локально:** создай файл `.env` с ключом."
    )
    st.stop()

client = OpenAI(api_key=api_key, base_url=base_url)
MODEL = model_name
DB = "progress.db"


# ============================================================
# 2. БАЗА ДАННЫХ (SQLite)
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


def get_last_course(student, topic):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "SELECT plan_json FROM courses WHERE student=? AND topic=? ORDER BY id DESC LIMIT 1",
        (student, topic),
    )
    row = c.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


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
# 3. AI-ФУНКЦИИ
# ============================================================

def ai_json(prompt: str, system: str = "Ты — опытный школьный учитель.",
            retries: int = 3) -> dict:
    """Запрос к LLM с гарантией JSON-ответа и повторами."""
    last_err = None
    for _ in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"LLM не вернул валидный JSON: {last_err}")


def build_plan(topic: str, grade: int, days: int = 7) -> dict:
    """Составляет план курса на N дней."""
    prompt = f"""Составь учебный план по теме «{topic}» для ученика {grade} класса.
Курс рассчитан на {days} дней, по 20–30 минут в день.

Верни строго JSON:
{{
  "topic": "{topic}",
  "grade": {grade},
  "prerequisites": ["тема1", "тема2"],
  "days": [
    {{
      "day": 1,
      "title": "Короткое название дня",
      "goal": "Что ученик освоит за день",
      "theory": "Объяснение на 3-5 предложений простым языком",
      "example": "Разобранный пример",
      "tasks": [
        {{"question": "Условие", "answer": "Правильный ответ", "hint": "Подсказка"}}
      ]
    }}
  ]
}}
В каждом дне ровно 3 задачи. Задачи — разные по сложности.
Все ответы — точные, проверяемые. Без воды."""
    return ai_json(prompt)


def generate_task(topic: str, day_title: str, goal: str,
                  previous_mistakes: list) -> dict:
    """Генерирует одну задачу с учётом ошибок ученика."""
    mistakes_text = "; ".join(previous_mistakes[-5:]) or "нет"
    prompt = f"""Тема: {topic}
День: {day_title}
Цель дня: {goal}
Прошлые ошибки ученика: {mistakes_text}

Сгенерируй ОДНУ новую задачу, похожую по типу на прошлые ошибки,
если они есть. Иначе — новую задачу по теме дня.

Верни JSON:
{{"question": "...", "answer": "...", "hint": "...", "difficulty": 1-5}}"""
    return ai_json(prompt)


def check_answer_ai(topic: str, question: str, correct: str, given: str) -> dict:
    """Проверяет ответ, допуская разные формы записи."""
    prompt = f"""Тема: {topic}
Вопрос: {question}
Правильный ответ: {correct}
Ответ ученика: {given}

Проверь, верен ли ответ ученика по смыслу (разные формы записи допустимы).
Верни JSON:
{{"is_correct": true,
  "feedback": "Короткий комментарий ученику (1-2 предложения)",
  "mistake_type": "тип ошибки или null"}}"""
    return ai_json(prompt)


def check_explanation(topic: str, student_text: str) -> dict:
    """Проверяет понимание через объяснение своими словами."""
    prompt = f"""Тема: {topic}
Ученик объясняет своими словами:
"{student_text}"

Оцени понимание.
Верни JSON:
{{"score": 0,
  "good": "что ученик понял верно",
  "gaps": "чего не хватает",
  "advice": "1 совет"}}"""
    return ai_json(prompt)


# ============================================================
# 4. ИНТЕРФЕЙС
# ============================================================

st.set_page_config(page_title="Тема за 7 дней · AI", page_icon="🧠", layout="wide")
init_db()

st.title("🧠 Тема за 7 дней · AI-версия")
st.caption("Введи любую школьную тему — нейросеть составит персональный курс")


# --- Боковая панель ---
with st.sidebar:
    st.header("Настройки")
    student = st.text_input("Имя ученика", value="Ученик")
    grade = st.selectbox("Класс", list(range(1, 12)), index=7)

    st.subheader("Быстрый выбор темы")
    quick_topics = [
        "Квадратные уравнения",
        "Производная функции",
        "Закон Ома",
        "Причастие и деепричастие",
        "Строение атома",
        "Теорема Пифагора",
        "Химические реакции",
        "Синтаксис сложного предложения",
    ]
    picked = st.selectbox("Или выбери готовую:", ["— своя тема —"] + quick_topics)
    topic = st.text_input("Своя тема:", value="" if picked == "— своя тема —" else picked)

    days = st.slider("Сколько дней в курсе", 3, 14, 7)

    if st.button("🚀 Составить курс", type="primary"):
        if not topic.strip():
            st.error("Введи тему.")
        else:
            with st.spinner("Нейросеть составляет план курса..."):
                plan = build_plan(topic, grade, days)
                save_course(student, topic, plan)
                st.session_state.plan = plan
                st.session_state.topic = topic
                st.session_state.mistakes = []
                st.rerun()

    if st.button("📊 Моя статистика"):
        st.session_state.show_stats = True


# --- Статистика ---
if st.session_state.get("show_stats"):
    st.subheader("📊 Ваш прогресс")
    if "topic" in st.session_state:
        rows = get_progress(student, st.session_state.topic)
        if rows:
            df = pd.DataFrame(rows, columns=["День", "Всего задач", "Правильно"])
            df["Процент"] = (df["Правильно"] / df["Всего задач"] * 100).round(1)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Пока нет решённых задач.")


# --- Если курса ещё нет ---
if "plan" not in st.session_state:
    st.info("👈 Введи тему в боковой панели и нажми «Составить курс»")
    st.stop()

plan = st.session_state.plan
topic = st.session_state.topic

st.header(f"📚 {plan['topic']} · {plan['grade']} класс")

# Предварительные темы
with st.expander("🔍 Что стоит вспомнить перед курсом"):
    for p in plan.get("prerequisites", []):
        st.write(f"• {p}")

# Вкладки по дням
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
                            result = check_answer_ai(
                                topic, task["question"], task["answer"], answer
                            )
                        save_attempt(
                            student, topic, day["day"], task["question"],
                            task["answer"], answer, result["is_correct"],
                            result.get("feedback", ""),
                        )
                        if result["is_correct"]:
                            st.success(f"✅ {result['feedback']}")
                        else:
                            st.error(f"❌ {result['feedback']}")
                            st.session_state.mistakes.append(task["question"])

            if st.button("💡 Подсказка", key=f"hint_{key}"):
                st.info(task["hint"])

            st.divider()

        # Проверка понимания
        st.markdown("### 🧠 Проверь понимание")
        explanation = st.text_area(
            "Объясни тему своими словами (2-3 предложения):",
            key=f"expl_{idx}",
        )
        if st.button("Оценить", key=f"eval_{idx}"):
            if len(explanation.strip()) < 10:
                st.warning("Напиши чуть больше.")
            else:
                with st.spinner("AI оценивает..."):
                    ev = check_explanation(topic, explanation)
                st.metric("Понимание", f"{ev['score']}/10")
                st.write(f"✅ **Понял:** {ev['good']}")
                st.write(f"⚠️ **Пробелы:** {ev['gaps']}")
                st.write(f"💡 **Совет:** {ev['advice']}")


# --- Адаптивная доп. задача ---
st.divider()
st.subheader("🎯 Дополнительная задача (адаптивная)")
st.caption("Нейросеть сгенерирует задачу с учётом ваших ошибок")

if st.button("Сгенерировать задачу"):
    day = plan["days"][0]
    with st.spinner("Генерируем..."):
        extra = generate_task(
            topic, day["title"], day["goal"],
            st.session_state.get("mistakes", []),
        )
    st.session_state.extra_task = extra

if "extra_task" in st.session_state:
    extra = st.session_state.extra_task
    st.markdown(f"**{extra['question']}**")
    st.caption(f"Сложность: {extra.get('difficulty', '?')}/5")
    a = st.text_input("Ответ:", key="extra_ans")
    if st.button("Проверить доп. задачу"):
        if a.strip():
            r = check_answer_ai(topic, extra["question"], extra["answer"], a)
            if r["is_correct"]:
                st.success(f"✅ {r['feedback']}")
            else:
                st.error(f"❌ {r['feedback']}")
                st.info(f"💡 Подсказка: {extra['hint']}")
