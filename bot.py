# -*- coding: utf-8 -*-
import asyncio
import logging
import math
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.storage.memory import MemoryStorage

# ===== НАСТРОЙКИ =====
# Токен теперь читается из переменной окружения (безопасно)TOKEN = os.getenv("BOT_TOKEN")
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise ValueError("BOT_TOKEN не задан в переменных окружения!")
# ВАЖНО: замените токен на новый через @BotFather и удалите эту строку, используйте только переменные окружения!

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== ФАЙЛ ДЛЯ ИСТОРИИ =====
HISTORY_FILE = "history.json"

# ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ РАБОТЫ С ИСТОРИЕЙ =====
def load_history(user_id: int) -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(str(user_id), [])
    except json.JSONDecodeError:
        # Если файл повреждён, начинаем заново
        logger.warning(f"Файл {HISTORY_FILE} повреждён, создаём новый.")
        return []

def save_history_entry(user_id: int, entry: dict):
    data = {}
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}  # если битый, перезаписываем
    uid = str(user_id)
    if uid not in data:
        data[uid] = []
    data[uid].append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        **entry
    })
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ===== КЛАВИАТУРЫ =====
def main_keyboard():
    keyboard = [
        [KeyboardButton(text="🧮 Байес калькуляторы")],
        [KeyboardButton(text="📊 7-дневный мониторинг")],
        [KeyboardButton(text="📔 История")],
        [KeyboardButton(text="📖 Инструкция")],
    ]
    return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)

def cancel_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="❌ Отмена")]],
        resize_keyboard=True
    )

# ===== СОСТОЯНИЯ FSM =====
class BayesStates(StatesGroup):
    waiting_p_a = State()
    waiting_p_b_given_a = State()
    waiting_p_b_given_not_a = State()
    waiting_check_count = State()

class MonitorStates(StatesGroup):
    waiting_for_day = State()

# ===== ХРАНИЛИЩЕ ДАННЫХ МОНИТОРИНГА (в памяти) =====
user_monitor_data: Dict[int, dict] = {}

# ===== МАТЕМАТИЧЕСКИЕ ФУНКЦИИ =====
def bayes(p_a: float, p_b_given_a: float, p_b_given_not_a: float) -> float:
    """P(A|B) = P(B|A)*P(A) / (P(B|A)*P(A) + P(B|~A)*P(~A))"""
    numerator = p_b_given_a * p_a
    denominator = numerator + p_b_given_not_a * (1 - p_a)
    if denominator == 0:
        return 0.0
    return numerator / denominator

def chain_bayes(prior: float, p_b_given_a: float, p_b_given_not_a: float, n: int) -> List[float]:
    """Применяет формулу Байеса n раз (цепочка), возвращает все апостериорные вероятности."""
    results = []
    current = prior
    for _ in range(n):
        current = bayes(current, p_b_given_a, p_b_given_not_a)
        results.append(current)
    return results

def compute_normal_analysis(values: List[float]) -> Optional[dict]:
    """
    Вычисляет:
    - среднее
    - выборочное стандартное отклонение
    - 95% доверительный интервал
    - сравнивает последнее значение с интервалом и определяет тренд
    """
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    z95 = 1.96
    c = z95 * std / math.sqrt(n)
    lo = mean - c
    hi = mean + c
    last = values[-1]
    if last < lo:
        trend = "decrease"
    elif last > hi:
        trend = "increase"
    else:
        trend = "stable"
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "c": c,
        "lo": lo,
        "hi": hi,
        "last": last,
        "trend": trend,
        "values": values,
    }

def format_table(values: List[float], title: str = "Результаты") -> str:
    """Формирует текстовую таблицу с полосами для визуализации.
       Для вероятностей (0..1) умножаем на 10; для оценок (0..10) используем значение напрямую.
       Автоматически определяем по диапазону."""
    lines = []
    lines.append(f"📊 *{title}*")
    lines.append("```")
    lines.append(f"{'№':<4} {'Значение':<10} {'Визуализация'}")
    lines.append("-" * 40)
    # Определяем, вероятности это (0..1) или оценки (0..10)
    max_val = max(values) if values else 1
    if max_val <= 1.0:
        # для вероятностей
        for i, v in enumerate(values, 1):
            filled = int(round(v * 10))
            filled = max(0, min(filled, 10))
            bar = "█" * filled + "░" * (10 - filled)
            lines.append(f"{i:<4} {v:>8.4f}  {bar}")
    else:
        # для оценок 0..10
        for i, v in enumerate(values, 1):
            filled = int(round(v))
            filled = max(0, min(filled, 10))
            bar = "█" * filled + "░" * (10 - filled)
            lines.append(f"{i:<4} {v:>8.2f}  {bar}")
    lines.append("```")
    return "\n".join(lines)

def format_bayes_report(results: List[float], prior: float, p_b_given_a: float, p_b_given_not_a: float) -> str:
    """Формирует отчёт по цепочке Байеса с таблицей."""
    final = results[-1]
    msg = "🧮 *Результат цепочки Байеса*\n\n"
    msg += f"Исходные данные:\n"
    msg += f"P(A) = {prior:.4f} ({prior*100:.1f}%)\n"
    msg += f"P(B|A) = {p_b_given_a:.4f} ({p_b_given_a*100:.1f}%)\n"
    msg += f"P(B|¬A) = {p_b_given_not_a:.4f} ({p_b_given_not_a*100:.1f}%)\n"
    msg += f"Количество проверок: {len(results)}\n\n"
    msg += format_table(results, "Цепочка Байеса (вероятности)")
    msg += f"\n\n✅ *Итоговая вероятность:* {final:.4f} ({final*100:.1f}%)\n"
    if final < 0.1:
        msg += "💚 Риск очень низкий. Беспокоиться не о чем."
    elif final < 0.3:
        msg += "🟡 Вероятность низкая. Проверка не нужна."
    elif final < 0.6:
        msg += "🟠 Средняя вероятность. Достаточно одной проверки."
    else:
        msg += "🔴 Высокая вероятность. Проверка оправдана."
    return msg

def format_normal_report(analysis: dict) -> str:
    """Формирует отчёт по нормальному распределению с таблицей."""
    n = analysis["n"]
    mean = analysis["mean"]
    std = analysis["std"]
    c = analysis["c"]
    lo = analysis["lo"]
    hi = analysis["hi"]
    last = analysis["last"]
    trend = analysis["trend"]
    values = analysis["values"]

    msg = "📐 *Анализ нормального распределения*\n\n"
    msg += format_table(values, "Значения")
    msg += f"\n📊 *Статистика:*\n"
    msg += f"  • Среднее (x̄) = {mean:.4f} ({mean*100:.1f}%)\n"
    msg += f"  • Ст. отклонение (S) = {std:.4f}\n"
    msg += f"  • c = 1.96 × S/√{n} = {c:.4f}\n\n"
    msg += f"📏 *95% доверительный интервал:*\n"
    msg += f"  [{lo:.4f} ; {hi:.4f}]  (в процентах: [{lo*100:.1f}% ; {hi*100:.1f}%])\n\n"
    msg += f"🔍 *Последнее значение:* {last:.4f} ({last*100:.1f}%)\n\n"
    if trend == "decrease":
        msg += (
            "✅ *Результат: ПОЛОЖИТЕЛЬНАЯ ДИНАМИКА*\n"
            f"Последнее значение ({last:.4f}) ниже нижней границы интервала ({lo:.4f}).\n"
            "Уровень тревожности *снижается* — хороший знак! 💚\n"
            "Продолжайте наблюдение."
        )
    elif trend == "increase":
        msg += (
            "⚠️ *Результат: СИГНАЛ ТРЕВОГИ*\n"
            f"Последнее значение ({last:.4f}) выше верхней границы интервала ({hi:.4f}).\n"
            "Уровень тревожности *растёт* — обратите внимание! 🔴\n"
            "Рекомендуется консультация психолога."
        )
    else:
        msg += (
            "🟡 *Результат: СТАБИЛЬНЫЙ УРОВЕНЬ*\n"
            f"Последнее значение ({last:.4f}) находится внутри интервала.\n"
            "Уровень тревожности *стабилен* — продолжайте мониторинг."
        )
    return msg

# ===== ОБРАБОТЧИКИ КОМАНД И СООБЩЕНИЙ =====
# Инициализация бота и диспетчера
bot = Bot(token=TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# ---- Команда /start ----
@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "👋 *Привет!*\n\n"
        "Я — *OCD-Monitor Bot*. Помогаю отслеживать уровень тревожности и оценивать риски с помощью математики.\n\n"
        "🔹 *Байес калькулятор* — оценивает вероятность на основе ваших симптомов.\n"
        "🔹 *7-дневный мониторинг* — ежедневный ввод уровня тревожности (0–10) с полной статистикой.\n"
        "🔹 *История* — все ваши расчёты сохраняются.\n\n"
        "Используйте кнопки меню ниже 👇",
        parse_mode="Markdown",
        reply_markup=main_keyboard()
    )

# ---- Обработка главного меню ----
@dp.message(lambda msg: msg.text in ["🧮 Байес калькуляторы", "📊 7-дневный мониторинг", "📔 История", "📖 Инструкция"])
async def menu_handler(message: Message, state: FSMContext):
    text = message.text

    if text == "🧮 Байес калькуляторы":
        await state.set_state(BayesStates.waiting_p_a)
        await message.answer(
            "🧮 *Байес калькулятор*\n\n"
            "Этот калькулятор поможет вам математически оценить вероятность того, что ваша тревога обоснована.\n\n"
            "**Шаг 1.** Введите *априорную вероятность* P(A) — вашу начальную оценку риска (число от 0 до 1).\n"
            "Например: 0.05 (5%)\n\n"
            "📌 Введите число (например, 0.05):",
            parse_mode="Markdown",
            reply_markup=cancel_keyboard()
        )

    elif text == "📊 7-дневный мониторинг":
        user_id = message.from_user.id
        user_monitor_data[user_id] = {"days": [], "day_number": 1}
        await state.set_state(MonitorStates.waiting_for_day)
        await message.answer(
            "📊 *7-дневный мониторинг*\n\n"
            "Каждый день введите число от **0 до 10**, оценивающее ваш уровень тревожности:\n"
            "0 — полное спокойствие, 10 — сильнейшая тревога.\n\n"
            f"📅 *День 1 из 7*\nВведите число:",
            parse_mode="Markdown",
            reply_markup=cancel_keyboard()
        )

    elif text == "📔 История":
        await show_history(message)
        return

    elif text == "📖 Инструкция":
        await show_instructions(message)
        return

# ---- Инструкция ----
async def show_instructions(message: Message):
    await message.answer(
        "📖 *Инструкция*\n\n"
        "🔹 *Байес калькулятор*\n"
        "Позволяет обновить вероятность риска при появлении новых симптомов.\n"
        "Введите:\n"
        "• P(A) — начальная вероятность (например, 0.05)\n"
        "• P(B|A) — вероятность симптома, если риск есть (например, 0.6)\n"
        "• P(B|¬A) — вероятность симптома, если риска нет (например, 0.1)\n"
        "• Количество проверок (например, 7)\n\n"
        "🔹 *7-дневный мониторинг*\n"
        "Ежедневно вводите уровень тревожности (0–10). Через 7 дней вы получите:\n"
        "• Среднее, дисперсию, стандартное отклонение\n"
        "• 95% доверительный интервал\n"
        "• Байесовскую вероятность ОКР\n"
        "• Диагноз (норма / тревога / кризис)\n\n"
        "🔹 *История*\n"
        "Все ваши расчёты сохраняются и доступны в любое время.\n\n"
        "🔹 *Визуализация*\n"
        "Результаты выводятся в виде таблицы с полосами для наглядности.",
        parse_mode="Markdown",
        reply_markup=main_keyboard()
    )

# ---- История ----
async def show_history(message: Message):
    user_id = message.from_user.id
    history = load_history(user_id)
    if not history:
        await message.answer(
            "📔 История пуста. Сделайте хотя бы один расчёт.",
            reply_markup=main_keyboard()
        )
        return
    msg = "📔 *Ваша история расчётов:*\n\n"
    for i, entry in enumerate(reversed(history[-10:]), 1):
        date = entry.get("date", "")
        text = entry.get("text", "")
        msg += f"*{i}. {date}*\n{text}\n\n"
    await message.answer(msg, parse_mode="Markdown", reply_markup=main_keyboard())

# ---- Отмена (для всех состояний) ----
@dp.message(Command("cancel"))
@dp.message(lambda msg: msg.text == "❌ Отмена")
async def cmd_cancel(message: Message, state: FSMContext):
    await state.clear()
    user_id = message.from_user.id
    if user_id in user_monitor_data:
        del user_monitor_data[user_id]
    await message.answer(
        "❌ Действие отменено.",
        reply_markup=main_keyboard()
    )

# ---- Обработчики Байес-калькулятора ----
@dp.message(BayesStates.waiting_p_a)
async def bayes_get_p_a(message: Message, state: FSMContext):
    try:
        val = float(message.text.replace(",", "."))
        if not (0 < val < 1):
            raise ValueError
        await state.update_data(p_a=val)
        await state.set_state(BayesStates.waiting_p_b_given_a)
        await message.answer(
            f"✅ P(A) = {val:.4f}\n\n"
            "**Шаг 2.** Введите *правдоподобие* P(B|A) — вероятность симптома (B), если риск (A) есть.\n"
            "Например: 0.6 (60%)\n\n"
            "📌 Введите число от 0 до 1:",
            parse_mode="Markdown",
            reply_markup=cancel_keyboard()
        )
    except ValueError:
        await message.answer("❌ Ошибка! Введите число от 0 до 1 (например, 0.05).")

@dp.message(BayesStates.waiting_p_b_given_a)
async def bayes_get_p_b_given_a(message: Message, state: FSMContext):
    try:
        val = float(message.text.replace(",", "."))
        if not (0 <= val <= 1):
            raise ValueError
        await state.update_data(p_b_given_a=val)
        await state.set_state(BayesStates.waiting_p_b_given_not_a)
        await message.answer(
            f"✅ P(B|A) = {val:.4f}\n\n"
            "**Шаг 3.** Введите *ложную тревогу* P(B|¬A) — вероятность симптома, если риска нет.\n"
            "Например: 0.1 (10%)\n\n"
            "📌 Введите число от 0 до 1:",
            parse_mode="Markdown",
            reply_markup=cancel_keyboard()
        )
    except ValueError:
        await message.answer("❌ Ошибка! Введите число от 0 до 1.")

@dp.message(BayesStates.waiting_p_b_given_not_a)
async def bayes_get_p_b_given_not_a(message: Message, state: FSMContext):
    try:
        val = float(message.text.replace(",", "."))
        if not (0 <= val <= 1):
            raise ValueError
        await state.update_data(p_b_given_not_a=val)
        await state.set_state(BayesStates.waiting_check_count)
        await message.answer(
            f"✅ P(B|¬A) = {val:.4f}\n\n"
            "**Шаг 4.** Введите *количество проверок* (целое число от 1 до 20).\n"
            "Например: 7 (если вы проверили 7 раз).\n\n"
            "📌 Введите число:",
            parse_mode="Markdown",
            reply_markup=cancel_keyboard()
        )
    except ValueError:
        await message.answer("❌ Ошибка! Введите число от 0 до 1.")

@dp.message(BayesStates.waiting_check_count)
async def bayes_get_check_count(message: Message, state: FSMContext):
    try:
        n = int(message.text.strip())
        if not (1 <= n <= 20):
            raise ValueError
    except ValueError:
        await message.answer("❌ Ошибка! Введите целое число от 1 до 20.")
        return

    data = await state.get_data()
    p_a = data["p_a"]
    p_b_given_a = data["p_b_given_a"]
    p_b_given_not_a = data["p_b_given_not_a"]

    # Цепочка Байеса
    results = chain_bayes(p_a, p_b_given_a, p_b_given_not_a, n)

    # Отчёт по Байесу
    bayes_report = format_bayes_report(results, p_a, p_b_given_a, p_b_given_not_a)
    await message.answer(bayes_report, parse_mode="Markdown")

    # Если проверок >= 2, делаем нормальный анализ
    if n >= 2:
        analysis = compute_normal_analysis(results)
        if analysis:
            normal_report = format_normal_report(analysis)
            await message.answer(normal_report, parse_mode="Markdown")

    # Сохраняем в историю
    entry_text = (
        f"Байес: P(A)={p_a:.4f}, P(B|A)={p_b_given_a:.4f}, P(B|¬A)={p_b_given_not_a:.4f}, "
        f"проверок={n}, итоговая P={results[-1]:.4f} ({results[-1]*100:.1f}%)"
    )
    save_history_entry(message.from_user.id, {"text": entry_text})

    await state.clear()
    await message.answer(
        "✅ Расчёт завершён. Выберите следующее действие:",
        reply_markup=main_keyboard()
    )

# ---- Обработчики 7-дневного мониторинга ----
@dp.message(MonitorStates.waiting_for_day)
async def monitor_process_day(message: Message, state: FSMContext):
    user_id = message.from_user.id
    text = message.text.strip()

    try:
        value = float(text)
        if not (0 <= value <= 10):
            await message.answer("❌ Число должно быть от 0 до 10. Попробуйте ещё раз.")
            return
    except ValueError:
        await message.answer("❌ Это не число. Введите число от 0 до 10.")
        return

    # Сохраняем
    if user_id not in user_monitor_data:
        user_monitor_data[user_id] = {"days": [], "day_number": 1}
    data = user_monitor_data[user_id]
    data["days"].append(value)
    current_day = data["day_number"]

    if current_day == 7:
        # Все дни собраны
        days = data["days"]
        result_text = compute_monitor_results(days)
        await message.answer(result_text, parse_mode="Markdown")
        # Сохраняем в историю
        save_history_entry(user_id, {"text": f"Мониторинг 7 дней: {days}"})
        # Очищаем
        del user_monitor_data[user_id]
        await state.clear()
        await message.answer(
            "✅ Мониторинг завершён. Выберите действие:",
            reply_markup=main_keyboard()
        )
    else:
        data["day_number"] = current_day + 1
        await message.answer(
            f"📅 *День {current_day + 1} из 7*\nВведите число от 0 до 10:",
            parse_mode="Markdown"
        )

# ---- Функция расчёта результатов мониторинга (исправлена) ----
def compute_monitor_results(days: List[float]) -> str:
    n = len(days)
    mean = sum(days) / n
    variance = sum((x - mean) ** 2 for x in days) / (n - 1) if n > 1 else 0
    std_dev = math.sqrt(variance)
    z = 1.96
    margin = z * std_dev / math.sqrt(n)
    ci_low = mean - margin
    ci_high = mean + margin

    # === ИСПРАВЛЕНА БАЙЕСОВСКАЯ ВЕРОЯТНОСТЬ ===
    prior = 0.02            # априорная вероятность ОКР (фиксирована)
    likelihood = 0.6        # P(симптом | ОКР)
    false_positive = 0.1    # P(симптом | ¬ОКР) – теперь явно задано
    # Правильная полная вероятность симптома:
    evidence = likelihood * prior + false_positive * (1 - prior)
    posterior = (likelihood * prior) / evidence if evidence != 0 else 0
    # =========================================

    last_value = days[-1]
    if last_value < ci_low:
        diagnosis = "✅ Норма (положительная динамика)"
    elif last_value > ci_high:
        diagnosis = "⚠️ Тревожный сигнал – рекомендуется консультация"
    else:
        diagnosis = "🟡 В пределах нормы – продолжайте наблюдение"

    # Таблица с визуализацией (теперь корректно отображает 0–10)
    table = format_table(days, "Ежедневные значения")

    report = (
        "📊 *Результаты 7-дневного мониторинга*\n\n"
        f"{table}\n\n"
        f"📈 *Среднее:* {mean:.2f}\n"
        f"📉 *Стандартное отклонение:* {std_dev:.2f}\n"
        f"📊 *Дисперсия:* {variance:.2f}\n"
        f"🔍 *95% доверительный интервал:* [{ci_low:.2f} ; {ci_high:.2f}]\n"
        f"🧮 *Байесовская вероятность ОКР:* {posterior*100:.2f}%\n\n"
        f"📌 *Диагноз:* {diagnosis}"
    )
    return report

# ===== ЗАПУСК =====
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
