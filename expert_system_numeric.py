"""Lab #2: consultation engine loading the independent MiniES 2.0 KB from lab #1.
Python 3.9+, standard library only.

The one-vs-rest Bayesian probabilities are NOT mutually exclusive: their sum
need not equal one. Feature probabilities were entered as expert estimates.

Consultation accepts confidence coefficients from -5 (definitely not) to
+5 (definitely yes), with 0 = unknown. For partial confidence, the model
interpolates Bayesian evidence in LOG-ODDS space. This is an educational
implementation, not a verified reproduction of MiniES's proprietary mapping.
"""
import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Hypothesis:
    name: str
    prior: float
    rules: Dict[int, Tuple[float, float]]  # question -> P(yes|H), P(yes|not H)


@dataclass
class KnowledgeBase:
    description: str
    questions: List[str]
    hypotheses: List[Hypothesis]


def load_knowledge_base(path: Path) -> KnowledgeBase:
    raw = path.read_bytes()
    decoded = None
    for encoding in ('utf-8-sig', 'cp1251'):
        try:
            decoded = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            pass
    if decoded is None:
        raise ValueError('Не удалось распознать кодировку файла БЗ')
    chunks = decoded.replace('\r\n','\n').strip().split('\n\n', 2)
    if len(chunks) != 3:
        raise ValueError('Ожидались три секции БЗ, разделённые пустыми строками')
    description, question_text, rule_text = chunks
    questions = [x.strip() for x in question_text.splitlines() if x.strip()]
    if not questions:
        raise ValueError('Список свидетельств пуст')
    hypotheses = []
    for line in rule_text.splitlines():
        if not line.strip():
            continue
        parts = [c.strip() for c in line.split(',')]
        if len(parts) < 5 or (len(parts) - 2) % 3:
            raise ValueError(f'Некорректная строка гипотезы: {line[:60]}')
        name = parts[0]
        prior = float(parts[1])
        if not name or not 0 < prior < 1 or not math.isfinite(prior):
            raise ValueError(f'Недопустимые имя или априорная вероятность: {name}')
        rules = {}
        for i in range(2, len(parts), 3):
            idx = int(parts[i]); py = float(parts[i+1]); pn = float(parts[i+2])
            if idx in rules or not 0 <= idx < len(questions):
                raise ValueError(f'Ошибка номера свидетельства: {idx}')
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in (py, pn)):
                raise ValueError(f'Недопустимые вероятности для {name}')
            rules[idx] = py, pn
        hypotheses.append(Hypothesis(name, prior, rules))
    if not hypotheses:
        raise ValueError('Не найдено гипотез')
    return KnowledgeBase(description, questions, hypotheses)


def bayes_update(prior: float, py: float, pn: float, answer_yes: bool) -> float:
    if not answer_yes:
        py, pn = 1 - py, 1 - pn
    numerator = py * prior
    denominator = numerator + pn * (1 - prior)
    return numerator / denominator if denominator > 0 else prior


def confidence_update(prior: float, py: float, pn: float, confidence: float) -> float:
    """Bayesian update for an answer on the [-5, +5] confidence scale.

    c = +5: use ordinary Bayes for evidence E (yes).
    c = -5: use ordinary Bayes for not E (no).
    c =  0: leave the hypothesis unchanged (unknown).
    0 < |c| < 5: multiply prior odds by likelihood_ratio ** (|c|/5).
    Partial confidence is an explicit teaching-model assumption, NOT a
    guarantee of numerical equivalence to the original MiniES 2.0 program.
    """
    if not math.isfinite(confidence) or not -5 <= confidence <= 5:
        raise ValueError('Коэффициент уверенности должен быть от -5 до +5')
    if not 0 < prior < 1 or not all(math.isfinite(x) and 0 <= x <= 1 for x in (py, pn)):
        raise ValueError('Некорректные вероятности')
    if confidence == 0:
        return prior

    # Use bounded probabilities to avoid log(0). The supplied KB uses
    # non-extreme feature probabilities, so this clipping does not affect it.
    eps = 1e-12
    py = min(max(py, eps), 1 - eps)
    pn = min(max(pn, eps), 1 - eps)
    if confidence > 0:
        log_lr = math.log(py) - math.log(pn)
    else:
        log_lr = math.log1p(-py) - math.log1p(-pn)

    weight = abs(confidence) / 5.0
    log_odds = math.log(prior) - math.log1p(-prior) + weight * log_lr
    # Numerically stable logistic transformation.
    if log_odds >= 0:
        e = math.exp(-log_odds)
        return 1 / (1 + e)
    e = math.exp(log_odds)
    return e / (1 + e)


def evaluate(kb: KnowledgeBase, answers: Dict[int, float]) -> Dict[str, float]:
    """Return one-vs-rest probabilities; values need not sum to 100%."""
    probs = {}
    for hypothesis in kb.hypotheses:
        p = hypothesis.prior
        for index, confidence in answers.items():
            if index in hypothesis.rules:
                py, pn = hypothesis.rules[index]
                # Preserve compatibility with earlier yes/no callers/tests.
                if isinstance(confidence, bool):
                    confidence = 5 if confidence else -5
                p = confidence_update(p, py, pn, float(confidence))
        probs[hypothesis.name] = p
    return probs


def next_question(kb: KnowledgeBase, asked: set, probs: Dict[str,float]):
    """Pick a discriminative question using weighted variance of P(yes|H)."""
    total = sum(probs.values())
    weights = {k: (v/total if total else 1/len(probs)) for k,v in probs.items()}
    best_idx = None; best_score = -1
    for idx, _question in enumerate(kb.questions):
        if idx in asked:
            continue
        values = [(weights[h.name], h.rules[idx][0]) for h in kb.hypotheses if idx in h.rules]
        if not values:
            continue
        sw = sum(w for w, _ in values)
        mean = sum(w*p for w,p in values)/sw
        variance = sum(w*(p-mean)**2 for w,p in values)/sw
        if variance > best_score:
            best_idx, best_score = idx, variance
    return best_idx


def show_top(probs, n=5):
    for name,p in sorted(probs.items(),key=lambda kv: kv[1], reverse=True)[:n]:
        print(f'  {name:<12} {p*100:6.2f}%')


def demo(kb, language):
    hypothesis = next((h for h in kb.hypotheses if h.name.lower()==language.lower()),None)
    if hypothesis is None:
        raise ValueError(f'Неизвестный язык {language}. Доступны: {", ".join(h.name for h in kb.hypotheses)}')
    answers = {}
    print('Демонстрация для профиля:', hypothesis.name)
    print('Синтетический сценарий: +5 при P(да|H) >= 0.5, иначе -5.')
    while len(answers) < len(kb.questions):
        idx = next_question(kb, set(answers), evaluate(kb, answers))
        if idx is None: break
        answer = 5 if hypothesis.rules[idx][0] >= 0.5 else -5
        answers[idx] = answer
        print(f'  {idx+1:02d}. {kb.questions[idx]} -> {answer:+d}')
    print('Результат (независимые вероятностные оценки):')
    show_top(evaluate(kb, answers))


def interactive(kb):
    print(kb.description)
    print(f'\nВопросов: {len(kb.questions)}. Языков: {len(kb.hypotheses)}.')
    print('Шкала ответов: -5 = точно нет, 0 = не знаю, +5 = точно да.')
    print('Промежуточные числа (-4, -3, ..., +4) означают степень уверенности.')
    print('Можно вводить дробные значения, например 2.5 или 2,5.')
    print('Введите «стоп», чтобы завершить консультацию досрочно.\n')
    answers = {}
    asked = set()
    while len(asked) < len(kb.questions):
        idx = next_question(kb, asked, evaluate(kb, answers))
        if idx is None:
            break
        print(f'\nВопрос {len(asked)+1}/{len(kb.questions)}: {kb.questions[idx]}')
        raw = input('Уверенность [-5..+5 / стоп]: ').strip().lower()
        if raw in ('стоп', 'stop', 'q', 'выход'):
            break
        try:
            confidence = float(raw.replace(',', '.'))
            if not math.isfinite(confidence) or not -5 <= confidence <= 5:
                raise ValueError()
        except ValueError:
            print('Ошибка: введите число от -5 до +5 (например -3, 0, 4.5) или «стоп».')
            continue
        answers[idx] = confidence
        asked.add(idx)
        if confidence == 0:
            print('Ответ «не знаю»: вероятности не изменились.')
        print('Текущий рейтинг (топ-3):')
        show_top(evaluate(kb, answers), 3)

    print('\n' + '='*45)
    print('РЕЗУЛЬТАТ КОНСУЛЬТАЦИИ')
    print('='*45)
    show_top(evaluate(kb, answers), len(kb.hypotheses))
    print(f'Задано вопросов: {len(asked)}; содержательных ответов: '
          f'{sum(value != 0 for value in answers.values())}; '
          f'ответов «не знаю»: {sum(value == 0 for value in answers.values())}.')
    print('Вероятности вычисляются для каждого языка отдельно и не обязаны суммироваться до 100%.')
    print('Промежуточные ответы учитываются по учебной модели ослабления байесовских свидетельств.')


def main():
    parser=argparse.ArgumentParser(description='ЛР №2: экспертная система языков программирования')
    local_path = Path(__file__).with_name('БЗ_языки_программирования_UTF8.txt')
    if not local_path.exists():
        local_path = Path(__file__).with_name('programming_languages.mkb')
    parser.add_argument('--kb', type=Path, default=local_path,
                        help='Путь к БЗ из лабораторной №1 (UTF-8 или CP1251)')
    parser.add_argument('--demo',help='Автоматически пройти анкету для заданного языка (например Python)')
    parser.add_argument('--list',action='store_true',help='Показать доступные гипотезы')
    args=parser.parse_args()
    kb=load_knowledge_base(args.kb)
    if args.list: print(', '.join(h.name for h in kb.hypotheses))
    elif args.demo: demo(kb,args.demo)
    else: interactive(kb)

if __name__ == '__main__':
    main()
