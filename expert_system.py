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


def evaluate(kb: KnowledgeBase, answers: Dict[int, bool]) -> Dict[str, float]:
    probs = {}
    for hypothesis in kb.hypotheses:
        p = hypothesis.prior
        for index, answer_yes in answers.items():
            if index in hypothesis.rules:
                py, pn = hypothesis.rules[index]
                p = bayes_update(p, py, pn, answer_yes)
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
    print('Сценарий синтетический: ответ «да» при P(да|H) >= 0.5.')
    while len(answers) < len(kb.questions):
        idx = next_question(kb, set(answers), evaluate(kb, answers))
        if idx is None: break
        answer = hypothesis.rules[idx][0] >= 0.5
        answers[idx] = answer
        print(f'  {idx+1:02d}. {kb.questions[idx]} -> {"да" if answer else "нет"}')
    print('Результат (независимые вероятностные оценки):')
    show_top(evaluate(kb, answers))


def interactive(kb):
    print(kb.description)
    print(f'Вопросов: {len(kb.questions)}. Языков: {len(kb.hypotheses)}.')
    print('Ответы: д = да, н = нет, ? = не знаю, стоп = результат.')
    answers = {}; asked=set()
    while len(asked)<len(kb.questions):
        idx = next_question(kb, asked, evaluate(kb,answers))
        if idx is None: break
        print(f'\nВопрос {len(asked)+1}/{len(kb.questions)}: {kb.questions[idx]}')
        answer = input('Ответ [д/н/?/стоп]: ').strip().lower()
        if answer in ('стоп','stop','q'):
            break
        if answer in ('д','да','yes','y'):
            answers[idx]=True
        elif answer in ('н','нет','no','n'):
            answers[idx]=False
        elif answer not in ('?','не знаю','не знаю.',''):
            print('Введите д, н, ? или стоп.'); continue
        asked.add(idx)
        print('Топ-3 гипотезы:')
        show_top(evaluate(kb,answers), 3)
    print('\nИТОГ:')
    show_top(evaluate(kb,answers), len(kb.hypotheses))
    print(f'Обработано ответов: {len(answers)}; пропущено: {len(asked)-len(answers)}.')
    print('Важно: оценки не нормированы и не являются статистикой рынка.')


def main():
    parser=argparse.ArgumentParser(description='ЛР №2: экспертная система языков программирования')
    parser.add_argument('--kb',type=Path, default=Path(__file__).with_name('БЗ_языки_программирования_UTF8.txt'))
    parser.add_argument('--demo',help='Автоматически пройти анкету для заданного языка (например Python)')
    parser.add_argument('--list',action='store_true',help='Показать доступные гипотезы')
    args=parser.parse_args()
    kb=load_knowledge_base(args.kb)
    if args.list: print(', '.join(h.name for h in kb.hypotheses))
    elif args.demo: demo(kb,args.demo)
    else: interactive(kb)

if __name__ == '__main__':
    main()
