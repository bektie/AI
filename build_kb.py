"""Create the knowledge base for lab #1 in MiniES 2.0 format.

The probabilities below are DIDACTIC EXPERT ESTIMATES, not observed statistics.
Each attribute is a simplified indicator, not an exclusive/absolute property.
"""
from pathlib import Path

LANGUAGES = ["Python", "Java", "JavaScript", "C++", "C#", "Go", "Rust", "Kotlin", "Swift", "PHP"]

# Question and high-likelihood / moderately-likely languages.
FEATURES = [
 ("Язык является стандартным средством программирования в браузере?", ["JavaScript"], []),
 ("Язык часто выбирают для машинного обучения и анализа данных?", ["Python"], []),
 ("Язык широко применяется для нативной разработки Android?", ["Java", "Kotlin"], []),
 ("Язык является основным выбором для нативных приложений iOS?", ["Swift"], []),
 ("Язык обычно используется для написания игрового кода в Unity?", ["C#"], []),
 ("Язык широко применяется в игровых движках уровня Unreal Engine?", ["C++"], ["Rust"]),
 ("Язык часто используют для сайтов на WordPress?", ["PHP"], []),
 ("Программы на этом языке обычно выполняются на JVM?", ["Java", "Kotlin"], []),
 ("Язык является основным в экосистеме .NET?", ["C#"], []),
 ("В языке применяются владение и проверка заимствований при компиляции?", ["Rust"], []),
 ("В языке встроены горутины как механизм конкурентного выполнения?", ["Go"], []),
 ("Для языка характерна динамическая типизация?", ["Python", "JavaScript", "PHP"], []),
 ("Язык подходит для написания системного программного обеспечения?", ["C++", "Rust"], ["Go"]),
 ("Язык часто применяют для высоконагруженных сетевых сервисов?", ["Go", "Java"], ["Rust", "C#", "Kotlin"]),
 ("Язык удобен для автоматизации с помощью коротких сценариев?", ["Python"], ["JavaScript", "PHP"]),
 ("Для языка характерны прямые операции с указателями и ручное управление памятью?", ["C++"], ["Rust"]),
 ("Язык распространен в крупных корпоративных приложениях на JVM?", ["Java"], ["Kotlin"]),
 ("В экосистеме языка используется SwiftUI?", ["Swift"], []),
 ("Для создания Android-приложений используется Kotlin Multiplatform?", ["Kotlin"], []),
 ("Язык широко применяется с серверным фреймворком Laravel?", ["PHP"], []),
 ("В типичном коде языка проверка типов преимущественно статическая?", ["Java", "C++", "C#", "Go", "Rust", "Kotlin", "Swift"], []),
 ("Язык может непосредственно компилироваться в машинный код без обязательной виртуальной машины?", ["C++", "Go", "Rust", "Swift"], []),
]


def generate() -> str:
    text = [
        "Экспертная система: выбор языка программирования по признакам.",
        "Лабораторная работа 1. Тема 22. Учебные вероятности заданы экспертно, не по статистике.",
        "Версия 1.0. Признаки не являются исключительными; вывод носит рекомендательный характер.",
        "",
    ]
    text += [q for q, _strong, _mid in FEATURES]
    text += [""]
    prior = 1.0 / len(LANGUAGES)
    for lang in LANGUAGES:
        fields = [lang, f"{prior:.4f}"]
        for i, (_question, strong, middle) in enumerate(FEATURES):
            py = 0.92 if lang in strong else 0.65 if lang in middle else 0.08
            # Negative-hypothesis response probability estimated from other languages' attributes.
            other = [l for l in LANGUAGES if l != lang]
            raw = [0.92 if l in strong else 0.65 if l in middle else 0.08 for l in other]
            pn = sum(raw) / len(raw)
            fields += [str(i), f"{py:.5f}", f"{pn:.5f}"]
        text.append(", ".join(fields))
    return "\n".join(text) + "\n"


def main():
    base = Path(__file__).resolve().parent
    content = generate()
    (base / 'БЗ_языки_программирования_UTF8.txt').write_text(content, encoding='utf-8')
    (base / 'БЗ_языки_программирования_МЭС_cp1251.txt').write_bytes(content.replace('\n', '\r\n').encode('cp1251'))
    print(f"Созданы две копии базы: {len(FEATURES)} вопросов и {len(LANGUAGES)} гипотез")

if __name__ == '__main__':
    main()
