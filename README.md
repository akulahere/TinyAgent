# TinyAgent

Учебный агент по *An Illustrated Guide to AI Agents* — Maarten Grootendorst и Jay Alammar.
[Материалы авторов](https://github.com/HandsOnLLM/An-Illustrated-Guide-To-AI-Agents).

## Реализовано

| Глава | Код и примеры |
| --- | --- |
| 2 — LLM | `llm.py`, `trajectory.py`, базовый `demo.ipynb` |
| 3 — Reasoning | `temperature` в LLM; prompting, self-consistency, Best-of-N и native reasoning в `chapter3.ipynb` |
| 4 — Memory | `memory.py`: Memory, TrimmingMemory, SummarizationMemory, RAGMemory; EmbeddingModel в `llm.py`; `chapter4.ipynb` |

Агент выполняет один вызов генерации на запрос и записывает результат в траекторию.
SummarizationMemory дополнительно вызывает LLM для сводки после ответа.
Инструменты и автономный цикл относятся к следующим главам.

## Запуск

Основной код и тесты используют только стандартную библиотеку Python 3.12+.
По умолчанию ожидается локальный Ollama с API `http://localhost:11434/v1`.
Перед запуском загрузите выбранные модели самостоятельно: `gemma4:e4b` для генерации,
`embeddinggemma` для RAG. Примеры не скачивают модели автоматически.

```python
from agent import TinyAgent
from llm import LLM
from memory import Memory

llm = LLM(model="gemma4:e4b", temperature=0)
agent = TinyAgent(llm=llm, memory=Memory())
print(agent.run("My name is Sarah and I live in Lisbon."))
print(agent.run("What is my name and where do I live?"))
print(agent.memory.get_messages())
print(agent.trajectory.runs)
```

Для другого сервера передайте `base_url` и при необходимости `api_key` в LLM и EmbeddingModel.
`think=False` отправляет `reasoning_effort="none"`; `think=True` оставляет выбор режима backend.
Поддержка этого параметра и поля `reasoning` зависит от модели и сервера.

## Notebook-примеры

Из корня проекта:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebooks.txt
.venv/bin/jupyter lab
```

Откройте `demo.ipynb`, `chapter3.ipynb` или `chapter4.ipynb`, выберите ядро
созданного окружения и запускайте ячейки по порядку. TrajectoryViewer отображается
в notebook. Файлы сохраняются без ответов модели и истории запусков.

В главе 3 книги prompting показан на `gemma3:12b`. В нашем notebook по умолчанию
выбрана `gemma4:e4b` с `think=False`; модель и число попыток меняются в первой ячейке.
Best-of-N проверяет JSON с ответами на набор примеров вместо исполнения
сгенерированной функции: это демонстрация отбора по оценке, не тест универсального конвертера.

## Поведение памяти

- **Memory:** хранит всю историю и передаёт её при следующем запросе.
- **TrimmingMemory:** сохраняет system-инструкции и последние `max_turns` ходов,
  начиная с user-сообщения. Текущий незавершённый ход тоже учитывается.
- **SummarizationMemory:** обновляет сводку после каждого assistant-сообщения.
  Сводка отделена от исходных system-инструкций. При пустой сводке история остаётся;
  при ошибке запроса исключение передаётся вызывающему коду, история и шаг траектории сохраняются.
- **RAGMemory:** индексирует переданные документы, выбирает `top_k=3` по cosine
  similarity и добавляет их в пользовательский запрос. В траектории остаётся исходный запрос.

В отличие от упрощённого среза последних четырёх сообщений в книге, обрезка сохраняет
целые ходы. `TinyAgent(llm=...)` создаёт собственную Memory по умолчанию, поэтому
старый пример главы 2 продолжает работать.

Все варианты работают в RAM: «долгосрочная память» здесь означает извлечение из
внешних документов, а не сохранение на диск. История и сводка не имеют жёсткого
лимита токенов. RAG использует простой поиск без порога релевантности и векторной БД.
MemoryBank, agentic RAG и обучение reasoning-моделей в этих главах обсуждаются
теоретически и не входят в реализацию TinyAgent.

## Проверки

```sh
python3 -B -m unittest discover -s tests -v
```

Тесты работают без сервера и загрузки моделей: проверяют HTTP-контракт через mock,
историю диалога, обрезку, суммаризацию, RAG, траекторию и код notebook-примеров.
Для проверки реального LLM запускайте notebook. Раздел RAG требует отдельной
embedding-модели; успешные offline-тесты не подтверждают качество её поиска.

Повторяемая живая проверка памяти (запускается явно, расходует ресурсы локальной модели):

```sh
python3 -B scripts/live_check.py
python3 -B scripts/live_check.py --rag  # требуется embeddinggemma
```

Это небольшие smoke-проверки по ответам модели, не полноценная оценка качества.
GitHub Actions запускает offline-тесты на Python 3.12, 3.13 и 3.14.
