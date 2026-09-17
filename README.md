# TinyAgent

Учебный агент по *An Illustrated Guide to AI Agents* — Maarten Grootendorst и Jay Alammar.
[Материалы авторов](https://github.com/HandsOnLLM/An-Illustrated-Guide-To-AI-Agents).

## Реализовано

| Глава | Код и примеры |
| --- | --- |
| 2 — LLM | `llm.py`, `trajectory.py`, базовый `demo.ipynb` |
| 3 — Reasoning | `temperature` в LLM; prompting, self-consistency, Best-of-N и native reasoning в `chapter3.ipynb` |
| 4 — Memory | `memory.py`: Memory, TrimmingMemory, SummarizationMemory, RAGMemory; EmbeddingModel в `llm.py`; `chapter4.ipynb` |
| 5 — Tools | `tools.py`: Tools, NativeTools, tool_to_schema; `toolbox.py`: multiply; `chapter5.ipynb` |
| 6 — Planning and Reflection | `planning.py`: ReAct, NativeReAct; цикл с `max_steps` в TinyAgent; add/subtract в `toolbox.py`; `chapter6.ipynb` |

Без planner агент выполняет один вызов генерации на запрос и записывает результат в траекторию.
При выборе инструмента агент выполняет его и возвращает observation, без повторной
генерации ответа. SummarizationMemory дополнительно вызывает LLM для сводки после
завершения ответа или получения результата инструмента. С planner агент продолжает
цикл после observation до окончательного ответа или лимита шагов; сводка создаётся
только после окончательного ответа.

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

Откройте `demo.ipynb` или `chapter3.ipynb`–`chapter6.ipynb`, выберите ядро
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
- **SummarizationMemory:** обновляет сводку после окончательного assistant-сообщения
  или получения результата инструмента. Вызов инструмента не сжимается до получения observation.
  При включённом planner сводка откладывается до окончательного ответа всего запуска.
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

## Инструменты: глава 5

```python
from agent import TinyAgent
from llm import LLM
from toolbox import multiply
from tools import Tools, NativeTools

tools = NativeTools()  # Tools() для JSON-вызова через текстовый промпт
tools.add_tool("multiply", multiply, "Multiplies two numbers: multiply(a: str, b: str)")
agent = TinyAgent(llm=LLM(model="gemma4:e4b", temperature=0), tools=tools)
print(agent.run("Use the multiply tool to calculate 5.1 times 7.3."))
print(agent.memory.get_messages())
print(agent.trajectory.runs)
```

- **Tools:** модель выводит `{"tool": "multiply", "kwargs": {"a": "5.1", "b": "7.3"}}`.
  Результат сохраняется как user-сообщение `OBSERVATION: 37.23`.
- **NativeTools:** схемы функций передаются через `tools`. В памяти сохраняется
  native-вызов и ответ с ролью `tool` и соответствующим `tool_call_id`.
- В обоих случаях без planner один `run()` возвращает observation. Следующий запрос пользователя
  может использовать этот результат; скрытого цикла повторных вызовов нет.
- Обычный ответ без инструмента завершает шаг. В текстовом режиме поддерживается
  `final_answer` с `kwargs={"answer": "..."}` или строкой в `kwargs`.
- Вызовы идут только через реестр Python-функций. Неизвестное имя, неправильные
  аргументы, ошибка функции и отказ пользователя возвращаются как observations.
  Некорректный JSON и несколько вызовов за шаг отклоняются до выполнения.
- `Tools(requires_approval=["name"])` и `NativeTools(...)` спрашивают в терминале
  подтверждение с именем и аргументами. По умолчанию отказ. Для приложения или notebook
  можно передать `approval(name, kwargs) -> bool`; пример notebook явно отказывает.
- `tool_to_schema` поддерживает именованные параметры с простыми аннотациями
  `str`, `int`, `float`, `bool`, `list`, `dict`; параметры с default не обязательны.
  Неаннотированные параметры описываются как строки. Сложные типы и variadic-параметры
  отклоняются; конвертация и проверка значений остаются задачей самой функции.

Observations не считаются новым ходом в TrimmingMemory и не запускают поиск в
RAGMemory. SummarizationMemory сжимает вызов и его результат вместе. В траектории
остаются исходные шаги независимо от сжатия памяти.

В главе 5 книги текстовый подход показан на `gemma3:12b`; локальные примеры используют
доступную `gemma4:e4b` с отключённым reasoning. MCP, Skills и обучение tool-calling
в этой главе описаны теоретически: внешние MCP-серверы не подключаются, модели не обучаются.

## Планирование: глава 6

```python
from agent import TinyAgent
from llm import LLM
from planning import NativeReAct
from toolbox import add, multiply, subtract
from tools import NativeTools

tools = NativeTools()
for function in (add, multiply, subtract):
    tools.add_tool(function.__name__, function)

agent = TinyAgent(
    llm=LLM(model="gemma4:e4b", think=True, temperature=0),
    tools=tools,
    planner=NativeReAct(max_steps=6),
)
print(agent.run("Use the tools to calculate (4.6 + 6.685) * 4 - 3.14, one operation per tool call."))
print(agent.trajectory.runs)
```

- **ReAct + Tools:** текстовые `THOUGHT` / `ACTION`; завершение через `final_answer`.
  Пример использует `think=False`. Сгенерированный THOUGHT описывает шаг, но не
  гарантирует достоверного объяснения внутренних вычислений модели.
- **NativeReAct + NativeTools:** структурированные вызовы; ответ без `tool_call`
  завершает запуск. Reasoning записывается, если backend его возвращает.
- Каждый шаг включает один запрос генерации и максимум одно исполнение инструмента.
  Для `add → multiply → subtract → ответ` нужны четыре шага. Промежуточные результаты:
  `11.285`, `45.14`, `42.0`.
- `max_steps` — положительное целое, по умолчанию 10. Лимит начинается заново для
  каждого `run()`. При исчерпании возвращается `Max steps reached without completion.`;
  выполненные действия остаются в памяти и траектории, финальный ответ не создаётся.
- Лимит ограничивает генерации, а не длительность инструментов или токены.
  Запрос SummarizationMemory после ответа идёт дополнительно к шагам planner.
- Весь запуск считается одним ходом для TrimmingMemory. RAG не повторяет поиск
  после инструмента, SummarizationMemory не сжимает промежуточные observations.
- Ошибки исполнения и отказы становятся observations для следующего шага.
  Подтверждение проверяется при каждой попытке. Ошибки протокола и HTTP прерывают
  запуск исключением, автоматических повторов нет.

В `chapter6.ipynb` показаны оба варианта на живой модели, проверка фактически
выполненной цепочки, TrajectoryViewer и детерминированный пример ограничения шагов.
Self-Refine, Reflexion и обучение через RL в главе обсуждаются теоретически;
отдельные механизмы самокритики и обучения здесь не реализованы.

## Проверки

```sh
python3 -B -m unittest discover -s tests -v
```

Тесты работают без сервера и загрузки моделей: проверяют HTTP-контракт через mock,
историю диалога, обрезку, суммаризацию, RAG, выполнение инструментов, подтверждения,
native-контракт, циклы planner, лимиты шагов, траекторию и код notebook-примеров.
Для проверки реального LLM запускайте notebook. Раздел RAG требует отдельной
embedding-модели; успешные offline-тесты не подтверждают качество её поиска.

Повторяемая живая проверка памяти (запускается явно, расходует ресурсы локальной модели):

```sh
python3 -B scripts/live_check.py
python3 -B scripts/live_check.py --rag  # требуется embeddinggemma
python3 -B scripts/live_check.py --tools  # prompt и native multiply, затем повторное использование результата
python3 -B scripts/live_check.py --planning  # полные ReAct / NativeReAct: три инструмента и ответ 42
```

Это небольшие smoke-проверки по ответам модели, не полноценная оценка качества.
GitHub Actions запускает offline-тесты на Python 3.12, 3.13 и 3.14, а также отдельную
проверку HTML TrajectoryViewer с установленными notebook-зависимостями. Локально этот
тест включается при запуске через `.venv/bin/python`; без пакета `illustrated-agents`
он явно пропускается. Для рендеринга нужны и `rich`, и `pygments` из requirements.
