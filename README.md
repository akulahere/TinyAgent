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
| 7 — Evaluating Agents | `evaluator.py`: Benchmark, Evaluator, scorers, pass@k/pass^k; `chapter7.ipynb` |
| 8 — Multi-Agent Systems | `multi_agent.py`: AgentTeam, create_agent_team; today/days_between в `toolbox.py`; `chapter8.ipynb` |
| 9 — Multi-Modal Understanding | MultimodalMemory в `memory.py`, `TinyAgent.run(..., image_data=...)`, `chapter9.ipynb` |
| 10 — Code Agents and Code LLMs | CodeWorkspace/make_code_tools в `toolbox.py`, события TinyAgent, `display.py`, `cli.py`, `chapter10.ipynb` |

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

Откройте `demo.ipynb` или `chapter3.ipynb`–`chapter10.ipynb`, выберите ядро
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

## Оценка агента: глава 7

```python
from agent import TinyAgent
from evaluator import Benchmark, Evaluator, exact_match_scorer
from llm import LLM

llm = LLM(model="gemma4:e4b", temperature=0)
benchmark = Benchmark(
    name="Arithmetic: one demonstration question",
    examples=[{"task": "What is 2 + 2? A) 3 B) 4 C) 5. Answer with only the letter.", "expected": "B"}],
    scorer=exact_match_scorer,
)
result = Evaluator(lambda: TinyAgent(llm)).run(benchmark)
print(result)
```

`Evaluator` вызывает фабрику для каждого примера. Она должна возвращать **новый
TinyAgent** со своей памятью и траекторией; клиент LLM можно переиспользовать.
Отчёт содержит `name`, `pass_rate` и `results` с полями `task`, `prediction`,
`completed`, `passed`.

- **exact_match_scorer:** строго одна буква A–J, без учёта регистра и внешних пробелов.
  Буква внутри объяснения не считается точным совпадением.
- **programmatic_scorer:** запускает функцию `example["check"]` для непустого ответа.
  Проверка возвращает boolean или число от 0 до 1.
- **make_judge_scorer(judge):** передаёт отдельному LLM-клиенту вопрос, эталон и ответ.
  Ожидает единственное десятичное число от 0 до 1; невалидная оценка вызывает ошибку.
  Данные отделены от системной инструкции; это не гарантия защиты от prompt injection.
- Для boolean-scorer `pass_rate` означает долю пройденных задач; для дробных оценок —
  средний балл. Поле `passed` сохраняет исходную оценку, без скрытого порога.
- Пустой ответ, observation вместо финального ответа и исчерпание `max_steps`
  получают ноль без вызова scorer. `completed` описывает завершение протокола,
  а не правильность ответа. Ошибки HTTP, агента и scorer передаются вызывающему коду;
  отчёт не выдаётся за успешно завершённую оценку при сбое инфраструктуры.

`chapter7.ipynb` содержит три задания MMLU-Pro из главы, три упрощённые проверки
IFEval и три открытых вопроса с LLM-судьёй. Это учебные подвыборки и эвристики,
**не официальный запуск бенчмарков**. Проверка длины или отсутствия запятой не
оценивает смысл и стиль ответа. Ожидаемые ответы в notebook следуют примерам книги.

Для запуска без внешнего API судья по умолчанию использует ту же локальную
`gemma4:e4b`, с `think=False`. Это демонстрация оценщика, не независимое подтверждение
качества. В книге используется внешний судья Gemini; здесь можно явно передать
другой настроенный `LLM` в `make_judge_scorer`.

Метрики для повторных запусков **одного задания** с бинарной проверкой:

```python
from evaluator import pass_at_k, pass_hat_k

print(pass_at_k(10, 6, 3))   # ≈ 0.967: хотя бы один успех среди трёх попыток
print(pass_hat_k(10, 6, 3))  # ≈ 0.167: все три попытки успешны
```

Аргументы — общее число запусков `n`, число успехов `c` и число выбираемых попыток
`k`: `0 <= c <= n`, `1 <= k <= n`. Расчёт использует выбор без возвращения;
при `k=1` обе метрики равны `c/n`. Пример синтетический, не оценка Gemma.
Дробные баллы судьи и ответы на разные задания нельзя считать такими попытками.

Rubric-based evaluation, проверка траекторий и safety-бенчмарки в главе обсуждаются
концептуально. Новых отдельных подсистем для них нет; `Evaluator` оценивает итог,
а траектории по-прежнему можно изучать через `agent.trajectory` и TrajectoryViewer.

## Несколько агентов: глава 8

```python
from llm import LLM
from multi_agent import create_agent_team

team = create_agent_team(LLM(model="gemma4:e4b", think=True, temperature=0), max_steps=6)
answer = team.orchestrator_agent.run(
    "If I save EUR 4 per day starting today and stopping before 2030-12-31, "
    "how much will I save? Ask the date specialist to get today and compute "
    "the date difference with its tools, then ask the math specialist to "
    "multiply that number of days by 4 with its tool. Exclude the end date."
)
print(answer)
print(team.orchestrator_agent.trajectory.runs)
print(team.date_agent.trajectory.runs)
print(team.math_agent.trajectory.runs)
```

Оркестратор имеет два инструмента: `ask_date_agent(question)` и
`ask_math_agent(question)`. Это функции, вызывающие других TinyAgent:

- **Агент дат:** `today()` возвращает локальную дату компьютера;
  `days_between(a, b)` считает `b - a` для ISO-дат, без прибавления единицы.
  Одинаковые даты дают 0, обратный порядок — отрицательное число.
- **Математик:** `add`, `subtract`, `multiply`.
- **Оркестратор:** выбирает специалиста, получает его финальный ответ как observation
  и продолжает свой цикл. Он не получает внутреннюю историю специалиста.

Система централизованная, вызовы синхронные. Клиент LLM общий, но память, planner,
инструменты и траектория у каждого агента собственные. Повторные запросы к той же
команде сохраняют истории; `create_agent_team` создаёт свежую команду. Для независимых
оценок главы 7 передавайте `Evaluator(lambda: create_agent_team(llm).orchestrator_agent)`.

`max_steps` ограничивает каждый `run()`, включая каждое делегирование, отдельно.
Общего бюджета нет: при лимите `S` эта фиксированная схема допускает до `S + S²`
запросов генерации. Вложенные вызовы не выполняются параллельно; лимит не ограничивает
токены или время. Исчерпание лимита специалиста, пустой ответ или исключение дают
ошибочный observation у оркестратора; он может повторить запрос в пределах своего
лимита. Финальный ответ оркестратора не гарантирует успешного выполнения подзадач.

Подтверждения `requires_approval` задаются отдельно в реестре оркестратора и
реестрах специалистов. Встроенные инструменты примера читают дату и считают числа.

В `chapter8.ipynb` проверяется реальная цепочка делегирования и аргументы инструментов;
TrajectoryViewer показывает все три траектории. Пример накоплений использует
интервал от сегодня до **2030-12-31, исключая конечную дату**, поэтому ожидаемая
сумма вычисляется из фактической даты запуска, а не копируется из книги.

CAMEL, MetaGPT, A2A, социальные симуляции и research-системы описаны в главе
теоретически. Они не подключаются; код реализует паттерн «агенты как инструменты».

## Изображения: глава 9

```python
import base64
from pathlib import Path

from agent import TinyAgent
from llm import LLM
from memory import MultimodalMemory

agent = TinyAgent(
    llm=LLM(model="gemma4:e4b", think=True, temperature=0),
    memory=MultimodalMemory(),
)
image_data = base64.b64encode(Path("examples/vision/shapes.png").read_bytes()).decode("ascii")
print(agent.run("Describe the shapes and their colors from left to right.", image_data=image_data))
print(agent.run("What color was the circle?"))
```

`MultimodalMemory` сохраняет пользовательское сообщение с двумя блоками `content`:
`image_url` и `text`. `LLM.generate` уже передаёт такую структуру через JSON;
кодирование изображения в vision tokens выполняет сервер модели.

`image_data` принимает HTTP(S)-URL, raw base64 для **PNG** или полный data URL
с MIME-типом `image/png`, `image/jpeg`, `image/webp`, `image/gif`. Для JPEG и других
форматов используйте полный data URL с правильным MIME-типом. Это не путь к файлу:
локальный файл сначала нужно прочитать и закодировать, как в примере выше.

Проверяется формат URL/base64, а не содержимое декодированного файла. Поддержка
форматов и загрузки внешних URL зависит от backend; наш код URL не скачивает.
Живой notebook использует локальный PNG через base64, без внешних запросов за картинкой.
Файл `examples/vision/shapes.png` создан для проекта; его можно воспроизвести командой
`python3 scripts/create_vision_fixture.py` без дополнительных зависимостей.

Изображение остаётся в истории и отправляется при следующих запросах, включая
шаги planner. Сохраняются native tool calls и соответствующие observations.
В `chapter9.ipynb` есть сравнение запроса без картинки и с картинкой, вопрос по истории,
а также пример «посчитать видимые фигуры → вызвать multiply → ответить».

- Используйте `MultimodalMemory()` явно; другие виды памяти отклоняются при передаче
  `image_data`, чтобы изображение не потерялось и не попало в неверное поле API.
- Один запрос принимает одно изображение. Можно добавлять новые изображения
  следующими запросами; история хранится в RAM, без сжатия и лимита токенов.
- Траектория содержит исходный текст вопроса и шаги, но не байты изображения.
  TrajectoryViewer показывает ответы и действия; картинка отображается отдельно.
- Vision-модель должна поддерживаться используемым inference-сервером.
  Ошибка backend передаётся вызывающему коду; ответ при ошибке не создаётся.
- В многоагентной сборке главы 8 делегирования остаются текстовыми: изображения
  автоматически не передаются специалистам.

ViT/CLIP, аудио, видео и способы соединения энкодеров с LLM — теория главы.
Код добавляет понимание входных изображений; генерация изображений, аудио/видео
и обучение мультимодальных моделей не реализованы.

## Coding agent и CLI: глава 10

Из корня проекта, при запущенном Ollama:

```sh
mkdir -p /tmp/tinyagent-playground
.venv/bin/python cli.py --workspace /tmp/tinyagent-playground
```

Например: «Создай calculator.py с функцией сложения и проверь её». `exit`, `quit`,
Ctrl-D или Ctrl-C завершают чат; пустая строка пропускается. Память сохраняется
между запросами в одной сессии. Параметры CLI: `--model`, `--base-url`,
`--max-steps`, `--timeout`, `--no-color`. При необходимости API-ключ читается из
`TINYAGENT_API_KEY`; по умолчанию используется локальная `gemma4:e4b` с reasoning.

Программная сборка:

```python
from agent import TinyAgent
from display import Display
from llm import LLM
from planning import NativeReAct
from toolbox import CodeWorkspace, make_code_tools

workspace = CodeWorkspace("/tmp/tinyagent-playground")  # папка должна существовать
agent = TinyAgent(
    LLM("gemma4:e4b", think=True, temperature=0),
    tools=make_code_tools(workspace),
    planner=NativeReAct(max_steps=10),
    display=Display(),
)
agent.run("List the files in the workspace.")
```

- `read_file(path)` читает UTF-8; `list_files(directory=".")` показывает один уровень.
- `write_file(path, content)` создаёт родительские папки и заменяет содержимое файла.
- `execute_python(code)` запускает текущий Python в рабочей папке, возвращает stdout
  и stderr; при ошибке — также exit code. Тайм-аут по умолчанию 30 секунд.
- `make_code_tools` требует подтверждения **каждого** `write_file` и `execute_python`
  с показом конкретных аргументов. Пустой ответ `[y/N]` означает отказ. Для приложения
  можно явно передать `approval(name, kwargs) -> bool`.

**Локальный Python не является песочницей.** Он наследует права и окружение процесса,
может обращаться к сети и файлам вне рабочей папки. Проверка путей и symlink действует
на файловые инструменты, а не на произвольный Python. Прямой вызов методов CodeWorkspace
также не спрашивает разрешение: подтверждения обеспечивает реестр `make_code_tools`.
На POSIX тайм-аут завершает группу процессов; на других платформах — основной процесс.
Это не изоляция ресурсов и не защита от намеренно обходящего ограничения кода.

`max_output=20000` ограничивает вывод в observation с явным маркером обрезки. Для
stdout/stderr это обрезка после сбора данных, не ограничение памяти. Файлы читаются
только до лимита плюс один символ для обнаружения обрезки.

`Display` показывает THOUGHT, ACTION, OBSERVATION, ANSWER и остановку по лимиту.
Промежуточный текст при вызове инструмента не помечается окончательным ответом.
Цвета включаются автоматически в терминале; `--no-color` отключает их. Без `display`
TinyAgent остаётся без вывода событий. Пользовательский callback получает копию
Response и не может изменить исполняемые аргументы через этот объект; исключения
callback передаются вызывающему коду.

В `chapter10.ipynb` используются временная папка и callback, разрешающий только
конкретный показанный файл и фиксированный расчёт. Проверяются чтение, запись,
реальное исполнение, отказ и траектория. Это проверка интеграции инструментов,
а не способность модели самостоятельно написать или исправить произвольную программу.

Hosted execution Gemini, репозиторные карты, SQL, SWE-bench, Agentless и обучение
coding LLM остаются теорией главы; внешние API и контейнерные сервисы не подключены.

## Проверки

```sh
python3 -B -m unittest discover -s tests -v
```

Тесты работают без сервера и загрузки моделей: проверяют HTTP-контракт через mock,
историю диалога, обрезку, суммаризацию, RAG, выполнение инструментов, подтверждения,
native-контракт, циклы planner, лимиты шагов, траекторию и код notebook-примеров.
Для главы 7 проверяются изоляция примеров, scorers, незавершённые ответы, ошибки
судьи и формулы метрик (включая полный перебор небольших наборов попыток).
Для главы 8 проверяются даты, вложенные вызовы с реальными результатами инструментов,
раздельные истории, лимиты специалистов, ошибки, подтверждения и свежие команды для evals.
Для главы 9 проверяются image content blocks, URL/base64, сохранение изображений
в истории и planner, native tool-контракт и фактическое тело HTTP-запроса через mock.
Для главы 10 проверяются файловые пути и ссылки, подтверждения, Python-процесс,
stdout/stderr, тайм-аут, события Display, CLI и сохранение истории между запросами.
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
