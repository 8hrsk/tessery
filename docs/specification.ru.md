# Техническое задание: Python MLX Embeddings

Статус: draft 1.0  
Целевая платформа: macOS Apple Silicon  
Лицензия проекта: Apache License 2.0  
Рабочее название репозитория: `yuri-mlx-embeddings`  

## 1. Назначение

Создать самостоятельную Python-библиотеку и локальный inference runtime для
построения текстовых embeddings на macOS через MLX/Metal. Проект должен:

- заменить GPLv3-зависимость `mlx-embeddings` независимо реализованным кодом;
- работать полностью локально и не требовать сети во время inference;
- поддерживать текущий embedding-протокол Yuri без изменения памяти, SQLite,
  hybrid retrieval или frontend;
- предоставлять удобный Python API и минимальный OpenAI-compatible HTTP API;
- иметь расширяемую архитектуру для новых моделей, pooling, projection и
  batching-стратегий;
- поставляться как воспроизводимый standalone runtime для вложения в macOS app;
- иметь доказанное функциональное соответствие и сравнительные benchmarks с
  текущим `mlx-embeddings`.

Первая production-версия предназначена для `darwin/arm64`, macOS 14.0 или
новее. Linux, Windows, Intel macOS и macOS 11–13 не входят в область MLX/Metal
v1. Конкретные Python и MLX versions и Mach-O deployment target фиксируются
lock-файлом после baseline phase; release не может использовать плавающий
`latest`. Официальный MLX требует Apple Silicon, native Python >=3.10 и macOS
>=14.0.

## 2. Не входит в проект

Библиотека не должна реализовывать:

- SQLite, vector index, RAG ranking, backfill или миграции памяти;
- политику согласия на передачу private/sensitive данных;
- TLS facade, bearer token, certificate pinning и Yuri runtime identity;
- chat, completion, audio, image или reranking API;
- автоматическое скачивание моделей в production-режиме;
- произвольный `trust_remote_code` или исполнение кода из model pack;
- полную совместимость со всеми вариантами OpenAI Embeddings API.

Эти границы обязательны: Python-проект отвечает за `text -> vector`, а
управление доверием и жизненным циклом выполняет Go-библиотека.

## 3. Лицензирование и независимая реализация

Исходный код проекта публикуется под Apache-2.0. Разрешены зависимости с
лицензиями MIT, BSD, Apache-2.0 и иными заранее одобренными permissive-лицензиями.

Production package, lock-файл и runtime не должны содержать, импортировать или
транзитивно устанавливать:

- `mlx-embeddings`;
- GPL/AGPL-компоненты;
- код, скопированный или адаптированный из GPL-реализации.

Реализация строится по официальной спецификации Qwen3 Embedding, публичным
форматам `config.json`, `tokenizer.json`, SafeTensors и API MLX. Текущая
GPL-библиотека может запускаться только как black-box baseline в отдельном
benchmark environment. Она не входит в dev/runtime dependencies основной
библиотеки.

Процесс разработки является независимой реализацией с документированным
provenance. Reviewer спецификации и golden corpus не пишет model implementation;
implementer подтверждает в каждом PR отсутствие копирования GPL-кода, заполняет
provenance checklist, а release проходит similarity и dependency scan. Если
участник ранее изучал исходники GPL-реализации, проект не использует термин
«clean room» и требует отдельного legal sign-off перед GA.

В репозитории обязательны:

- `LICENSE` с полным Apache-2.0;
- `NOTICE`;
- `THIRD_PARTY_NOTICES.txt`;
- dependency license allowlist;
- SPDX SBOM для каждого release artifact;
- документ `PROVENANCE.md` с источниками спецификации и model artifacts;
- CI gate, запрещающий GPL/AGPL в основном dependency graph.

Название и import namespace не должны создавать впечатление, что проект
является форком или официальной новой версией `mlx-embeddings`.

## 4. Зафиксированный профиль совместимости Yuri v1

Профиль является неизменяемым контрактом, а не набором defaults:

| Поле | Значение |
|---|---|
| Protocol | `yuri-embedding-protocol-v1` |
| Legacy Pack ID candidate | `macos-mlx-qwen3-embedding-0.6b-4bit-mrl384-v1` |
| Model | `Qwen3-Embedding-0.6B-4bit-DWQ` |
| Source | `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` |
| Revision | `6c3ae70858513f1a78e9cdca3cae330d9075cd2a` |
| Native dimensions | 1024 |
| Yuri dimensions | 384 |
| Distance | cosine |
| Normalization | unit L2 |
| Max sequence length | 512 tokens |
| Production batch | 1..32 |
| Query/document roles | отсутствуют, plain text |

Compatibility mode не должен самостоятельно добавлять query instruction,
менять EOS, padding side, truncation, pooling или normalization. Даже если
новая upstream-документация рекомендует instruction prefix, это создаёт другое
embedding space и требует нового профиля и переиндексации Yuri.

Legacy Pack ID не присваивается релизу заранее. Его разрешено сохранить только
после подписанного compatibility report из раздела 13. По умолчанию новая
реализация получает новый engine compatibility ID и новый Pack ID.

Baseline phase обязана выпустить machine-readable `preprocessing-profile.json`
с точными padding/truncation sides, EOS/BOS/special-token injection, token IDs,
mask dtype/shape, position IDs/RoPE parameters, 511/512/513 behavior, pooling,
порядком MRL truncate-before-normalize и hashes `config.json`/tokenizer files.
До утверждения этого файла заявлять совместимость со старым пространством
векторов запрещено.

## 5. Архитектура

Рекомендуемая структура:

```text
src/yuri_mlx_embeddings/
  __init__.py
  api.py
  errors.py
  descriptors.py
  config.py
  manifests.py
  batching.py
  projection.py
  cli.py
  server/
    protocol_v1.py
    uds.py
  tokenizers/
    base.py
    huggingface_json.py
  backends/
    base.py
    qwen3_mlx.py
  models/
    qwen3.py
tests/
  unit/
  contract/
  integration/
  real_mlx/
benchmarks/
model-manifests/
```

Слои model backend, tokenizer, pooling, MRL projection, normalization,
batch scheduler и transport должны зависеть от интерфейсов, а не друг от друга
напрямую. Добавление новой модели требует нового зарегистрированного
`ModelProfile` и backend adapter, но не изменения HTTP server.

Минимальный внутренний протокол:

```python
class EmbeddingBackend(Protocol):
    @property
    def descriptor(self) -> ModelDescriptor: ...

    def encode(
        self,
        texts: Sequence[str],
        *,
        dimensions: int,
        max_length: int,
    ) -> numpy.ndarray: ...

    def close(self) -> None: ...
```

`ModelDescriptor` должен включать model ID, immutable model и tokenizer
revisions, native и допустимые output dimensions, max length, pooling,
normalization, quantization, engine compatibility ID и hashes model manifest.

## 6. Публичный Python API

Обязательный API:

```python
from yuri_mlx_embeddings import EmbeddingModel, LoadOptions

model = EmbeddingModel.load(
    model_dir="/absolute/path/to/model",
    options=LoadOptions(
        model="Qwen3-Embedding-0.6B-4bit-DWQ",
        revision="6c3ae70858513f1a78e9cdca3cae330d9075cd2a",
        dimensions=384,
        max_length=512,
    ),
)

vectors = model.encode(
    ["текст 1", "text 2"],
    dimensions=384,
)
```

Требования:

- результат — `numpy.ndarray` формы `[batch, dimensions]`, dtype `float32`;
- входной и выходной порядок совпадает;
- значения finite, норма каждого вектора `1.0 ± 1e-4`;
- normalization для v1 всегда `unit_l2` и не отключается per request;
- допустимые dimensions задаются allowlist активного ModelProfile, а Yuri
  compatibility profile разрешает только 384;
- пустой batch в прямом Python API возвращает `[0, dimensions]` без inference;
- whitespace-only strings отклоняются typed exception;
- model/tokenizer загружаются один раз и переиспользуются;
- `encode` thread-safe либо явно сериализован bounded scheduler;
- cancellation и deadline предоставляются через отдельный async API или request
  context server-а; неподдерживаемое мгновенное прерывание Metal kernel должно
  быть честно задокументировано;
- `close()` идемпотентен и запрещает последующие вызовы;
- ошибки не содержат входной текст, tokens, embeddings или секретные пути.

Дополнительный стабильный API:

```python
@dataclass(frozen=True)
class HealthStatus:
    loaded: bool
    ready: bool
    compatibility_id: str

@dataclass(frozen=True)
class MemoryStats:
    active_bytes: int | None
    peak_bytes: int | None
    cache_bytes: int | None

model.descriptor: ModelDescriptor
model.health() -> HealthStatus
model.memory_stats() -> MemoryStats
model.warmup() -> None
model.close() -> None
await model.encode_async(texts, *, dimensions=384) -> numpy.ndarray
```

Typed exception hierarchy включает configuration, invalid input, model
manifest, unsupported profile, overload, canceled, timeout, inference и closed
errors. Async cancellation следует state machine раздела 14.

Публичные API получают type hints и reference documentation. Breaking changes
разрешены только в новой major SemVer.

## 7. Yuri protocol v1

Production server слушает приватный Unix socket. Обязательный endpoint:

```http
POST /v1/embeddings
Content-Type: application/json
```

Запрос:

```text
{
  "model": "Qwen3-Embedding-0.6B-4bit-DWQ",
  "input": ["текст 1", "text 2"],
  "dimensions": 384
}
```

Успешный ответ:

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": <ровно 384 finite числа>
    }
  ],
  "model": "Qwen3-Embedding-0.6B-4bit-DWQ",
  "usage": {
    "prompt_tokens": 0,
    "total_tokens": 0
  }
}
```

Это структурное сокращение, не JSON fixture. Canonical fixture обязан содержать
полный массив из 384 чисел.

Обязательная семантика:

- `input` — JSON array из 1..32 непустых строк;
- общий объём input — не более 1 MiB UTF-8;
- тело запроса — не более 2 MiB;
- `model` и `dimensions` должны точно совпадать с активным профилем;
- unknown fields отклоняются;
- duplicate JSON object keys, trailing payload, `NaN` и `Infinity` отклоняются;
- serializer не может выдавать non-RFC numeric constants;
- на каждый input возвращается ровно один уникальный index `0..N-1`;
- вектор содержит ровно 384 finite float32-compatible чисел;
- response не превышает 16 MiB;
- invalid request — `400` с безопасным стабильным error code;
- неизвестный route/method — `404`/`405`;
- overload — `429`;
- inference failure — `500` без traceback и пользовательских данных;
- отменённый клиентом запрос не должен добавляться в очередь повторно.

Дополнительные endpoints допускаются:

- `GET /healthz` — процесс жив;
- `GET /readyz` — model/tokenizer загружены и runtime готов;
- `GET /v1/models` — только безопасные descriptor metadata.

Обязательный внутренний UDS endpoint:

- `GET /__runtime/identity` возвращает protocol version, engine compatibility
  ID, active model/revision/dimensions и model/tokenizer manifest digests.

Go supervisor сверяет эти данные с проверенным pack до открытия внешнего TLS
facade. Endpoint существует только на private UDS и наружу не проксируется.
`/readyz` означает, что модель загружена; Go `Start` ждёт identity и health, а
полная warm readiness доказывается отдельным embedding probe в activation budget.

`GET /__yuri/identity`, TLS и bearer auth не реализуются Python server: ими
владеет Go runtime facade. Chat/audio/image routes должны отсутствовать.

## 8. CLI и процессный контракт

Обязательные команды:

```text
yuri-mlx-embeddings serve
yuri-mlx-embeddings validate-model
yuri-mlx-embeddings benchmark
```

Production launch:

```text
python3 -I -m yuri_mlx_embeddings serve \
  --model-dir /absolute/model \
  --model Qwen3-Embedding-0.6B-4bit-DWQ \
  --revision 6c3ae70858513f1a78e9cdca3cae330d9075cd2a \
  --dimensions 384 \
  --max-length 512 \
  --unix-socket /private/run/embedding.sock \
  --parent-pid 12345
```

Требования к `serve`:

- UDS является production default; socket создаётся с правами не шире `0600`
  внутри переданного каталога `0700`;
- TCP разрешён только на literal loopback и только явным dev-флагом;
- `0.0.0.0`, `::` и hostname bind запрещены;
- смерть исходного parent PID завершает процесс;
- до bind/model load проверяется `--parent-pid == os.getppid()`; произвольный
  существующий PID не принимается, reparenting завершает child;
- SIGTERM/SIGINT запускают bounded graceful shutdown;
- stdout/stderr не содержат пользовательские данные;
- startup/readiness не требуют сети;
- exit codes и error codes документированы и стабильны;
- queue, concurrency, batch и request limits имеют безопасные пределы.
- существующий socket/path отклоняется; каждый parent directory проверяется
  через `lstat`, принадлежит текущему UID и имеет mode `0700`;
- процесс использует `umask 077`, ограничивает длину UDS path и очищает только
  созданный им inode/path;
- header size, read-header, body-read и idle timeouts являются bounded и
  покрываются slowloris tests.

## 9. Model loading и offline security

Production profile обязан:

- принимать только абсолютный локальный model directory;
- работать с `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1`;
- никогда не скачивать и не устанавливать зависимости автоматически;
- запрещать `trust_remote_code`, pickle и `torch.load`;
- загружать только allowlisted architecture из `config.json`;
- проверять immutable revision и SHA-256/size каждого model artifact;
- отклонять symlink, hardlink, device, FIFO, socket и path traversal;
- не импортировать из cwd, user site или `PYTHONPATH`;
- применять body/input limits до tokenization и Metal allocation;
- не логировать texts, token IDs и embeddings;
- не обещать secure erase Python strings или GPU memory: гарантируется только
  отсутствие намеренной персистенции, логирования и сетевой передачи.
- использовать статический built-in registry; entry points, plugin discovery и
  dynamic imports из model directory/user site запрещены.

Проверяемые model artifacts профиля Yuri:

- `model.safetensors`;
- `model.safetensors.index.json`;
- `config.json`;
- `tokenizer.json`;
- `tokenizer_config.json`.

Manifest содержит точные hashes, sizes, modes, model/tokenizer revisions и
license identifiers. Он задаёт exact required и allowed file set; хешируются
все bytes, которые реально читает tokenizer/backend. Missing, duplicate или
неразрешённые extra artifacts должны fail closed.

## 10. MLX/Metal implementation

Qwen3 backend реализует явно зарегистрированную архитектуру из model config:

- tokenizer и attention mask;
- quantized embedding/linear layers;
- attention, RoPE, RMSNorm и MLP;
- last-token pooling, совместимый с утверждённым профилем;
- MRL `truncate to dimensions`, затем unit-L2 normalization;
- единственный host conversion после завершения batch;
- явный `mx.eval()` до завершения запроса и измерения времени.

Compatibility profile должен точно повторить утверждённую tokenization и
projection семантику старого Yuri. Расширенный API может позднее поддерживать
query/document roles и instruction templates, но они получают отдельный
compatibility ID.

## 11. Производительность

Обязательные оптимизации v1:

- singleton model/tokenizer на процесс;
- vectorized batch inference без Python-цикла по текстам модели;
- batch projection и normalization на MLX либо одним NumPy operation;
- минимум device-to-host копирований и `.tolist()`;
- bounded queue и backpressure;
- не более одного одновременного Metal forward для одной model instance;
- fast paths для foreground batch 1 и indexing batch 16;
- контроль graph/cache growth;
- warmup/compile отделены от измеряемого warm inference.

Экспериментальные оптимизации допускаются только за feature flag и после parity:

- `mx.compile` для стабильных shape buckets;
- length buckets для снижения padding;
- dynamic microbatching с ограниченным wait window;
- tokenization cache с ограничением памяти;
- adaptive batch по token count;
- fused/custom Metal operations.

Foreground batch 1 нельзя задерживать ради microbatch по умолчанию.
Оптимизация, меняющая embedding space, требует нового compatibility ID.

## 12. Benchmark suite

Baseline — точная pinned версия текущего `mlx-embeddings`, запускаемая как
отдельный optional benchmark environment. Сравнение выполняется на одном Mac,
одной версии macOS/MLX, одинаковых model bytes, power mode и thermal state.

Матрица:

- batch: 1, 4, 16, 32;
- token buckets: 32, 128, 512;
- русский, английский, mixed language, emoji, combining Unicode и code;
- direct Python API и полный HTTP/UDS path;
- cold model load + first embedding;
- warm p50/p95/p99;
- concurrency 1/2/N;
- 30–60 minute soak.

Метрики:

- model load и time-to-first-embedding;
- latency p50/p95/p99;
- texts/sec и tokens/sec;
- peak и steady RSS/unified memory;
- CPU/GPU utilization при доступности инструмента;
- package/runtime size;
- cache growth и память после soak;
- cancellation/overload behavior.

Методика сохраняет raw JSON, commit IDs, dependency/model hashes, warmup count,
число итераций и confidence interval. Нельзя делать вывод по одному среднему.

Baseline manifest фиксирует version/commit `mlx-embeddings`, MLX, tokenizer,
Python, полный lock hash и model hashes. Для gate используются фиксированный M1
low-memory host и минимум один более новый M-series host, не менее 30 cold и
100 warm samples на основной профиль. Performance waiver требует отдельного
reviewer, причины, raw report и срока устранения; формулировка «объяснимая
регрессия» сама по себе не является разрешением.

Hard gate: кандидат не должен иметь необъяснимую регрессию более 10% по warm
p95, throughput или steady memory на том же host. Если достижение паритета
невозможно, релиз блокируется до анализа и отдельного решения.

Optimization target, не гарантия приемки: улучшить минимум одну из метрик
cold-start, p95, throughput или RSS на 15% без регрессии остальных более 5%.

## 13. Functional parity и миграция vectors

До реализации создаётся frozen corpus и baseline artifact с:

- exact token IDs и attention masks;
- полными 1024D и итоговыми 384D vectors;
- RU, EN, cross-lingual, code и Unicode cases;
- границами 511/512/513 tokens;
- mixed-length batches;
- retrieval relevance labels.

Для сохранения старого Yuri Pack ID должны одновременно выполняться:

- output shape/order/dtype совпадают;
- `abs(norm-1) <= 1e-4`;
- cosine old/new для каждого vector не ниже предварительного порога `0.99999`;
- old-query/new-doc, new-query/old-doc и mixed index проходят retrieval tests;
- top-k overlap и ordering на Yuri fixture не ниже 99.9%;
- Recall@k, MRR и nDCG не имеют утверждённой регрессии.

Порог `0.99999` должен быть подтверждён измерениями, а не ослаблен для
прохождения теста. Если любой parity gate не выполнен, библиотека получает новый
engine compatibility ID/Pack ID, а Yuri обязан выполнить автоматическую shadow
reindex из SQLite с атомарным переключением поколения. Смешивание несовместимых
old/new vectors запрещено.

До начала implementation baseline phase фиксирует corpus/version/hash, значения
`k`, tie epsilon, формулы Recall/MRR/nDCG, aggregation и допустимое число
failures. Решение сохранить legacy ID оформляется подписанным immutable JSON и
Markdown report. Без такого report результат всегда новый Pack ID + shadow
reindex.

## 14. Тестирование

Canonical schemas и language-neutral fixtures протокола хранятся в
`yuri-embedding-go`. Python CI использует зафиксированный tagged artifact с
его SHA-256 и хранит локальную копию только как test input. Расхождение с
canonical artifact блокирует release; golden fixtures нельзя автоматически
перезаписывать из результатов тестируемой реализации.

### Unit и property tests без Metal

- config/descriptor/manifest validation;
- dimensions и max length boundaries;
- empty, whitespace, oversized input;
- Unicode, invalid UTF-8 boundary на wire level;
- truncation, last-token selection, MRL и normalization;
- NaN/Inf rejection;
- batching order и bounded queue;
- cancellation до очереди и overload;
- no sensitive data в logs/errors;
- parent watcher и graceful shutdown;
- socket permissions;
- missing/corrupt/extra model artifacts.

Cancellation state machine: отменённая queued job не запускается; уже начатый
Metal kernel может завершиться, но его result отбрасывается, slot и buffers
освобождаются, retry не создаётся. Библиотека не обещает мгновенно прервать уже
отправленный GPU kernel.

### Protocol contract tests

- exact `POST /v1/embeddings` request/response fixtures;
- batches 1, 16, 32;
- wrong model/dimensions, unknown fields, malformed/trailing JSON;
- request/input/response size limits;
- stable safe errors 400/404/405/429/500;
- health/readiness/models;
- отсутствие chat/audio/image routes;
- запрет non-loopback TCP;
- slow client, disconnect, timeout и overload.

### Real MLX tests

Opt-in suite на macOS arm64 с заранее установленным pinned model pack:

```text
YURI_REAL_MLX_EMBEDDING_TEST=1 \
YURI_MLX_MODEL_DIR=/absolute/pinned/model \
pytest -m real_mlx
```

Проверяются model load, deterministic repeat, batch order, 384D/finite/L2,
semantic RU и EN-to-RU margins, cold/warm execution, SIGTERM, parent death,
process/socket cleanup и отсутствие сетевых попыток.

### Fault и soak tests

- corrupt weights/tokenizer/config;
- Metal allocation failure;
- process crash during request;
- queue saturation;
- repeated start/stop;
- cancellation storm;
- длительный mixed-size workload без unbounded graph/RSS growth.

Минимальные quality gates устанавливаются после baseline corpus, но не могут
быть слабее текущего Yuri retrieval evaluation.

## 15. Packaging и releases

Обязательные outputs:

1. Python sdist и wheel для использования как библиотеки.
2. Deterministic standalone `darwin-arm64` runtime archive для Yuri.
3. Manifest с SHA-256, sizes, modes, versions и provenance.
4. SBOM, notices, benchmark и compatibility reports.

Standalone runtime:

- содержит собственный CPython и все зависимости;
- не зависит от system Python, PATH, Homebrew или package manager;
- не содержит symlinks и mutable package cache;
- имеет executable `bin/python3`;
- работает после перемещения в путь с пробелами и Unicode;
- не обращается в сеть при запуске;
- допускает последующую inside-out code signing в Yuri release pipeline.

Без Developer ID артефакт может быть source/wheel GA либо явно маркированным
ad-hoc community runtime. Он не должен называться notarized macOS application.

## 16. CI

На каждый PR:

- formatter, linter, type checker;
- unit/property/protocol tests;
- coverage report;
- dependency hash и license gates;
- secret scan и SBOM validation;
- build sdist/wheel;
- проверка protocol fixtures.

Workflow dependencies фиксируются по commit. Release также требует CVE scan,
artifact provenance/attestation и aggregate SBOM/NOTICE для CPython, Python
packages, MLX и model artifacts.

На macOS arm64 release/nightly:

- real MLX suite;
- differential test с изолированным baseline;
- benchmark regression gate;
- soak/fault suite;
- standalone runtime build и clean offline relocation test.

Критические parsing, manifest, projection и server-validation branches должны
иметь 100% branch coverage. Общий line coverage — не ниже 90%; coverage не
заменяет real-model и fault tests.

## 17. Definition of Done

Проект v1 завершён, когда:

1. Production dependency graph не содержит GPL/AGPL.
2. Python API и Yuri protocol v1 документированы и заморожены fixtures.
3. Pinned Qwen3 pack работает offline на macOS arm64.
4. Все unit, property, protocol, real MLX, fault и license tests проходят.
5. Differential и retrieval parity определили, можно ли сохранить старый Pack
   ID; при отрицательном результате выпущен новый ID и migration declaration.
6. Benchmarks воспроизводимы и проходят regression gate.
7. Standalone runtime relocatable, self-contained и имеет manifest/SBOM/notices.
8. Go integration library запускает runtime через UDS и проходит общий contract
   suite.
9. Yuri integration test подтверждает indexing batch 16, query batch 1,
   fallback при сбое и отсутствие изменений authoritative SQLite memory.

## 18. Этапы реализации

1. Protocol/spec fixtures, provenance и baseline corpus.
2. Descriptor, manifest и tokenizer layer.
3. Qwen3 MLX forward, pooling, projection и Python API.
4. UDS server и CLI.
5. Security/offline/fault hardening.
6. Differential correctness и retrieval quality.
7. Optimization и benchmark suite.
8. Standalone runtime packaging, SBOM и cross-repo integration.

Источники спецификации:

- [Qwen3 Embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Pinned MLX model pack](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ/tree/6c3ae70858513f1a78e9cdca3cae330d9075cd2a)
- [MLX](https://github.com/ml-explore/mlx)
- [MLX platform requirements](https://ml-explore.github.io/mlx/build/html/install.html)
