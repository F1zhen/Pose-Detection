# Видеоаналитика школьного класса

Локальный Python-пайплайн для анализа записей уроков. Текущий этап реализует шаг 1: для каждого обнаруженного человека определяет состояние `sit`, `stand` или `unknown` на основе комбинации:

- относительной высоты `bbox` относительно медианы по кадру;
- нормализованной `Y`-координаты плеч из `YOLOv8-pose`.

Если оба метода дают противоречивый результат, итоговая поза помечается как `unknown`.

## Что уже сделано

- Поддержка входных форматов: `MP4`, `AVI`, `MOV`, `MKV`
- Детекция и трекинг людей через `model.track()`
- Два режима работы:
  - `calibrate` — пишет на видео сырые диагностические метрики
  - `analyze` — применяет пороги и формирует отчёт
- Сохранение:
  - аннотированного видео с `bbox`, `ID`, позой и таймкодом;
  - `raw_detections.csv/xlsx` с сырыми наблюдениями по кадрам;
  - `report.csv/xlsx` со сглаженными состояниями;
  - `events.csv/xlsx/json` с интервалами событий

## Ограничение текущего этапа

Сейчас основной фокус — шаг 1 (`сидит/стоит`). Полноценные нарушения дисциплины (`встал без причины`, `резкое перемещение`, `драка`, `беготня`) ещё не реализованы. Для верхней камеры под потолком rule-based эвристики ограничены по качеству, поэтому в проект добавлен отдельный путь через обучаемый patch classifier `sit/stand`.

## Структура проекта

- [classroom_analytics.py](/c:/Users/esimb/Video-Analytics/classroom_analytics.py) — основной скрипт
- [export_person_crops.py](/c:/Users/esimb/Video-Analytics/export_person_crops.py) — экспорт кропов людей для разметки
- [label_pose_crops.py](/c:/Users/esimb/Video-Analytics/label_pose_crops.py) — мини-программа для ручной разметки кропов
- [train_pose_classifier.py](/c:/Users/esimb/Video-Analytics/train_pose_classifier.py) — обучение классификатора `sit/stand`
- [requirements.txt](/c:/Users/esimb/Video-Analytics/requirements.txt) — зависимости
- [data](/c:/Users/esimb/Video-Analytics/data) — исходные видео
- `outputs/` — результаты запуска
- `datasets/pose_classifier/` — кропы и разметка для обучения
- `models/pose_classifier/` — обученные веса классификатора

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Нужны:

- Python 3.10+
- CUDA-совместимая среда для GPU-ускорения
- модель `YOLOv8-pose` (по умолчанию `yolov8s-pose.pt`, будет загружена `ultralytics`)

Для Windows лучше запускать проект именно из локального `.venv`, а не из глобального Python Microsoft Store. Это снижает риск ошибок вида `c10.dll` / `WinError 1114`.

## Логика классификации позы

### Метод 1. Relative height ratio

Для ракурса сбоку глобальная медиана по кадру плохо работает из-за перспективы. Поэтому теперь базовая высота считается локально: человек сравнивается только с соседними детекциями на той же глубине кадра.

Для каждого человека берётся высота `bbox`, затем считается отношение к локальной медиане высот среди людей с похожей нижней границей `bbox`:

```text
relative_height_ratio = bbox_height / local_median_bbox_height_same_depth
```

Если значение `>= STAND_HEIGHT_RATIO`, человек считается стоящим по первому методу.

### Метод 2. Shoulder Y

Берутся плечевые keypoints (`left_shoulder`, `right_shoulder`) только если уверенность точки `> 0.4`.

```text
shoulder_y_normalized = mean(shoulder_y) / frame_height
```

Если `shoulder_y_normalized <= SHOULDER_Y_THRESHOLD`, человек считается стоящим по второму методу.

### Метод 3. Shoulder offset against local depth baseline

Для того же ракурса сбоку полезнее сравнивать не абсолютное положение плеч, а насколько плечи выше локальной нормы для этой глубины кадра:

```text
shoulder_offset = local_median_shoulder_y_same_depth - shoulder_y_normalized
```

Если `shoulder_offset >= SHOULDER_OFFSET_THRESHOLD`, человек считается стоящим по третьему методу.

### Комбинация решений

- итоговая `raw_pose` получается голосованием трёх эвристик;
- затем `raw_pose` сглаживается по истории `track id`, чтобы убрать скачки на перекрытиях;
- в отчёт пишутся оба поля: `raw_pose` и финальный `pose`.

## Стартовые пороги

```python
STAND_HEIGHT_RATIO = 1.15
SHOULDER_Y_THRESHOLD = 0.40
LOCAL_DEPTH_WINDOW = 0.12
SHOULDER_OFFSET_THRESHOLD = 0.02
POSE_SMOOTHING_WINDOW = 3
POSE_SCORE_THRESHOLD = 0.15
MOVEMENT_THRESHOLD = 80
STANDING_DURATION_SEC = 10
FRAME_SKIP = 10
KEYPOINT_CONF_THRESHOLD = 0.4
```

Пороги заведомо стартовые. Для каждого нового ракурса их нужно проверять через режим `calibrate`.

## Запуск

### Обработка всех видео из `data`

```bash
.venv\Scripts\python.exe classroom_analytics.py --mode calibrate
.venv\Scripts\python.exe classroom_analytics.py --mode analyze
```

Если в папке `data` лежит несколько файлов `MP4/AVI/MOV/MKV`, скрипт обработает каждый и создаст отдельную папку в `outputs/`.

### Явно указать один файл или каталог

```bash
.venv\Scripts\python.exe classroom_analytics.py --mode calibrate --input data
```

### Полезные аргументы

```bash
.venv\Scripts\python.exe classroom_analytics.py --mode calibrate ^
  --input data ^
  --frame-skip 5 ^
  --max-frames 300 ^
  --stand-height-ratio 1.15 ^
  --shoulder-y-threshold 0.40 ^
  --shoulder-offset-threshold 0.02 ^
  --conf 0.15 ^
  --imgsz 1280 ^
  --device auto
```

Дополнительно:

- `--save-every-frame` — сохранить все исходные кадры в выходное видео, а не только анализируемые;
- `--device auto` — автоматически выбрать `cuda:0`, если CUDA доступна, иначе `cpu`;
- `--max-frames 300` — короткий тестовый прогон на первых кадрах;
- `--local-depth-window 0.12` — окно сравнения только с людьми на похожей глубине кадра;
- `--shoulder-offset-threshold 0.06` — насколько плечи должны быть выше локальной нормы, чтобы считать человека стоящим;
- `--pose-smoothing-window 3` — сколько последних состояний `ID` использовать для сглаживания;
- `--min-state-frames 3` — сколько наблюдений подряд должно держаться новое состояние, чтобы считаться стабильным;
- `--min-event-duration-sec 1.0` — минимальная длина интервала для записи события;
- `--conf 0.15` — более чувствительная детекция людей в плотной сцене;
- `--imgsz 1280` — увеличенный размер входа для лучшей детекции дальних учеников;
- `--tracker botsort.yaml` — конфиг трекера для `ultralytics`;
- `--model yolov8s-pose.pt` — сменить модель на более тяжёлую;
- `--output-dir outputs` — директория результатов.

Для первого запуска на новой машине безопаснее явно указать `--device cpu`. Если CUDA настроена корректно, можно убрать этот флаг или передать `--device cuda:0`.

## Быстрый smoke test

Перед полным прогоном имеет смысл проверить первые 200-300 кадров:

```bash
.venv\Scripts\python.exe classroom_analytics.py --mode calibrate --input data --device cpu --save-every-frame --max-frames 300
```

## Обучение patch classifier

Для камеры сверху-сбоку это более реалистичный путь, чем дальнейшее усложнение эвристик.

### 1. Экспортировать кропы людей из видео

```bash
.venv\Scripts\python.exe export_person_crops.py --input data --frame-step 15 --max-frames 2000 --device cpu
```

Скрипт создаст структуру:

```text
datasets/pose_classifier/unlabeled/
  GXAY9994/
    crops/
    metadata.csv
```

### 2. Разложить кропы по классам вручную

Самый удобный вариант — использовать локальный разметчик:

```bash
.venv\Scripts\python.exe label_pose_crops.py
```

Горячие клавиши:

- `S` — `sit`
- `W` — `stand`
- `U` — `skip`
- `Backspace` — отменить последнюю разметку
- `Q` — выйти

Скрипт:

- читает изображения из `datasets/pose_classifier/unlabeled`
- переносит их в `datasets/pose_classifier/labeled/sit` или `datasets/pose_classifier/labeled/stand`
- сохраняет прогресс в `datasets/pose_classifier/label_progress.csv`

Если хотите разложить всё вручную без GUI, создайте папки:

```text
datasets/pose_classifier/labeled/
  sit/
  stand/
```

И вручную перенесите туда кропы из `unlabeled/crops/`.

Практический минимум:

- `150-300` кропов `sit`
- `150-300` кропов `stand`

Лучше брать именно из вашей камеры и ваших уроков.

### 2.1. Разметка баловства (Behavior Labeling)

Тот же скрипт поддерживает режим разметки баловства (`normal`, `distracted`, `active`). Для этого укажите аргумент `--task behavior`. Скрипт будет использовать другую целевую папку и другие горячие клавиши:

```bash
.venv\Scripts\python.exe label_pose_crops.py --task behavior
```

Горячие клавиши для режима `behavior`:
- `N` — `normal` (ученик сидит ровно, пишет, слушает)
- `D` — `distracted` (отвернулся, лежит на парте)
- `A` — `active` (балуется, дерётся, стоит без причины)
- `U` — `skip` (пропустить)
- `Backspace` — отменить последнюю разметку

Скрипт будет работать с папками:
- Исходники: `datasets/behavior_classifier/unlabeled`
- Разметка: `datasets/behavior_classifier/labeled/{normal, distracted, active}`
- Прогресс: `datasets/behavior_classifier/label_progress.csv`

### 3. Обучить классификатор

```bash
.venv\Scripts\python.exe train_pose_classifier.py --dataset-root datasets\pose_classifier\labeled --model-name efficientnet_b0 --epochs 12 --batch-size 32 --device cpu
```

Если GPU настроен:

```bash
.venv\Scripts\python.exe train_pose_classifier.py --dataset-root datasets\pose_classifier\labeled --model-name efficientnet_b0 --epochs 12 --batch-size 32 --device cuda:0
```

Результат:

- `models/pose_classifier/best_classifier.pt`
- `models/pose_classifier/training_metadata.json`

### 4. Что дальше

После обучения этот классификатор можно подключать в `classroom_analytics.py` как основной источник `sit/stand`, а текущие эвристики оставить как fallback/diagnostics.

### 5. Использование обученной модели в анализе видео

Только классификатор:

```bash
.venv\Scripts\python.exe classroom_analytics.py --mode analyze --input data --pose-classifier models\pose_classifier\best_classifier.pt --classifier-mode classifier_only --classifier-threshold 0.65 --device cuda:0
```

Гибридный режим:

```bash
.venv\Scripts\python.exe classroom_analytics.py --mode analyze --input data --pose-classifier models\pose_classifier\best_classifier.pt --classifier-mode hybrid --classifier-threshold 0.65 --device cuda:0
```

Разница:

- `classifier_only` — поза берётся только из обученной модели;
- `hybrid` — если классификатор уверен, используется он, иначе остаются эвристики.

## Что пишет режим `calibrate`

На каждом анализируемом человеке отображаются:

- `ID`
- итоговая поза
- `h_ratio`
- `shoulder_y`
- `knee_angle` — только как диагностическая метрика, если колено и голеностоп видны

Это позволяет руками подобрать рабочие значения порогов именно под вашу камеру и класс.

## Формат отчёта

### 1. Raw detections

`*_raw_detections.csv/xlsx` содержит сырые наблюдения по анализируемым кадрам:

- `timestamp`
- `timestamp_sec`
- `frame`
- `person_id`
- `track_confidence`
- `pose`
- `raw_pose`
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`
- `relative_height_ratio`
- `shoulder_y_normalized`
- `shoulder_offset`
- `local_height_median`
- `local_shoulder_median`
- `knee_angle`

### 2. State report

`*_report.csv/xlsx` содержит те же поля плюс итоговое сглаженное состояние:

- `timestamp`
- `timestamp_sec`
- `frame`
- `person_id`
- `track_confidence`
- `violation_type`
- `pose`
- `raw_pose`
- `stable_pose`
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`
- `relative_height_ratio`
- `shoulder_y_normalized`
- `shoulder_offset`
- `local_height_median`
- `local_shoulder_median`
- `knee_angle`

### 3. Events

`*_events.csv/xlsx/json` содержит уже склеенные интервалы:

- `person_id`
- `event_type`
- `start_timestamp`
- `end_timestamp`
- `start_frame`
- `end_frame`
- `duration_sec`
- `num_observations`

На шаге 1 поле `violation_type` в основном пустое, кроме случаев длительного стояния в режиме `analyze`.

## Цвета аннотации

- зелёный — `sit`
- оранжевый — `stand`
- серый — `unknown`
- красный — нарушение

## Следующий шаг

После проверки качества `sit/stand` на вашем видео можно расширять логику:

- детекция длительного стояния как полноценного нарушения;
- резкое перемещение между кадрами;
- события вставания с места;
- привязка к зонам парт и рядам для конкретной камеры.
#   P o s e - D e t e c t i o n 
 
 