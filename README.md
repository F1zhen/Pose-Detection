🎥 Classroom Video Analytics

Classroom Video Analytics — это локальный Python-пайплайн для анализа видеозаписей уроков.

Система автоматически:

обнаруживает людей в кадре

отслеживает их между кадрами

определяет позу sit / stand с помощью обученной нейросетевой модели

Проект предназначен для анализа поведения учеников на уроках и может быть расширен для обнаружения событий (вставание, перемещение, нарушения дисциплины).

🚀 Возможности

📹 Поддержка видеоформатов
MP4, AVI, MOV, MKV

🧍 Детекция людей
YOLOv8

🔁 Трекинг людей между кадрами
BoT-SORT tracker

🧠 Определение позы
нейросетевой классификатор sit/stand

🎥 Генерация аннотированного видео

📊 Экспорт аналитических отчётов

🧠 Как работает система

Pipeline анализа:

Video
   ↓
YOLOv8 person detection
   ↓
Multi-object tracking (BoT-SORT)
   ↓
Person crops extraction
   ↓
Neural pose classifier
   ↓
Temporal smoothing
   ↓
Reports + annotated video

Каждый обнаруженный человек:

1️⃣ детектируется моделью YOLOv8
2️⃣ получает уникальный track_id
3️⃣ его кроп передаётся в pose classifier
4️⃣ классификатор определяет:

sit
stand

Для стабильности результаты сглаживаются по истории track_id.

📂 Структура проекта
Video-Analytics/

classroom_analytics.py
    основной pipeline анализа видео

export_person_crops.py
    экспорт кропов людей из видео

label_pose_crops.py
    инструмент для ручной разметки

train_pose_classifier.py
    обучение модели sit/stand

requirements.txt
    зависимости

data/
    исходные видео

outputs/
    результаты анализа

datasets/pose_classifier/
    датасет для обучения

models/pose_classifier/
    обученные веса модели
⚙️ Установка

Создание виртуального окружения:

python -m venv .venv

Активация:

.venv\Scripts\activate

Установка зависимостей:

pip install -r requirements.txt

Требования:

Python 3.10+

CUDA (опционально для GPU)

PyTorch

Ultralytics YOLOv8

▶️ Запуск анализа

Обработка всех видео из папки data:

python classroom_analytics.py --mode analyze

Указать конкретный файл:

python classroom_analytics.py --mode analyze --input data

Пример запуска:

python classroom_analytics.py \
    --mode analyze \
    --input data \
    --pose-classifier models/pose_classifier/best_classifier.pt \
    --device cuda:0
🧪 Быстрый тест

Для тестового запуска на первых кадрах:

python classroom_analytics.py \
    --mode analyze \
    --input data \
    --max-frames 300 \
    --device cpu
🎥 Выходные данные

После анализа создаётся папка:

outputs/

Внутри:

📹 Annotated video

Видео с наложенными:

bounding boxes

ID человека

позой (sit / stand)

таймкодом

📊 Raw detections
*_raw_detections.csv

Содержит наблюдения по кадрам:

timestamp

frame

person_id

pose

bbox coordinates

classifier confidence

📈 State report
*_report.csv

Содержит:

сглаженную позу

стабильное состояние

📅 Events
*_events.csv

Интервалы событий:

person_id

event_type

start_time

end_time

duration

🧠 Обучение классификатора

Для повышения качества позы используется обученная модель классификации изображений.

1️⃣ Экспорт кропов людей

Извлечение изображений людей из видео:

python export_person_crops.py \
    --input data \
    --frame-step 15 \
    --max-frames 2000

Создаётся структура:

datasets/pose_classifier/unlabeled/

video_name/
    crops/
    metadata.csv
2️⃣ Разметка кропов

Запуск разметчика:

python label_pose_crops.py

Горячие клавиши:

S → sit
W → stand
U → skip
Backspace → undo
Q → quit

Минимальный датасет:

150–300 sit
150–300 stand

Рекомендуется использовать данные с той же камеры, где будет происходить анализ.

<<<<<<< HEAD
3️⃣ Обучение модели
python train_pose_classifier.py \
    --dataset-root datasets\pose_classifier\labeled \
    --model-name efficientnet_b0 \
    --epochs 12 \
    --batch-size 32
=======
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
>>>>>>> balovanie

Результат:

models/pose_classifier/best_classifier.pt
models/pose_classifier/training_metadata.json
🎨 Цвета аннотации
Цвет	Состояние
🟢	sit
🟠	stand
⚪	unknown
🔴	violation
🔜 Roadmap

Планируемые расширения:

обнаружение длительного стояния

детекция резкого перемещения

обнаружение драк

анализ движения между рядами

определение нарушений дисциплины

🧑‍💻 Автор

Zhenis Esimbekov

<<<<<<< HEAD
Software Engineering Student
AI / Computer Vision Projects
=======
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

## Использовал Claude 4.6/GPT 5.4 Codex/Открытые источники из интернета
>>>>>>> balovanie
