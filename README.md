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

3️⃣ Обучение модели
python train_pose_classifier.py \
    --dataset-root datasets\pose_classifier\labeled \
    --model-name efficientnet_b0 \
    --epochs 12 \
    --batch-size 32

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

Software Engineering Student
AI / Computer Vision Projects
