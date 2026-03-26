# Classroom Video Analytics

Локальный Python-пайплайн для анализа видеоуроков.

Система умеет:
- детектировать и трекать учеников;
- определять `sit / stand`;
- определять `focused / distracted`;
- определять `fight / non-fight` по сцене и локализовать вероятных участников;
- фиксировать `crowding` для групп из `3+` человек;
- собирать события в Excel и сохранять аннотированное видео.

## Структура

- `classroom_analytics.py` — основной пайплайн анализа
- `train_pose_classifier.py` — обучение модели `sit / stand`
- `train_behavior_classifier.py` — frame-based обучение модели поведения
- `train_distracted_classifier.py` — frame-based обучение модели `focused / distracted`
- `train_fight_classifier.py` — обучение модели `fight / non-fight`
- `export_person_crops.py` — экспорт кропов для `sit / stand`
- `export_track_clips.py` — экспорт кропов для `focused / distracted`
- `label_pose_crops.py` — разметка датасетов
- `outputs/` — результаты анализа

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision
```

Для аудио-модели нужен `ffmpeg` в `PATH` или явный путь через `--ffmpeg-bin`.

## Запуск анализа

Полный запуск со всеми моделями:

```powershell
.\.venv\Scripts\python.exe .\classroom_analytics.py `
  --mode analyze `
  --input .\data `
  --model .\yolov8s-pose.pt `
  --pose-classifier .\models\pose_classifier\best_classifier.pt `
  --behavior-classifier .\models\behavior_classifier\best_behavior_classifier.pt `
  --fight-classifier .\models\fight_classifier\best_fight_classifier.pt `
  --sound-classifier .\models\sound_classifier\classifier.pt `
  --behavior-threshold 0.8 `
  --fight-threshold 0.9 `
  --sound-threshold 0.8 `
  --sound-window-sec 1.0 `
  --fight-participant-proximity-px 140 `
  --fight-participant-movement-px 45 `
  --crowding-proximity-px 120 `
  --crowding-min-duration-sec 3 `
  --device cuda:0
```

Быстрый smoke-run:

```powershell
.\.venv\Scripts\python.exe .\classroom_analytics.py `
  --mode analyze `
  --input .\data `
  --model .\yolov8s-pose.pt `
  --pose-classifier .\models\pose_classifier\best_classifier.pt `
  --behavior-classifier .\models\behavior_classifier\best_behavior_classifier.pt `
  --fight-classifier .\models\fight_classifier\best_fight_classifier.pt `
  --sound-classifier .\models\sound_classifier\classifier.pt `
  --behavior-threshold 0.8 `
  --fight-threshold 0.9 `
  --sound-threshold 0.8 `
  --device cuda:0 `
  --frame-skip 2 `
  --imgsz 960 `
  --max-frames 300
```

## Что сохраняется

Для каждого видео создаётся папка:

```text
outputs/<video_name>_analyze/
```

Внутри:
- `*_annotated.mp4` — аннотированное видео
- `*_report.xlsx` — Excel-отчёт

В отчёте есть:
- `summary`
- `per_person`
- `events`

Основные события:
- `standing_interval`
- `rapid_motion_interval`
- `distracted_interval`
- `crowding_interval`
- `fight_interval`
- `horseplay_interval`
- `sound_interval`

## Датасеты

### Sit / Stand

```text
datasets/pose_classifier/labeled/
  sit/
  stand/
```

Обучение:

```powershell
.\.venv\Scripts\python.exe .\train_pose_classifier.py `
  --dataset-root .\datasets\pose_classifier\labeled `
  --output-dir .\models\pose_classifier `
  --model-name efficientnet_b0 `
  --epochs 12 `
  --batch-size 32 `
  --device cuda:0
```

### Behavior

Frame-based датасет:

```text
datasets/behavior_classifier/labeled/
  focused/
  distracted/
```

Обучение:

```powershell
.\.venv\Scripts\python.exe .\train_behavior_classifier.py `
  --dataset-root .\datasets\behavior_classifier\labeled `
  --output-dir .\models\behavior_classifier `
  --model-name efficientnet_b0 `
  --epochs 12 `
  --batch-size 32 `
  --device cuda:0 `
  --resume-from .\models\balovanie_classifier\balovanie_classifier.pt `
  --freeze-backbone-epochs 3
```

### Distracted

Frame-based датасет:

```text
datasets/distracted_classifier/labeled/
  focused/
  distracted/
```

Обучение:

```powershell
.\.venv\Scripts\python.exe .\train_distracted_classifier.py `
  --dataset-root .\datasets\distracted_classifier\labeled `
  --output-dir .\models\distracted_classifier `
  --model-name resnet18 `
  --epochs 12 `
  --batch-size 32 `
  --device cuda:0
```

### Fight

Датасет `RWF-2000`:

```text
datasets/RWF-2000/
  train/
    Fight/
    NonFight/
  val/
    Fight/
    NonFight/
```

Обучение:

```powershell
.\.venv\Scripts\python.exe .\train_fight_classifier.py `
  --dataset-root .\datasets\RWF-2000 `
  --output-dir .\models\fight_classifier `
  --model-name r3d_18 `
  --epochs 6 `
  --batch-size 4 `
  --workers 2 `
  --device cuda:0 `
  --num-frames 16 `
  --image-size 112
```

## Текущая логика событий

- `distracted` подтверждается только после `3` секунд непрерывного удержания.
- `crowding` подтверждается только для группы из `3+` человек.
- если ученики просто сидят рядом за одной партой и почти не двигаются, это не считается `crowding`.
- `fight` сначала определяется глобальной видеомоделью, затем локализуется на вероятных участниках по близости и агрессивной динамике.

## Проверка перед сдачей

Минимальный чек-лист:
- проверить запуск `classroom_analytics.py` на `100-300` кадрах;
- убедиться, что создаются `mp4` и `xlsx`;
- проверить, что в `events` появляются ожидаемые интервалы;
- проверить, что `fight` не красит всех детей в кадре;
- проверить, что `crowding` не срабатывает на обычных соседей по парте;
- проверить, что `distracted` не шумит на коротких ложных всплесках.
