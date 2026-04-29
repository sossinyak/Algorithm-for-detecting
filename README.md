# Алгоритм обнаружения изменений на разновременных спутников снимках с применением классический методв обработки

Проект сравнивает классические методы change detection на парах снимков `A/B`
с эталонной маской `label`.

Ожидаемая структура датасета:

```text
dataset/
  train|val|test/
    A/
    B/
    label/
```

## Исследуемые методы

В коде реализованы классические методы и комбинированные пайплайны:

- AbsDiff
- LogRatio
- NormDiff
- RGB-CVA
- LAB-CVA
- HSV-CVA
- CLAHE-CVA
- LocalMeanDiff
- EdgeDiff
- LaplacianDiff
- PCA-CVA
- Combined-AbsLogCVA
- Edge-Canny-XOR
- Adaptive PCA-CVA

Для методов предусмотрены настраиваемые этапы: сглаживание, CLAHE, разные способы бинаризации, морфологическая постобработка, фильтрация малых областей, комбинирование score-map и edge-based признаки.

## Полный исследовательский запуск:

```bash
python run_all_experiments.py --data-root data --split-root data --results-dir results
```

Итоговый полный протокол сохраняется в `results/full_protocol`.


## Структура проекта

```text
src/
  analysis/        метрики, parameter study, Monte Carlo, статистика, error analysis, HTML-отчет
  experiments/     отдельные экспериментальные запуски и сравнения методов
  pipelines/       реализации алгоритмов обнаружения изменений
  segmentation/    пороговая бинаризация score-map
  postprocessing/  очистка бинарных масок
  utils/           загрузка данных, конфиги, ввод-вывод изображений
  tools/           подготовка данных и вспомогательная визуализация
docs/              описание проекта и материалы для отчета
results/           результаты экспериментов
data/              исходные и подготовленные датасеты с train/val/test split
```

## Полный запуск

```bash
python run_all_experiments.py --data-root data --split-root data --results-dir results
```

Этапы полного запуска:

1. Проверяется или создается воспроизводимое разбиение `train/val/test`.
2. Для каждого метода выполняется parameter study:
   - систематическая сетка параметров;
   - дополнительные Monte Carlo кандидаты.
3. Лучшие параметры выбираются по F1-score на `val`.
4. Сохраненные лучшие параметры применяются к полному `test`.
5. Для каждого метода считаются Precision, Recall, F1-score, Accuracy и время обработки.
6. Строятся сравнительные графики F1-score.

По умолчанию используются:

```text
max_tune_samples = 24
max_eval_samples = 80
monte_carlo_trials = 48
```

Для более полного прогона можно увеличить выборки

## Отдельные этапы

Создать split:

```bash
python src/tools/data/split_datasets.py --data-root data --output-root data --in-place
```

Запустить parameter study:

```bash
python src/analysis/parameter_analyzer.py --data-root data --results-dir results/full_protocol/parameter_study
```

Построить визуализацию этапов Adaptive PCA-CVA:

```bash
python src/tools/visualization/visualize_adaptive_stages.py --data_path data/LEVIR-CD-filtred --split test
```

При запуске `run_all_experiments.py` такие визуализации автоматически создаются
в `results/<run>/stage_visualization/` и добавляются в HTML-отчет.

Отдельно оценить специальный пайплайн ZScorePCACVA:

```bash
python src/experiments/evaluate_zscore_pca_cva.py --data-root data --split test
```

## Основные артефакты

- `results/full_protocol/parameter_study/all_parameter_trials.csv` - все прогоны сетки и Monte Carlo.
- `results/full_protocol/parameter_study/all_parameter_summary.csv` - лучшие параметры по методам.
- `results/full_protocol/parameter_study/all_pair_metrics.csv` - метрики по отдельным патчам.
- `results/full_protocol/parameter_study/best_params.yaml` - лучшие параметры по датасетам и методам.
- `results/*/runs/*/manifest.json` - журнал запуска: параметры, метрики, этапы и артефакты.
- `results/full_protocol/statistical_validation_*.csv` - bootstrap-интервалы улучшения F1.
- `results/full_protocol/error_analysis/*_error_summary.csv` - типы ошибок.
- `results/full_protocol/research_report.html` - итоговый HTML-отчет рядом с результатами полного протокола.
- `results/full_protocol/stage_visualization/` - промежуточные карты Adaptive PCA-CVA для выбранных test-патчей.
- `results/full_protocol/research_report.html` - HTML-отчет.