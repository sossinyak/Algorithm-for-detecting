# Описание проекта

## Тема

Сравнение классических методов обнаружения изменений на парах изображений до/после (`A/B`) с бинарной эталонной маской (`label`).

## Цель

Построить воспроизводимый исследовательский протокол: разделить данные на `train/val/test`, подобрать параметры классических методов на `val`, сравнить методы по F1-score и собрать HTML-отчет с таблицами, графиками и промежуточными картами обработки.

## Датасеты

| Датасет | Train | Val | Test |
|---|---:|---:|---:|
| JL1-CD | 700 | 150 | 150 |
| LEVIR-CD-filtred | 4536 | 1022 | 1032 |
| synthetic-lab | 6 | 1 | 2 |

## Методы

В протокол входят AbsDiff, LogRatio, NormDiff, RGB-CVA, LAB-CVA, HSV-CVA, CLAHE-CVA, LocalMeanDiff, EdgeDiff, LaplacianDiff, PCA-CVA, Combined-AbsLogCVA, Edge-Canny-XOR и Adaptive PCA-CVA.

## Протокол

1. Параметры подбираются на `val`.
2. Итоговая оценка выполняется на `test`.
3. Для подбора используется сетка параметров и Monte Carlo кандидаты.
4. Сохраняются лучшие параметры в YAML.
5. Строятся графики F1-score.
6. Выполняется bootstrap-проверка улучшений.
7. Строятся промежуточные визуализации Adaptive PCA-CVA.
8. Все результаты собираются в HTML-отчет.

## Последний полный запуск

Актуальный запуск находится в:

```text
results/full_protocol/
```

Журналы воспроизводимости:

```text
results/full_protocol/runs/20260429_170222_research_protocol/
results/full_protocol/parameter_study/runs/20260429_170224_parameter_study/
```

Лучшие методы по последнему запуску:

| Датасет | Лучший метод | Precision | Recall | F1-score | Samples |
|---|---|---:|---:|---:|---:|
| JL1-CD | Adaptive PCA-CVA | 0.2265 | 0.3397 | 0.2718 | 150 |
| LEVIR-CD-filtred | Adaptive PCA-CVA | 0.2251 | 0.4339 | 0.2964 | 1032 |
| synthetic-lab | Adaptive PCA-CVA | 0.9912 | 0.9111 | 0.9495 | 2 |

## Отчет

HTML-отчет для просмотра:

```text
docs/research_report.html
```

Каноническая версия рядом с результатами:

```text
results/full_protocol/research_report.html
```
