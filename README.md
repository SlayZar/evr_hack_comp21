# evr_hack_comp21

## Модель предсказания содержания углерода и температуры чугуна во время процесса продувки металла.

https://reg-hackathon.evraz.com/

***Public score:*** 0.6358974358974359

***Private Score:*** 0.6198717948717949

#### 3 место по прайват Лидерборду и 6 итоговое место после защиты

## Решение:

***1 этап*** - агрегация фичей из всех доступных таблиц. 

Ноутбук creating_features.ipynb расчитывает необходимые фичи, используя функции из feature_generates.py

Ноутбук подтягивает данные и клонирует данный репозиторий, чтобы использовать необходимые функции для агрегации данных в выданных таблицах.

На выходе таблица вида [NPLV - множество фичей], которая сохраняется в папку data

Время работы скрипта для train данных ~ 30-40 минут.

***2 этап*** - обучение моделей.

В ноутбуке evraz_lama_model.ipynb берём признаки, полученные выше, и запускаем LightAutoML для каждого таргета.

В модель предсказаниии температуры чугуна как фичу подаём содержание углерода, которую при скоринге предсказываем первой моделью.

В ноутбуке предусмотрено сохранение моделей в папку models (можно взять модели оттуда и проскорить данные, не обучая модели заново).

Также результат скоринга сохраняется в папку solutions.

Презентация: https://docs.google.com/presentation/d/1qTszORkDFR5nCedjC2CQLFvpB0RKq5Uj8-QkaYkWUq4/edit?usp=sharing
