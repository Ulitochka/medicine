Извлечение симптомов из жалоб пациентов.
==========================================

## Порядок действий

- Скачивание докера: `docker pull mikkymouse/medicine`
- Запуск докера: `docker run --rm -it --memory="8g" --cpus="4" mikkymouse/medicine:latest /bin/bash`
- Тестирование rule-based подхода. `./run_rule_based.sh` Запуск тестирования всех правил для обнаружения симптомов.
  Тесты для отдельных правил выключены.    
- Подготовка данных для обучения классификатора. Разбиение и нормализация данных требуют подключения к сети для работы
  компонента исправления ошибок. Файл создан заранее.
- Запуск обучения классификатора. `./run_train.sh`. В файле лога и консоле должны быть результаты прогона на тестовых 
    данных по 10 фоллам.

