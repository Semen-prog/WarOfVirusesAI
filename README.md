# WarOfVirusesAI
Искусственный интеллект для игры "Война вирусов" на основе Deep Q-learning.
## Описание игры
Подробное описание правил находится в файле *__war-of-viruses.pdf__*     (протокол взаимодействия можно не читать, он не отсюда ;). 

Для упрощения задачи правила были немного модифицированы:
1. Нет непроходимых клеток.
2. Интеллект начинает игру в левом верхнем углу, его противник - в правом нижнем.

## Структура проекта
Папки:
1. *__source__*: код бота
2. *__wov__*: вспомогателные программы (интерактор/визуализатор)
3. *__wovenv__*: код среды (папка *__env__*) и интеллекта (папка *__ai__*), а также логи интеллекта и бота.

Также есть блокнот *__main.ipynb__* реализующий сам процесс обучения.

## Конфиг
В папке *__wovenv__* есть конфигурационный файл *__init.py__*, в котором необходимо заполнить путь до папки с репозиторием (*root*), размеры поля (*n, m*) и макс. кол-во ходов для каждого игрока (*max_turn*).

## Визуализация
Если в *__wovenv/log/access.log__* записан лог игры (это можно сделать через *__main.ipynb__*), то для визуализации перейдите в папку репозитория
    
```console
$ cd /path/to/repo
```

и запустите

```console
$ java -jar wov/Visualizer.jar wovenv/log/access.log 300
```

где 300 - задержка между отрисовками ходов в миллисекундах.