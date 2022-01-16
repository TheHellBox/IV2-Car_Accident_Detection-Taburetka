# Хакатон IVision 2.0. Команда Табуретка.
# Определение наличия ДТП на записях с камер видеонаблюдения

## Вступление
Мы смогли составить модель способную определять наличие ДТП на записях с точностью ~75%
(Преведены значения с записей данных организаторами).
Наш метод заключается в использовании нейронной сети обученной методом переноса на датасете состоящем из более чем 6 тысяч снимков
включащих в себя 5 различных классов. Датасет состоит из снимков составленных на основе предоставленных организаторами записей,
а так же снимков сделанных в автосимуляторе BeamNG.drive(С полученным разрешением).
Использованные снимки имеют разрешение 96x96 пикселей и включают в себя элементы разбитых автомобилей, элементы целых автомобилей,
снимки пешеходов и полиции, а так же окружения и дорог.
Видеозапись делится на куски небольшого разрешения(~100 кусков на запись с соотношением сторон 16:9) и подается на вход нейронной сети.
Для подавления ложных сигналов, а так же для большего шанса поиска элементов ДТП при неудачном положении кусков, 
эта процедура повторяется дважды, второй раз со смещением(Если возможно - 64 пикселя). Значения сигналов с двух попыток комбинируются(Среднее значение),
и в дальшейшем используются в общем рейтинге для каждого кадра записи(Используется по 1 кадру на 1 секунду видеоролика)
На основе графика общего рейтинга каждого из кадров состовляется решение об наличии ДТП на записи. 

## Обучение нейронной сети
Из за ограничений датасета обучение нейронной сети проводилось методом переноса. В качестве основы нейронной сети была выбранна EfficientNetB0
(Так же были протестированны варианты на основе MobileNetV2 и Xception, B7 и B3. Стоит отметить, что больших отличий B3 и B7 от B0 не показывает),
с весами обученными на датасете imagenet. Первые 150 слоев нейронной сети остаются замороженными, остальные слои обучаются на нашем датасете. Использован оптимизатор Adam с коофициентом обучения 8e-6.
Всего используется 13 циклов обучения. Итоговые значения точности и потерь на клалификационном датасете составляют 0.90 и 0.32 соответственно.

![alt text](training_performance.png?raw=true)

## Использование
### Зависимости:
tensorflow, matplotlib, numpy, pillow, opencv3(Для чтения видео)
### Запуск
`python3 video_graph.py {filename}` - построение графика вероятности ДТП

`python3 detect_on_frame.py` - Утилита позволяющая просматривать как нейронная сеть реагирует на тот или иной кадр. Файл обязан называтся test.png

`python3 test_suit.py {video}` - Проверка записей на наличие в ней ДТП и построение статистики. Использует видео из папки test и файл truth.txt

`python3 check_videos.py {folder}` - Проверяет видеоролики в папке на наличие в них ДТП. В отличии от test_suit выдает результат в виде файла с отметками возможных моментов с ДТП 