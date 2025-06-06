## Общий вывод

- Проведено исследование по данным банковских транзакций, о времени транзакции, суммах и связанные с ними личные данные и данные продавца.

- Выполнена предобработка данных:

    1) Исследованы и заполнены пропуски. Задача в дальнейшем будет доизучаться с целью более точного определения пропущенных значений;

    2) Изучены типы данных, изменены и даны пояснения к изменениям.
    Проанализированы уникальные наименования, проверены дубликаты.

    3) В проекте произведена подготовка параметров для проведения анализа: рассчитан возраст клиента, выделены временные параметры даты транзакций (год, месяц, день, время суток).

- Проведённый анализ позволяет сделать следующие выводы:

  * распределение целевой переменной - 1289169(нормальные) и 7506(мошеннеческие). Можно отметить дисбаланс классов.
  
  * большинство мошеннических транзакций осуществляется  вечером - между 18 и 24 часами и также с 00 часов до 06 утра;
  
  * распределение мошеннических транзакций по полу равномерное
  
  * медианная сумма кражи денег за одну транзакцию - 396, среднее - 531
  
  * больше всего мошеннических транзакций в категориях покупок - grocery_pos,  shopping_net, misc_net
  
  * по доле мошеннических транзакций среди общего кол-ва лидируют штаты: NY, TX, PA
  
  * по доле мошеннечиских транзакций среди общего кол-ва лидируют города: Houston, Warren, Naples
  
  * по доле мошеннических транзакций среди общего кол-ва транзакций на клиента - Chelsea Silva

В течение всего проекта выявлялись моменты, когда можно сделать еще более детальный анализ по группам - транзакциям, возрасту, времени суток и т.д., чтобы оценить особенности более четко.

Поэтому можно предложить дальнейшее исследование факторов и взаимосвязи в рамках признаков, что даст более точное представление о прогнозе мошеннической операции для конкретного клиента.

Предложения для дальнейшей доработки:
- Заполнение merch_zipcode через geopy
- Изучение информации по картам (МСП/банки) через bin checker
- Изучение основных фрод локаций (города/штаты)
- Изучение основных фродеров
