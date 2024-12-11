На данном этапе построения безлайна выбрана модель логистической регрессии, так как задача - классификация.
Рассмотрены: 
 - простая модель логистической регресии
 - Log Reg с параметрами class_weight': "balanced"
 - Log REg c параметрами: class_weight: balanced, penalty: l2, C: 0.01, solver: sag
 - Log Reg с параметрами: class_weight: balanced, penalty: l1, C: 0.01, solver: liblinear

Применены энкодеры:
 - TargetEncoder
 - LeaveOneOutEncoder
 - WOEEncoder
 - CatBoostEncoder

Для каждого подхода рассчитаны метрики:
 - f1
 - precision 
 - recall
 - ROC_AUC
 - PR_AU

Датасет с существенным дисбалансом классов 99% к 1%. Поэтому не все метрики подходят для измерения.
Наиболее лучшими являются ROS_AUC 

Построены пайплайны.

По итогам результатов на данном этапе лучший результат по метрике ROC_AUC получается Log Reg с параметрами: class_weight: Log Reg с параметрами: class_weight: balanced, penalty: l1, C: 0.01, solver: liblinear:
ROC_AUC = 0.965324
