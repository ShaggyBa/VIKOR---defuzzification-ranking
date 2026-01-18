# Методы семейства VIKOR, способы преобразования нечетких множеств в интервальные и точные

Используется модификация **alternative-generator** для генерации синтетических данных с нечеткими множествами

Инструмент *VIKOR* предоставляет следующие возможности:

1. Обработка входных данных в формате JSON 
2. Выбор модификации (точная, интервальная, нечеткая) для расчета методом VIKOR
3. Если исходные данные имеют формат нечетких чисел и имеется необходимость преобразовать эти данные в другой формат, то можно выбрать способ деффазификации (a-cut, множественные a-cut, центроиды, граничные значения)
4. Расчет метода и ранжирование альтернатив
5. Вывод результатов в простом и расширенном (поэтапном) виде.

### Использование VIKOR
```python
in_data = load_vikor_json("alternative-generator/example/test2.json")
matrix = in_data["matrix"]
weights = in_data["weights"]
benefit_criteria = in_data["criteria_types"]
v = in_data["v"]
tr_custom_values = in_data["tr_custom_values"]

# Интервальные с простым преобразованием путем извлечения носителя
matrix_interval_simple = np.zeros((matrix.shape[0], matrix.shape[1], 2))
weights_interval = np.zeros((weights.shape[0]))
for i in range(weights_interval.shape[0]):
    weights_interval[i] = weights[i, 1]

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        matrix_interval_simple[i, j] = np.array(fuzzy_to_interval_simple(val[0], val[1], val[2]))

# Интервальные на основе альфа-среза в 0.5
matrix_interval_ac = np.zeros((matrix.shape[0], matrix.shape[1], 2))
weights_interval = np.zeros((weights.shape[0]))
for i in range(weights_interval.shape[0]):
    weights_interval[i] = weights[i, 1]

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        matrix_interval_ac[i, j] = np.array(fuzzy_to_interval_ac(val[0], val[1], val[2]))

# Интервальные на основе множественных альфа-срезов
matrix_interval_wac = np.zeros((matrix.shape[0], matrix.shape[1], 2))
weights_interval = np.zeros((weights.shape[0]))
for i in range(weights_interval.shape[0]):
    weights_interval[i] = weights[i, 1]

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        matrix_interval_wac[i, j] = np.array(fuzzy_to_interval_wac(val[0], val[1], val[2]))
```



Модификация **alternative-generator**:

1. Добавлен параметр к конструктору AlternativeGenerator - *fuzzy_numbers: boolean*, который позволяет сгенерировать нечеткие числа
2. Генерация нечетких чисел использует сгенерированное по умолчанию число, затем вычисляет разбросы для левой и правой границы нечеткого числа. 
3. На основе вычисленных разбросов генерируются границы нечеткого числа, центральным числом остается сгенерированное по умолчанию число.
4. Добавлен экспорт сгенерированных данных в формат json, который использовался для экспериментов в инструменте *VIKOR*

### Использование модифицированного **alternative-generator**

```python
generator = AlternativeGenerator(
    num_criteria=8,
    num_alternatives=10,
    abs_ratio=0.7,
    abs_value_range=(-30, 50),
    ord_value_range=8,
    abs_distribution="uniform",
    ord_distribution="uniform",
    avg_group_size=2,
    group_size_std=1,
    fuzzy_numbers=True,
)
generator.generate_criteria()
alternatives = generator.generate_alternatives()
preferences = generator.generate_preferences()
generator.export_to_json("test.json")
```

### Выходной файл:

```JSON
{
  "alternatives": [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6"
  ],
  "criteria": [
    {
      "name": "criterion1",
      "type": "positive",
      "weight": 1
    },
    {
      "name": "criterion2",
      "type": "positive",
      "weight": 1
    },
    {
      "name": "criterion3",
      "type": "negative",
      "weight": 1
    },
    {
      "name": "criterion4",
      "type": "negative",
      "weight": 1
    },
    {
      "name": "criterion5",
      "type": "negative",
      "weight": 1
    },
    {
      "name": "criterion6",
      "type": "negative",
      "weight": 1
    },
    {
      "name": "criterion7",
      "type": "positive",
      "weight": 1
    },
    {
      "name": "criterion8",
      "type": "negative",
      "weight": 1
    }
  ],
  "dms": [
    {
      "scores": [
        [
          [
            40.64,
            41.72,
            42.28
          ],
          [
            26.72,
            27.44,
            28.41
          ],
          [
            16.74,
            22.96,
            24.85
          ],
          [
            "1",
            "1",
            "2"
          ],
          [
            23.71,
            23.96,
            24.02
          ],
          [
            33.19,
            38.7,
            45.46
          ],
          [
            "1",
            "2",
            "2"
          ],
          [
            "1",
            "1",
            "1"
          ]
        ],
        [
          [
            42.32,
            43.5,
            43.5
          ],
          [
            26.38,
            27.27,
            28.38
          ],
          [
            17.3,
            19.17,
            22.67
          ],
          [
            "3",
            "3",
            "3"
          ],
          [
            23.82,
            24.2,
            24.44
          ],
          [
            25.55,
            26.69,
            30.72
          ],
          [
            "1",
            "2",
            "2"
          ],
          [
            "1",
            "2",
            "3"
          ]
        ],
        [
          [
            39.58,
            40.08,
            41.31
          ],
          [
            28.09,
            28.85,
            30.21
          ],
          [
            8.81,
            9.4,
            10.3
          ],
          [
            "1",
            "1",
            "2"
          ],
          [
            23.31,
            23.34,
            23.66
          ],
          [
            8.33,
            8.39,
            16.01
          ],
          [
            "1",
            "1",
            "1"
          ],
          [
            "1",
            "1",
            "2"
          ]
        ],
        [
          [
            40.94,
            42.17,
            43.0
          ],
          [
            31.34,
            31.62,
            31.75
          ],
          [
            19.17,
            24.8,
            26.01
          ],
          [
            "1",
            "1",
            "3"
          ],
          [
            23.31,
            23.36,
            23.74
          ],
          [
            15.98,
            19.5,
            24.84
          ],
          [
            "1",
            "1",
            "1"
          ],
          [
            "2",
            "3",
            "3"
          ]
        ],
        [
          [
            41.2,
            42.49,
            43.24
          ],
          [
            29.81,
            30.38,
            30.54
          ],
          [
            21.76,
            23.17,
            25.4
          ],
          [
            "3",
            "3",
            "3"
          ],
          [
            24.13,
            24.22,
            24.59
          ],
          [
            31.16,
            36.75,
            42.24
          ],
          [
            "1",
            "1",
            "1"
          ],
          [
            "1",
            "1",
            "1"
          ]
        ],
        [
          [
            41.47,
            41.94,
            43.12
          ],
          [
            25.05,
            25.14,
            25.25
          ],
          [
            -3.61,
            -1.61,
            2.14
          ],
          [
            "2",
            "2",
            "2"
          ],
          [
            23.17,
            23.17,
            23.41
          ],
          [
            38.31,
            38.39,
            43.39
          ],
          [
            "1",
            "1",
            "2"
          ],
          [
            "1",
            "2",
            "2"
          ]
        ]
      ]
    }
  ]
}
```

## Зависимости
Стандартная библиотека Python 3.9+.
Пакеты для VIKOR: numpy, json, typing

Пакеты для alternative_generator: pandas, numpy, matplotlib, json

## Гарантии / Warranty
Разработчики не дают никаких гарантий по поводу использования данного программного обеспечения.

The contributors give no warranty for the using of the software.



## Лицензия / License
Эта программа открыта для использования и распространяется под лицензией GNU General Public License v3.0.

This program is open to use anywhere and is licensed under the GNU General Public License v3.0.
