this is a project task for arabot

the task is to use a machinelearning classfication and regression models to predict new data

libraries used
python = "^3.12"
numpy = "^1.26.4"
pandas = "^2.2.2"
matplotlib = "^3.8.4"
tensorflow = "^2.16.1"
scikit-learn = "^1.4.2"
gradio = "^4.31.4"


the data used were for heart attack analysis & prediction
the project tries to to predict weather a paitent will have a high chance of a heart attack or not using Gaussian probability and logistic regression
using a gradio web platform

the data came up very small so i'll include some inputs for the users choice, you can still make up your own data

output = 1, high chance of heart attack
output = 0, low chance of heart attack

+-----+-----+----+-------+-----+-----+--------+---------+-----+--------+-----+-----+-----+--------+
| age | sex | cp | trtbps| chol| fbs | restecg| thalachh| exng| oldpeak| slp | caa | thall| output |
+-----+-----+----+-------+-----+-----+--------+---------+-----+--------+-----+-----+-----+--------+
|  63 |  1  |  3 |   145 | 233 |  1  |    0   |   150   |  0  |   2.3  |  0  |  0  |  1  |    1   |
|  37 |  1  |  2 |   130 | 250 |  0  |    1   |   187   |  0  |   3.5  |  0  |  0  |  2  |    1   |
|  41 |  0  |  1 |   130 | 204 |  0  |    0   |   172   |  0  |   1.4  |  2  |  0  |  2  |    1   |
|  56 |  1  |  1 |   120 | 236 |  0  |    1   |   178   |  0  |   0.8  |  2  |  0  |  2  |    1   |
|  57 |  0  |  0 |   120 | 354 |  0  |    1   |   163   |  1  |   0.6  |  2  |  0  |  2  |    1   |
|  57 |  1  |  0 |   140 | 192 |  0  |    1   |   148   |  0  |   0.4  |  1  |  0  |  1  |    1   |
|  56 |  0  |  1 |   140 | 294 |  0  |    0   |   153   |  0  |   1.3  |  1  |  0  |  2  |    1   |
|  44 |  1  |  1 |   120 | 263 |  0  |    1   |   173   |  0  |   0    |  2  |  0  |  3  |    1   |
+-----+-----+----+-------+-----+-----+--------+---------+-----+--------+-----+-----+-----+--------+
|  47 |  1  |  0 |   110 | 275 |  0  |    0   |   118   |  1  |   1    |  1  |  1  |  2  |    0   |
|  52 |  1  |  0 |   125 | 212 |  0  |    1   |   168   |  0  |   1    |  2  |  2  |  3  |    0   |
|  58 |  1  |  0 |   146 | 218 |  0  |    1   |   105   |  0  |   2    |  1  |  1  |  3  |    0   |
|  57 |  1  |  1 |   124 | 261 |  0  |    1   |   141   |  0  |   0.3  |  2  |  0  |  3  |    0   |
|  58 |  0  |  1 |   136 | 319 |  1  |    0   |   152   |  0  |   0    |  2  |  2  |  2  |    0   |
|  61 |  1  |  0 |   138 | 166 |  0  |    0   |   125   |  1  |   3.6  |  1  |  1  |  2  |    0   |
|  42 |  1  |  0 |   136 | 315 |  0  |    1   |   125   |  1  |   1.8  |  1  |  0  |  1  |    0   |
|  52 |  1  |  0 |   128 | 204 |  1  |    1   |   156   |  1  |   1    |  1  |  0  |  0  |    0   |
|  59 |  1  |  2 |   126 | 218 |  1  |    1   |   134   |  0  |   2.2  |  1  |  1  |  1  |    0   |
|  40 |  1  |  0 |   152 | 223 |  0  |    1   |   181   |  0  |   0    |  2  |  0  |  3  |    0   |
|  61 |  1  |  0 |   140 | 207 |  0  |    0   |   138   |  1  |   1.9  |  2  |  1  |  3  |    0   |
|  46 |  1  |  0 |   140 | 311 |  0  |    1   |   120   |  1  |   1.8  |  1  |  2  |  3  |    0   |
|  59 |  1  |  3 |   134 | 204 |  0  |    1   |   162   |  0  |   0.8  |  2  |  2  |  2  |    0   |
+-----+-----+----+-------+-----+-----+--------+---------+-----+--------+-----+-----+-----+--------+












