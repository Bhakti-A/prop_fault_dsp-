| model        | rotor                                           |   accuracy |   balanced_acc |   macro_F1 |   recall_healthy_rotor |   recall_faulty_rotor |
|:-------------|:------------------------------------------------|-----------:|---------------:|-----------:|-----------------------:|----------------------:|
| RandomForest | A                                               |      0.964 |          0.954 |      0.962 |                  1     |                 0.909 |
| RandomForest | B                                               |      0.914 |          0.91  |      0.91  |                  0.931 |                 0.889 |
| RandomForest | C                                               |      0.924 |          0.904 |      0.917 |                  0.996 |                 0.812 |
| RandomForest | D                                               |      0.765 |          0.734 |      0.741 |                  0.878 |                 0.589 |
| RandomForest | ALL (exact 4-bit match, windows)                |      0.597 |        nan     |    nan     |                nan     |               nan     |
| RandomForest | ALL (exact 4-bit match, flights, majority vote) |      0.643 |        nan     |    nan     |                nan     |               nan     |
| RandomForest | ALL (per-rotor acc, flights, majority vote)     |      0.902 |        nan     |    nan     |                nan     |               nan     |