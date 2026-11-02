| model        | rotor                                           |   accuracy |   balanced_acc |   macro_F1 |   recall_healthy_rotor |   recall_faulty_rotor |
|:-------------|:------------------------------------------------|-----------:|---------------:|-----------:|-----------------------:|----------------------:|
| RandomForest | A                                               |      0.956 |          0.945 |      0.953 |                  0.994 |                 0.897 |
| RandomForest | B                                               |      0.875 |          0.866 |      0.869 |                  0.91  |                 0.822 |
| RandomForest | C                                               |      0.87  |          0.841 |      0.856 |                  0.974 |                 0.708 |
| RandomForest | D                                               |      0.629 |          0.592 |      0.592 |                  0.764 |                 0.42  |
| RandomForest | ALL (exact 4-bit match, windows)                |      0.411 |        nan     |    nan     |                nan     |               nan     |
| RandomForest | ALL (exact 4-bit match, flights, majority vote) |      0.429 |        nan     |    nan     |                nan     |               nan     |
| RandomForest | ALL (per-rotor acc, flights, majority vote)     |      0.848 |        nan     |    nan     |                nan     |               nan     |