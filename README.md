# Task 5: Decision Trees and Random Forests – Heart Disease Classification ❤️

## ✅ What I Did:
- Loaded the Heart Disease Dataset
- Trained a Decision Tree with depth control
- Visualized the decision tree (`decision_tree.png`)
- Trained a Random Forest and compared performance
- Visualized feature importances (`feature_importance.png`)
- Evaluated both using classification report and cross-validation

## 📊 Results:
| Model            | Accuracy | CV Accuracy |
|------------------|----------|-------------|
| Decision Tree    | 80%      | 83%         |
| Random Forest    | 99%      | 100%        |

## 💡 Key Insights:
- Random Forest performed significantly better
- Decision Trees can overfit without depth control
- Feature importance showed that `ca`, `thal`, and `oldpeak` were strong predictors

## 📁 Files Included:
- `tree_models.py`: Full training and evaluation code
- `decision_tree.png`: Visual representation of the decision tree
- `feature_importance.png`: Random Forest feature impact
- `heart.csv`: The dataset used
