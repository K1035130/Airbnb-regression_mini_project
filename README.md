# NYC Airbnb Popularity Prediction 🏠
 
Predicting **`reviews_per_month`** (a proxy for listing popularity) for New York City Airbnb
listings using a full supervised-learning regression pipeline — from EDA and feature
engineering to model comparison, hyperparameter tuning, and SHAP-based interpretation.
 
> Course mini-project for **CPSC 330 – Applied Machine Learning** (UBC). Open-ended,
> project-style assignment covering everything up to tree-based ensembles and model interpretation.
 
---
##  Problem
 
Given a set of Airbnb listing attributes (room type, location, availability, review history,
host info, etc.), predict how many reviews per month a listing receives. This kind of model
could help hosts estimate the likely popularity of a new listing *before* it goes live.
 
It's framed as a **regression** problem, evaluated primarily with **R²**, supported by **MAE**
and **RMSE**.

---
 
## Dataset
 
- Source: [New York City Airbnb Open Data (Kaggle)](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)
  / [Inside Airbnb](http://insideairbnb.com/) `listings.csv`
- ~453 raw listings, 84 features + 1 target
- **Target:** `reviews_per_month` (right-skewed; most values fall between 0–2)
- Notable challenges: many empty/constant columns, missing target values for some rows,
  and lots of irrelevant identifiers/URLs that need cleaning
> Place `listings.csv` in the project root before running the notebook.
--- 

##  Pipeline & Workflow
 
1. **EDA** — feature types, missingness, target distribution, breakdown by `room_type`
2. **Feature engineering** — derived features:
   - `beds_per_room` = beds / (bedrooms + 1)
   - `has_multiple_bathrooms` = bathrooms > 1
   - `review_momentum` = reviews_l30d / (reviews_ltm + 1)
3. **Preprocessing** (`ColumnTransformer`):
   - Numeric → median imputation + `StandardScaler`
   - Categorical → constant imputation + `OneHotEncoder(handle_unknown="ignore")`
   - Dropped IDs, URLs, and uninformative columns
4. **Baseline** — `DummyRegressor` (predict-the-mean)
5. **Linear model** — `Ridge`, with `alpha` tuned (best ≈ 41)
6. **Other models** — Random Forest, Gradient Boosting, LightGBM
7. **Feature selection** — `RFECV` (selected ~19 features)
8. **Hyperparameter tuning** — `GridSearchCV` over Random Forest (4 params, 48 candidates)
9. **Interpretation** — `SHAP` waterfall plots on individual predictions
10. **Final evaluation** — best model applied to the held-out test set
---
 
##  Results
 
5-fold cross-validation (train) and final held-out test scores:
 
| Model                       | CV R²   | CV MAE  |
| --------------------------- | ------- | ------- |
| Dummy (mean)                | 0.00    | ~1.28   |
| Ridge (alpha = 41)          | 0.820   | 0.49    |
| LightGBM                    | ~0.81   | low     |
| Gradient Boosting           | >0.83   | lowest  |
| **Random Forest (tuned)**   | **0.838** | **best** |
 
**Final model:** Random Forest (`max_depth=15`, `min_samples_leaf=1`,
`min_samples_split=5`, `n_estimators=400`)
 
**Test set performance:**
 
| Metric | Score  |
| ------ | ------ |
| R²     | 0.7704 |
| MAE    | 0.6558 |
 
The ~0.06 drop from CV to test suggests mild overfitting / optimization bias, which is
discussed in the notebook.
 
---
 
##  Key Findings
 
- **Review-related features dominate.** SHAP shows `number_of_reviews_ltm`,
  `number_of_reviews_ly`, `number_of_reviews_l30d`, and the engineered `review_momentum`
  drive most predictions.
- Location, cleanliness, and guest-experience scores had comparatively small effects.
- Tree-based ensembles outperformed the linear baseline, but only modestly — good features
  mattered more than model complexity.
- Feature selection barely changed performance while improving interpretability.

## 🚀 How to Run
 
```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn lightgbm shap matplotlib seaborn
 
# 2. Add the dataset
#    place listings.csv in the project root
 
# 3. Open the notebook
jupyter lab airbnb_miniproj.ipynb
#    Kernel → Restart Kernel and Run All Cells
```
##  Acknowledgements
 
Completed as Homework 5 for **CPSC 330 – Applied Machine Learning** at the University of
British Columbia. Dataset courtesy of Kaggle / Inside Airbnb.

 
