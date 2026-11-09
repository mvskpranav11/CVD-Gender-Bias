# Gender-Aware Cardiovascular Disease Prediction

This is our research internship (Forschungspraktikum) project from the M.Sc. Web and Data Science program at the University of Koblenz (March 2026).

We looked at how heart disease prediction models treat male and female patients. Women make up only about 21% of the dataset, and the baseline models missed many more sick women than sick men. We then tried different ways to reduce that gap.

## Dataset

We used the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) by fedesoriano on Kaggle. It combines five older heart disease datasets from the UCI repository.

- 918 patients (725 men, 193 women)
- 11 features: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
- Target: `HeartDisease` (1 = heart disease, 0 = no heart disease)

## What we did

1. Cleaned the data (filled zero cholesterol and blood pressure values, handled outliers, transformed skewed features).
2. Trained 5 baseline models: Logistic Regression, Random Forest, SVM, Decision Tree and Gradient Boosting.
3. Ran EDA and compared feature importance for men vs. women.
4. Tried several fixes: oversampling, SMOTE on female data, sample reweighting, Fairlearn constraints, and female-specific feature sets.
5. Compared recall for men and women, and the **EO gap**: the difference between male and female recall. Lower is fairer.

## Results

Logistic Regression (LR) and Random Forest (RF) had the best baseline recall, about 0.89–0.91 overall. Split by sex, though, they found only 50% (LR) and 67% (RF) of the women with heart disease.

Female recall and EO gap, taken from the report:

| Approach | LR female recall | LR EO gap | RF female recall | RF EO gap |
|---|---|---|---|---|
| Baseline | 0.50 | 0.42 | 0.67 | 0.26 |
| Random oversampling | 0.67 | 0.23 | 0.67 | 0.26 |
| SMOTE (female data only) | 0.83 | 0.08 | 0.83 | 0.07 |
| Sample reweighting | 0.83 | 0.03 | 0.83 | 0.09 |
| Fairlearn: Demographic Parity | 0.83 | 0.01 | 1.00 | 0.07 |
| Fairlearn: Equalized Odds | 0.83 | 0.03 | 0.67 | 0.27 |
| AIF360-style reweighing | 0.67 | 0.13 | 0.83 | 0.09 |
| Pushing-4 features (Age, Cholesterol, FastingBS, ST_Slope) | 0.89 | 0.03 | 0.91 | 0.08 |

Main takeaways:
- Overall accuracy looked good, but it hid a big gap between men and women.
- Most methods raised female recall. Sample reweighting and SMOTE were the most consistent.
- A small, well-chosen feature set (Pushing-4) gave the best female recall with a small gap.

A few things to keep in mind:
- The test set has only 38 women, 6 of whom have heart disease, so one patient changes female recall by about 0.17.
- Fairlearn's predictions are randomized, so the saved notebook outputs for those rows differ slightly from the report.

All 15 plots from the notebook are in the [`images/`](images/) folder.

## Conclusion

The baseline models looked accurate overall but missed far more women than men with heart disease. This kind of bias only shows up when you evaluate each group separately. Most of the methods we tried narrowed the gap. Sample reweighting, SMOTE on female data, and a small, clinically chosen feature set worked best, and no single method was best for every model. The main limitation is the small number of women in the data (193 patients, only 6 positive cases in the test set). A larger and more balanced dataset would be needed before drawing strong conclusions.

## How to run

```bash
git clone https://github.com/mvskpranav11/CVD-Gender-Bias.git
cd CVD-Gender-Bias
pip install -r requirements.txt
jupyter notebook notebooks/CVD_Analysis_Final.ipynb
```

The notebook still includes its saved outputs, so you can also read it on GitHub without running anything.

## Repository structure

```
├── data/          heart999.csv (the dataset)
├── notebooks/     CVD_Analysis_Final.ipynb (full analysis)
├── report/        CVD_Report_Final.pdf (final report, 87 pages)
├── images/        plots from the notebook
├── requirements.txt
└── LICENSE
```

## Team

This was a group project by 11 students:
Akhil Daggu, Amrutha Korada, Karunasagar, Rakshitha Ramya Sadanand, Raju Yadav Narmala, Sarvani Reddy Yalam, Suprit Sanjeev Vaidya, Tejas Mahendra Chaudhari, Tejaswi Mudunuri, Venkata Sesha Krishna Pranav Manuguri (me) and Yashwanth Chenna Reddy Gotike.

Our supervisors were Prof. Dr. Andreas Mauthe and Christopher Latz, M.Sc. (Institut für Wirtschafts- und Verwaltungsinformatik, FG Mauthe, University of Koblenz). Thanks to both of them for their guidance.

The original team repo is at [research903/CVD](https://github.com/research903/CVD).

## License

The code is under the [MIT License](LICENSE). The dataset keeps its original Kaggle license (ODbL), and the report belongs to its authors.
