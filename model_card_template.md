# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a machine learning classifier trained to predict income levels based on various demographic and work-related features in census data. It was developed using the scikit-learn library and trained on a subset of the U.S. Census data. The model takes into account features such as workclass, education, marital-status, occupation, relationship, race, sex, and native-country to classify whether a person’s income is above or below $50K.

## Intended Use
The intended use of this model is for educational and analytical purposes. It can be used to analyze patterns in income based on demographics and understand which features contribute most to income classification. However, this model should not be used in production systems or for making real-world financial or hiring decisions, as it may contain biases inherent in the data.

Audience: Data scientists, analysts, and students interested in machine learning applications in demographic data.
Primary Uses:

Analyzing income patterns based on demographic data.
Understanding model metrics for income classification on census data.
## Training Data
The training data used for this model is derived from the U.S. Census dataset, which contains a wide range of demographic and work-related information. The dataset includes:

Total Records: Approximately 32,000 samples (after splitting into training and testing datasets).
Features: Workclass, education, marital-status, occupation, relationship, race, sex, native-country, and other demographic attributes.
Target: Binary classification of income (<=50K or >50K).
The data has undergone preprocessing steps including encoding of categorical features and label binarization. The process_data function was used to standardize these steps across training and testing datasets.

## Evaluation Data
he model was evaluated on a test set derived from the same census data, comprising 20% of the original data (around 6,400 samples). This test set is representative of the training set and contains similar distributions of demographic attributes.



## Metrics

The model’s performance was evaluated using Precision, Recall, and F1 score:

Precision: 0.7419
Recall: 0.6384
F1 Score: 0.6863
Additionally, the model was evaluated on different categorical slices to understand its performance across demographic subgroups. The slice_output.txt contains detailed metrics for each slice.

## Ethical Considerations
This model was trained on U.S. Census data, which may contain inherent biases related to gender, race, and socio-economic factors. These biases can lead to inaccurate or unfair predictions for certain demographic groups. For instance, certain income groups may be overrepresented or underrepresented based on factors such as race or gender, leading to skewed predictions.

Additionally, income is influenced by numerous contextual factors not present in this dataset, such as regional cost of living and job availability. Therefore, this model should not be used for high-stakes decision-making, such as hiring, lending, or policy-making.

## Caveats and Recommendations
Data Limitations: The dataset may not represent the current U.S. population as it is based on census data from a specific period. This limits its applicability in current real-world scenarios.
Biases in Categorical Slices: The model performed differently across demographic subgroups, as noted in slice_output.txt. It shows high variance in metrics across different slices, which suggests that the model may not generalize equally across all groups.
Further Improvements: This model could benefit from hyperparameter tuning, additional data cleaning, and possibly using an updated dataset to improve accuracy and fairness. Including more features related to socio-economic background could also provide a more nuanced view of income classification.
