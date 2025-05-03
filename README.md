# Diabetic-Foot-Ulcer-Prediction-Survival-Analysis


Diabetic Foot Ulcers (DFUs) are a severe complication of diabetes that can lead to
infection, amputation, and mortality if not identified and treated in time. The increasing prevalence
of diabetes worldwide highlights the need for early detection and personalized risk assessment to
mitigate these consequences. Existing models often fail to account for the complex interactions
between multiple high-impact factors such as Peripheral Neuropathy, Peripheral Arterial Disease
(PAD), and Glycemic Control, resulting in suboptimal prediction accuracy. Moreover, these
models lack survival analysis, which is crucial in estimating the time until the occurrence or
recurrence of foot ulcers. To address these challenges, this research proposes a Hybrid Diabetic
Foot Ulcer Prediction Model (HDFUPM) that combines the strengths of three machine learning
techniques: Random Forest (RF), Deep Neural Network (DNN), and Random Survival Forest
(RSF). The RF model is utilized for feature-based classification, while the DNN processes highdimensional data, capturing intricate patterns that linear models often miss. RSF is incorporated to
model time-to-event data and estimate survival probabilities, offering insights into patient-specific
risks over time. An adaptive fusion mechanism is implemented to intelligently combine the logit
outputs of these models and further refine the predictions through logistic regression. The fusion
mechanism dynamically assigns optimal weights to the models based on their individual
performance, ensuring robust and accurate predictions. The proposed system delivers not only a
binary classification of DFU risk but also provides survival probabilities for different time frames
(1 year, 3 years, and 5 years), offering clinicians a more comprehensive view of patient outcomes.
Extensive experimentation and evaluation on a dataset demonstrate the superior performance of
the HDFUPM model over traditional approaches. Key performance metrics such as accuracy, F1-
score, and Area Under the Receiver Operating Characteristic Curve (AUC-ROC) validate the
effectiveness of the proposed system. This hybrid approach significantly improves risk prediction
and survival analysis, contributing to timely interventions and better patient outcomes
