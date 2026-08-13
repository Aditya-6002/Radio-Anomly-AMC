\# Radio Anomaly + Automatic Modulation Classification



\## Overview



This project explores machine learning for radio-frequency (RF) signal analysis. The goal is to classify modulation types and investigate anomalous signal behavior under different signal-to-noise ratio (SNR) conditions.



\## Objectives



\* Perform preprocessing of RF signal data.

\* Explore signal characteristics across different SNR levels.

\* Build a model for Automatic Modulation Classification (AMC).

\* Investigate how signal quality affects classification performance.

\* Explore anomaly detection as a second stage of RF signal analysis.



\## Dataset



The dataset contains RF signal samples represented as I/Q data across multiple modulation classes and SNR levels.



The original dataset is stored locally and is \*\*not included in this repository\*\* because of its large size (\~38 GB).



\## Project Structure



```text

.

├── Radio\_Anomly\_AMC'.ipynb

├── README.md

├── .gitignore

└── Dataset/              # Local only — not uploaded to GitHub

```



\## Current Progress



\* \[x] Dataset exploration

\* \[x] Understanding class labels and SNR values

\* \[ ] Signal preprocessing

\* \[ ] Baseline AMC model

\* \[ ] Model evaluation

\* \[ ] RF anomaly detection

\* \[ ] Anomaly analysis across SNR levels



\## Technologies



\* Python

\* NumPy

\* Pandas

\* Matplotlib

\* Scikit-learn

\* Jupyter Notebook



\## Results



Results and experiments will be added as the project develops.



\## Future Work



\* Compare different ML/DL architectures.

\* Analyze performance across SNR levels.

\* Investigate misclassified signals.

\* Develop an anomaly detection pipeline.

\* Study whether learned signal representations can distinguish normal and anomalous RF behavior.



\## Author



Aditya Mishra



