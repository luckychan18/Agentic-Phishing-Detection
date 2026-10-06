\# An Intelligent Agentic Framework for Explainable Phishing Detection



An intelligent agentic framework for phishing URL detection that combines machine learning, rule-based analysis, dynamic evidence selection, conflict detection, and explainable AI (XAI) to produce an interpretable final decision.



\## 📌 Overview



Traditional phishing detection systems generally follow a static pipeline:



```text

Input URL → Machine Learning Model → Prediction → Explanation

Although such systems can achieve high classification performance, they typically do not dynamically decide what additional evidence should be investigated when different evidence sources disagree.

This project implements an agentic decision-making layer that can:

\- Select relevant evidence sources

\- Combine machine learning and rule-based evidence

\- Detect conflicting evidence

\- Request additional XAI evidence when required

\- Use SHAP explanations as decision evidence

\- Fuse multiple evidence sources

\- Produce a final decision with human-readable reasoning

The goal is to move from a fixed prediction pipeline toward an adaptive evidence-driven phishing detection framework.

🎯 Research Gap Addressed

The underlying research work identifies limitations in existing phishing detection and explainable AI systems, particularly the lack of autonomous and dynamic reasoning.

This project addresses the following gaps:

Research Gap	Proposed Implementation

Static detection workflow	Agent-based decision controller

No autonomous evidence selection	Dynamic evidence selection

Limited model comparison	Random Forest and XGBoost comparison

No explicit uncertainty handling	Evidence sufficiency and conflict checks

No additional investigation	Additional XAI analysis

XAI treated only as an explanation	SHAP output used as evidence

Conflicting evidence not resolved	Evidence conflict detection and fusion

Limited interpretable reasoning	Human-readable final reasoning





🧠 Proposed Architecture

&#x20;                        USER INPUT

&#x20;                            │

&#x20;                            ▼

&#x20;                   ┌─────────────────┐

&#x20;                   │  Agent Controller│

&#x20;                   └────────┬────────┘

&#x20;                            │

&#x20;             ┌──────────────┼──────────────┐

&#x20;             ▼              ▼              ▼

&#x20;      URL Structural   ML Evidence    Rule-Based

&#x20;         Evidence       Evidence       Evidence

&#x20;             │              │              │

&#x20;             └──────────────┼──────────────┘

&#x20;                            ▼

&#x20;                   ┌─────────────────┐

&#x20;                   │ Evidence Check  │

&#x20;                   └────────┬────────┘

&#x20;                            │

&#x20;                     Evidence Conflict?

&#x20;                       /            \\

&#x20;                     YES             NO

&#x20;                      │               │

&#x20;                      ▼               ▼

&#x20;               Additional XAI      Continue

&#x20;                 Investigation

&#x20;                      │

&#x20;                      ▼

&#x20;                    SHAP

&#x20;                      │

&#x20;                      ▼

&#x20;             ┌────────────────────┐

&#x20;             │ Evidence Fusion   │

&#x20;             └─────────┬──────────┘

&#x20;                       │

&#x20;                       ▼

&#x20;            ┌─────────────────────┐

&#x20;            │ Final Agent Decision│

&#x20;            └─────────┬───────────┘

&#x20;                      │

&#x20;             ┌────────┼─────────┐

&#x20;             ▼        ▼         ▼

&#x20;         PHISHING  SUSPICIOUS  LEGITIMATE

&#x20;                      │

&#x20;                      ▼

&#x20;            Human-Readable Explanation



🔍 Evidence Sources

The framework currently combines multiple evidence sources.

1\. URL Structural Evidence

The system extracts URL-level characteristics such as:

\- URL length

\- Domain length

\- IP address usage

\- URL similarity

\- Subdomain count

\- Obfuscation indicators

\- Special-character statistics

\- Digit statistics

\- Query and parameter characteristics

2\. Machine Learning Evidence

The project uses machine learning models trained on the PhiUSIIL Phishing URL (Website) Dataset.

Implemented models include:

\- Random Forest

\- XGBoost

The agent can compare model outputs and use their predictions as one source of evidence.

3\. Rule-Based Evidence

Independent URL heuristics are used to detect suspicious characteristics such as:

\- HTTP instead of HTTPS

\- Security/login-related keywords

\- Suspicious hyphenated domain structures

\- Excessive subdomains

\- IP-based URLs

\- Punycode

\- Suspicious query parameters

\- Credential-related URL patterns

The rule-based layer provides an additional source of evidence rather than replacing the ML model.

4\. Explainable AI Evidence

SHAP (SHapley Additive exPlanations) is used to explain individual model predictions.

The framework interprets SHAP contributions as evidence supporting either:

\- Phishing

\- Legitimate

This allows the agent to use XAI output during evidence fusion rather than treating explainability as a purely post-processing step.

🤖 Agentic Decision Process

The central component of the project is the agent controller.

The agent follows an adaptive process:

1\. Receive URL

&#x20;      ↓

2\. Select initial evidence

&#x20;      ↓

3\. Obtain ML prediction

&#x20;      ↓

4\. Obtain rule-based evidence

&#x20;      ↓

5\. Check evidence consistency

&#x20;      ↓

6\. If evidence is insufficient/conflicting

&#x20;      ↓

7\. Request additional SHAP/XAI evidence

&#x20;      ↓

8\. Compare evidence sources

&#x20;      ↓

9\. Fuse evidence

&#x20;      ↓

10\. Generate final decision

&#x20;      ↓

11\. Generate human-readable reasoning



This allows the system to investigate a URL further instead of relying blindly on a single prediction.

🔬 Explainable AI with SHAP

SHAP is used to identify which features contributed most strongly to an individual prediction.

For each analyzed URL, the system can display:

\- Feature name

\- Feature value

\- SHAP contribution

\- Direction of contribution

Example:

Feature                  SHAP       Direction

\------------------------------------------------

URLSimilarityIndex       -0.1888    Legitimate

LetterRatioInURL         -0.1123    Legitimate

URLLength                +0.0093    Phishing

NoOfOtherSpecialChars    +0.0391    Phishing



Positive SHAP contributions indicate evidence supporting the phishing class, while negative contributions indicate evidence supporting the legitimate class for the current explanation model.

⚖️ Evidence Conflict Resolution

A key part of the framework is handling disagreement between evidence sources.

For example:

ML prediction       → LEGITIMATE

Rule analysis       → SUSPICIOUS

SHAP evidence       → LEGITIMATE

Raw URL analysis    → SUSPICIOUS



Instead of immediately accepting the ML prediction, the agent identifies the conflict and performs additional analysis.

The evidence is then combined to produce a final decision such as:

SUSPICIOUS



with an explanation of why the evidence sources disagreed.

📊 Dataset

The project uses the:

PhiUSIIL Phishing URL (Website) Dataset

Source:

https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset

Dataset characteristics:

\- 235,795 instances

\- 54 original features

\- No missing values

\- URL and webpage-derived features

\- Binary classification

\- Label 1 = legitimate

\- Label 0 = phishing

The dataset contains pre-extracted URL/webpage-derived features. Therefore, this prototype focuses on adaptive evidence orchestration over these available features rather than live website crawling or real-time HTML/DNS retrieval.

🛠️ Technologies Used

\- Python

\- Streamlit

\- Scikit-learn

\- XGBoost

\- SHAP

\- Pandas

\- NumPy

\- Matplotlib

\- Joblib

📁 Project Structure

Agentic-Phishing-Detection/

│

├── src/

│   ├── \_\_init\_\_.py

│   ├── agent.py

│   ├── agent\_second.py

│   ├── url\_analyzer.py

│   ├── url\_predict.py

│   └── xai.py

│

├── app.py

├── app\_second.py

│

├── agent.py

├── url\_predict.py

│

├── train.py

├── train\_random\_url\_model.py

├── train\_xgb\_second.py

│

├── evidence\_models.py

├── ablation.py

├── calibrate.py

├── calibrate\_second.py

├── dataset\_check.py

├── diagnose.py

├── find\_legitimate.py

├── leakage\_check.py

├── python\_legitimate.py

├── second\_dataset\_test.py

├── shap\_analysis.py

├── shap\_predict.py

├── shap\_second.py

│

├── .gitignore

├── requirements.txt

└── README.md



🚀 Installation

1\. Clone the repository

git clone https://github.com/luckychan18/Agentic-Phishing-Detection.git

cd Agentic-Phishing-Detection



2\. Create a Python environment

Using Conda:

conda create -n phishingml python=3.11

conda activate phishingml



3\. Install dependencies

pip install -r requirements.txt



📦 Dataset Setup

The PhiUSIIL dataset is not included in this repository.

Download the dataset from the official UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset

Place the required dataset file inside:

data/



The dataset and trained model files are excluded from Git using .gitignore.

▶️ Running the Application

After installing the dependencies and preparing the dataset/model files:

streamlit run app.py



The application provides an interactive interface for analyzing URLs and displaying:

\- Agent controls

\- ML predictions

\- Evidence sources

\- Rule-based analysis

\- Model comparison

\- SHAP explanations

\- Feature-level evidence

\- Conflict resolution

\- Final agent decision

\- Human-readable reasoning

🧪 Example Analysis

Example legitimate URL:

https://www.google.com



Example suspicious URL:

http://login-securityexample.com/login?user=123



The second example demonstrates why relying solely on an ML probability can be insufficient. The agent can identify suspicious structural and rule-based evidence even when the learned model output is not strongly phishing-oriented.

📈 Model Performance

The underlying PhiUSIIL feature-based models achieve very high performance on the evaluated dataset.

The project also evaluates model behavior using:

\- Accuracy

\- Precision

\- Recall

\- F1-score

\- ROC-AUC

\- Confusion matrix

\- Domain-disjoint evaluation

The emphasis of this project, however, is not simply maximizing classification accuracy.

The primary contribution is the agentic evidence orchestration layer that determines when additional evidence is required and how conflicting evidence is handled.

🔐 Important Limitations

This prototype has several limitations:

1\. It does not perform live website crawling.

2\. It does not perform live DNS or WHOIS lookups.

3\. It does not retrieve real-time threat intelligence.

4\. It does not inspect live HTML or screenshots during inference.

5\. The dataset contains pre-extracted URL/webpage-derived features.

6\. Very high performance on the dataset should not be interpreted as guaranteed real-world phishing detection performance.

7\. The rule-based evidence layer is heuristic and should be further validated on external datasets.

Future versions can extend the evidence space with live threat intelligence, HTML/DOM analysis, screenshots, retrieval-augmented evidence, and LLM-based evidence synthesis.

🔮 Future Work

Potential extensions include:

\- Live URL and domain reputation analysis

\- External threat-intelligence integration

\- HTML and DOM analysis

\- Screenshot/visual phishing analysis

\- Retrieval-Augmented Generation (RAG)

\- LLM-based evidence synthesis

\- Multi-agent evidence analysis

\- Online/adaptive learning

\- Improved uncertainty estimation

\- External dataset validation

\- Real-world deployment evaluation

📚 Dataset Reference

Prasad, A., \& Chandra, S. (2024).

PhiUSIIL: A diverse and comprehensive phishing URL dataset.

Computers \& Security.

DOI:

https://doi.org/10.1016/j.cose.2023.103545

Dataset:

https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset

👩‍💻 Project

Agentic Phishing Detection

An academic/research prototype exploring the integration of machine learning, explainable AI, and agentic evidence-driven decision making for phishing URL detection.

⭐ Acknowledgement

This project uses the PhiUSIIL Phishing URL (Website) Dataset provided through the UCI Machine Learning Repository.

