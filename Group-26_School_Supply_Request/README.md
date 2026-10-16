# Group 26 — School Supply Request Classification

A predictive classification system that tells a school supply office whether a request should be **Approve**, **Review**, or **Defer**, based on four inputs: quantity, urgency, current stock, and purpose.

| Item | Value |
|---|---|
| Primary algorithm | Random Forest Classifier |
| Comparison algorithm | Decision Tree Classifier |
| Inputs | Quantity, urgency (Low/Medium/High), current stock, purpose |
| Output | Approve, Review, Defer |
| Tools | KNIME Analytics Platform 5.12, Python (pandas, scikit-learn), Streamlit, Google Colab |

## Files

| File | Description |
|---|---|
| `school_supply_requests_clean.csv` | 90 simulated supply requests (cleaned) |
| `Group26_SupplyRequest_Classification.knwf` | KNIME workflow with both algorithms (import via File → Import KNIME Workflow) |
| `Group26_School_Supply_Request_Classification.ipynb` | Full Python process: data preparation, data dictionary, training, comparison, Streamlit launcher, testing, interpretation |
| `app.py` | Streamlit interface (standalone copy) |
| `screenshots/` | KNIME results and Streamlit test cases |

## Dataset

90 simulated requests labeled with rules that mimic a supply officer's decisions, with about 8% of labels changed at random to imitate inconsistent decisions. Class balance: 39 Approve, 29 Review, 22 Defer. Request ID, requesting office and date are kept for reference only and excluded from the model.

## KNIME workflow

CSV Reader → Column Filter → Missing Value → One to Many → Table Partitioner (75/25, stratified, seed 42) → Decision Tree and Random Forest Learner/Predictor → Scorer.

![KNIME workflow](screenshots/01_knime_workflow.jpg)

## Results

| Test | Random Forest | Decision Tree |
|---|---|---|
| KNIME, 23 test records: accuracy | **73.9%** | 69.6% |
| KNIME: Cohen's kappa | 0.584 | 0.522 |
| Python, 5-fold cross-validation: accuracy | **81.1%** | 74.4% |
| Python, 5-fold cross-validation: macro F1 | **0.80** | 0.70 |

Random Forest performed better in both tools and was selected. Quantity and current stock are the most important features (about 35% and 30% of feature importance). The weakest class is Defer (recall 0.33 in KNIME).

## Streamlit interface

The app accepts a single request or a CSV upload, validates inputs (quantity 1–150, required urgency and purpose), and shows the predicted category, a short reason, and the model's confidence. Two safety rules apply: confidence below 55% triggers a manual-review warning, and a request whose stock cannot cover the quantity is never auto-approved.

| Test type | Input | Result |
|---|---|---|
| Normal | 30 units, stock 200, High, Examination | Approve (93%) |
| Normal | 10 units, stock 150, Low, Personal | Defer (70%) |
| Uncertain | 60 units, stock 55, High, Personal | Review + low-confidence warning |
| Boundary | 1 unit, stock 0, High, Examination | Review (safety rule) |
| Boundary | 200 units | Rejected: quantity must be 1–150 |
| Missing | Urgency not selected | Rejected: urgency required |

![Streamlit approve](screenshots/06_streamlit_normal_approve.jpg)

## How to run

1. Open the notebook in Google Colab and click **Runtime → Run all**.
2. Scroll to the last cell and open **Link 1 (Google)** to use the Streamlit app.

To run the app locally instead: put `app.py` and `supply_request_model.joblib` (created by the notebook) in one folder, then run `pip install streamlit scikit-learn pandas joblib` and `streamlit run app.py`.

## Limitations

The data is simulated, so results show how well the models learn assumed officer rules, not real-world performance. The model is weak at catching Defer cases, and factors like item cost, budget and supplier lead time are not included. The system is a decision aid; the supply officer keeps the final say.
