# Market Basket Analysis — Kaggle Groceries (Association Rules + Robust Visuals)

Discover which products are commonly purchased together using the Kaggle **Groceries** dataset.  
This notebook mines **association rules** using **Apriori** and **FP-Growth**, and guarantees meaningful insights with **co-occurrence/PMI** visuals when rules are sparse.

---

## What You’ll Learn
- How to turn receipts into **transactions** (baskets) and **items**
- How to mine **frequent itemsets** and **association rules** (support, confidence, lift)
- How to **adapt thresholds** and switch algorithms (Apriori → FP-Growth) to get useful rules
- How to visualize patterns: **bar chart**, **rule network**, **PMI heatmap**, **co-occurrence network**
- How to export results as tidy CSVs under `./artifacts/`

---

## Dataset
- **Kaggle**: `heeraldedhia/groceries-dataset`  
- Typical columns: `Member_number`, `Date`, `itemDescription`  
- We create a transaction ID: **`tid = Member_number + Date`**

---

## Environment
- **Python**: 3.8 recommended (works great with `mlxtend`; use 3.8 if you also want `pycaret==2.3.10`)
- Key packages:
  - `numpy==1.23.5`, `pandas==1.5.3`
  - `mlxtend==0.22.0` (Apriori, FP-Growth, association rules)
  - `matplotlib`, `networkx`
  - *(Optional)* `pycaret==2.3.10` if you want to experiment with its `arules` module (Python 3.8)

> Tip: If you see “command not found” or version conflicts, create a clean Python 3.8 virtual environment and install the packages inside it.

---

## How to Run (Notebook Cells)

**Step 0 — Sanity Check**  
Verifies Python and installs safe NumPy/Pandas pins. If PyCaret is missing, the notebook automatically falls back to `mlxtend`.

**Step 1 — Kaggle Download**  
Downloads `heeraldedhia/groceries-dataset` into `./data/groceries_assoc/`.  
Make sure your Kaggle API token is set (`~/.kaggle/kaggle.json` or env vars).

**Step 2 — Load & Prepare**  
- Load CSV, normalize columns (`Member_number`, `Date`, `itemDescription`)  
- Build `tid = Member_number + Date`  
- Drop missing rows  
- Print basket count, item count, and top-N items

**Step 3 — Itemsets & Rules (mlxtend)**  
- Convert baskets to one-hot table  
- Run **Apriori** (`min_support=0.01`)  
- Compute **association rules** (`min_confidence=0.25`)  
- Save:
  - `./artifacts/groceries_frequent_itemsets.csv`
  - `./artifacts/groceries_rules.csv`

**Step 4 (ALT) — Adaptive Mining**  
If you got **0 rules**, this cell:
- Gradually lowers `min_support` and `min_confidence`
- Switches to **FP-Growth** if Apriori still returns nothing
- Stops once it finds a reasonable number of rules  
Saves: `./artifacts/groceries_rules_adaptive.csv`

**Step 5 (ALT) — Visuals That Always Work**  
- If rules exist:  
  - Bar chart: top consequents by **mean(lift × confidence)**  
  - Rule network: top rules by **confidence × lift × support**  
- If not:  
  - **PMI heatmap** (co-occurrence strength)  
  - **Co-occurrence network** (top PMI edges)

---

## Interpreting Metrics
- **Support**: fraction of baskets containing the itemset  
- **Confidence**: P(consequent | antecedent) — “If A is in the basket, how often do we also see C?”  
- **Lift**: how many times more often A and C occur together vs. by chance (>1.0 is usually interesting)

---

## Tuning Tips
- Too few rules? Lower:
  - `min_support` (e.g., 0.01 → 0.005 → 0.002)  
  - `min_confidence` (e.g., 0.30 → 0.20)  
- Too many rules? Raise those thresholds or restrict `max_len` (itemset length).  
- For small/long-tail data, **FP-Growth** is often faster and more forgiving.

---

## Outputs
- `./artifacts/groceries_frequent_itemsets.csv`  
- `./artifacts/groceries_rules.csv`  
- `./artifacts/groceries_rules_adaptive.csv` *(adaptive mining)*  
- `./artifacts/groceries_rules_summary.csv` *(pretty/filtered summary, if you used the non-adaptive Step 4)*

---

## Why the “Adaptive” Path?
Market-basket datasets can be sparse. A single strict threshold can easily yield **zero** rules.  
Our adaptive pass **relaxes** thresholds and tries **FP-Growth**, so you **always** get either:
- solid association rules, **or**
- useful **co-occurrence/PMI** insights you can present.

---

## Ideas to Extend
- Segment by **weekday/weekend** or **time of day**  
- Group items into **categories** (e.g., dairy, bakery) and mine rules at the category level  
- Compare stores or time periods (pre/post promotion)  
- Build a simple **recommender**: given items in a basket, suggest high-lift consequents

---

## License
- Code: MIT (adjust as needed)  
- Data: per Kaggle dataset terms for `heeraldedhia/groceries-dataset`


### Here is the explanation vide : https://drive.google.com/drive/folders/1rxDyHGmSOPSVs0LSOhMbVTd93iJ_7xc7
