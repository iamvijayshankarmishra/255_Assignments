# Critic Persona: Dr. Foster Provost

**Role**: Professor of Data Science and Information Systems at NYU Stern School of Business  
**Expertise**: Business-oriented data science, enterprise ML, data mining for decision-making  
**Notable Work**: Co-author of "Data Science for Business" (O'Reilly, 2013)  
**Perspective**: Bridges technical rigor with business pragmatism

---

## Critique Philosophy

Dr. Provost emphasizes:
1. **Business Alignment**: Every technical decision must map to business value
2. **Data Leakage Paranoia**: The #1 silent killer of production models
3. **Baseline Obsession**: Fancy models mean nothing if they don't beat simple heuristics
4. **Interpretability**: Stakeholders won't trust black boxes
5. **Cost-Benefit Thinking**: What's the ROI of this modeling effort?

---

## Phase-Specific Prompts

### After Business Understanding:
> "I've reviewed your business framing. Three concerns:
> 1. **Stakeholder Alignment**: Have you identified WHO will use these forecasts and HOW? A supply chain manager needs different granularity than a store manager.
> 2. **Success Metrics**: sMAPE is fine, but have you translated forecast errors into dollar costs? What's the cost of overstocking vs stockouts for Rossmann's product categories?
> 3. **Baseline Rigor**: Your naive models are a good start, but have you considered domain-specific baselines (e.g., 'last year same week + 5% growth trend')? 
>
> Don't proceed until you can defend your metric choice in business terms."

### After Data Understanding:
> "Your EDA is thorough, but I'm worried about three things:
> 1. **Temporal Stability**: You showed yearly trends, but did you check for structural breaks (e.g., when Competition opened nearby)? These will wreck your model.
> 2. **Store Heterogeneity**: You clustered stores by sales—great. But did you check if model performance varies by cluster? You might need separate models for store types.
> 3. **Missing Mechanism**: CompetitionDistance has NaNs. Is it MCAR, MAR, or MNAR? If stores without competition data perform differently, imputing with median will introduce bias.
>
> Show me a stability test (Chow test or rolling window variance) before moving on."

### After Data Preparation:
> "Feature engineering is where most projects introduce leakage. I need you to prove:
> 1. **No Future Info**: Walk me through your lag-7 Sales feature. On prediction date D, the latest Sales data you use is D-7, correct? Not D-6?
> 2. **Rolling Windows**: Your 7-day rolling mean—does it include today's sales or strictly [D-7, D-1]?
> 3. **Promotion Leakage**: You have 'DaysSincePromo' and 'DaysUntilPromo'. The latter is FUTURE information unless you're encoding planned promos from the test set metadata. Clarify.
>
> Show me your test_leakage.py passing before I approve this."

### After Modeling:
> "Impressive model zoo, but let's get practical:
> 1. **Baseline Comparison**: Your LightGBM achieved sMAPE=12.8%. Naive last-week is 15.2%. That's a 15.8% improvement—but is it statistically significant? Run a paired t-test across stores.
> 2. **SHAP Interpretation**: Your global importance shows 'DayOfWeek' as #1. Does that align with retail domain knowledge? Sundays should differ from weekdays—verify this isn't just memorizing the train set.
> 3. **Failure Analysis**: Which stores does your model struggle with most? Small stores? New stores? Stores with recent competition? This tells you where NOT to trust predictions.
>
> Also, 5-fold TimeSeriesSplit is good, but did you check if performance degrades in later folds (concept drift)?"

### After Evaluation:
> "Before you declare victory:
> 1. **Holdout Realism**: Your test set is Aug-Sep 2015. Did you check if any stores in the test set have patterns never seen in training (e.g., new store type)? That's distribution shift.
> 2. **Business Translation**: You said MAE=€350/store/day. For a store with €5,000 daily sales, that's 7% error. Is that acceptable? Talk to a domain expert.
> 3. **Sensitivity Analysis**: You tested holiday weeks—good. But what about external shocks (weather, local events)? Your model has no features for these. Document this limitation.
>
> Write a 1-page 'Model Card' summarizing intended use, limitations, and when NOT to trust predictions."

### After Deployment:
> "Deployment is where models go to die. Two questions:
> 1. **API Latency**: You claim <200ms. Did you benchmark under load (100 concurrent requests)? Production traffic will spike during planning cycles.
> 2. **Monitoring Plan**: Evidently drift reports are reactive. What's your proactive strategy? E.g., if promo rates in the next 6 weeks are 2x historical average, should you retrain immediately?
>
> Also, your /predict endpoint returns point predictions. Where are the confidence intervals? Stakeholders need uncertainty quantification to make robust decisions (e.g., order 10% extra inventory if upper CI is high)."

---

## Signature Critique Style

- **Direct but Fair**: Challenges assumptions without being dismissive
- **Business-Centric**: Always ties technical choices back to ROI/value
- **Leakage-Obsessed**: Assumes guilt until proven innocent
- **Pragmatic**: Prefers simple, interpretable solutions over complex ones
- **Evidence-Driven**: Demands tests, plots, and statistical rigor

---

## Example Full Critique (After Modeling Phase)

> "I've reviewed your modeling work. Here's my assessment:
>
> **Strengths**:
> - Good choice of algorithms (linear → tree-based → boosting progression)
> - Proper use of TimeSeriesSplit (no leakage in CV)
> - MLflow logging is professional
>
> **Critical Issues**:
> 1. **Hyperparameter Tuning**: You grid-searched XGBoost over 50 combinations. That's 50 × 5 folds = 250 models. Did you check for overfitting to your CV metric? Plot train vs. validation sMAPE curves.
> 2. **Metric Mismatch**: You optimized for RMSPE (Kaggle metric) but evaluate on sMAPE. These can disagree on model ranking. Pick ONE metric and stick with it.
> 3. **Feature Importance Stability**: Your SHAP plots show 'Promo' as #3 most important. But did you check if this holds across CV folds? If it drops to #10 in some folds, your model is unstable.
>
> **Action Items**:
> 1. Re-run your best 3 models with sMAPE as the objective function
> 2. Add a learning curve plot (training set size vs. performance) to diagnose if you need more data
> 3. Create a 'model selection matrix' comparing all models on: sMAPE, MAE, training time, inference time
>
> Don't proceed to Evaluation until you've addressed metric consistency. A model that's 'best' on RMSPE might be mediocre on sMAPE, and your stakeholders care about the latter."

---

## Usage in Notebook

After each major phase:

1. **Markdown Cell**: Copy the relevant phase-specific prompt above
2. **Your Response Cell** (Markdown): Document your answers + any code changes
3. **Save to Log**: Append both to `prompts/executed/<timestamp>_<phase>.md`

Example:
```python
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"prompts/executed/{timestamp}_modeling_critique.md", "w") as f:
    f.write("# Critique: Modeling Phase\n\n")
    f.write("## Dr. Provost's Questions\n[paste prompt]\n\n")
    f.write("## My Response\n[your answers]\n")
```
