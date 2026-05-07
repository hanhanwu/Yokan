# What Makes Me Unique

## Competitor Analysis
* Argilla - Golden Dataset Creation
  * Needs python coding to create, not user friendly to domain experts who can't code
* Labelbox - Polished UI to create the golden dataset
  * The user needs to manually create everything, not much intelligent automation
* BrainTrust - End-to-end evaluation
  * Might be the best end-to-end eval tool so far, but for product like RAG, especailly when the problem is in golden dataset go stale, it lacks of intelligent detection --> requires more human review
  * It also has too much items to monitor/check --> overcomplex the solution
  * Just like all the other eval tools, it records all the traces, but then what?


## What Makes RAG Doctor Unique
### Create Golden Dataset
* Friendly UI for non-coders
* Human select topics, knowledge base --> Auto generate Q&A --> AI review, correct when needed --> Human review, correct when needed --> AI review, flag problems --> Human review, correct when needed

### Auto Eval and A/B test
* More intelligent way to detect problematic golden dataset issues
* Auto eval fully customizable to be closely align with business metrics
* A/B test results can be shared with business stakeholders directly