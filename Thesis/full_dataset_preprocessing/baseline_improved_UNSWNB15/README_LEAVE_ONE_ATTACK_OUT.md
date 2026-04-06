# Leave-One-Attack-Type-Out Evaluation on UNSW-NB15 dataset

UNSW-NB15 does not provide explicit scenario-level splits like IoT-23, so a standard Leave-One-Scenario-Out (LOSO) evaluation is not directly applicable.

Instead, I will use this a **Leave-One-Attack-Type-Out** protocol. For each fold, one attack category from `attack_cat` is held out for testing, while the model is trained on all remaining attack categories together with benign traffic. The test set therefore contains:
- all benign samples
- only the held-out attack category

This setup evaluates how well a model generalizes to an unseen attack type and serves as an UNSW-NB15 analogue to out-of-distribution evaluation. I think it is the closest I can simulate a "LOSO" on this dataset!

