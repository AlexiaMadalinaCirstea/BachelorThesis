# Leave-One-Attack-Type-Out Evaluation on UNSW-NB15 dataset

UNSW-NB15 does not provide explicit scenario-level splits like IoT-23, so a standard Leave-One-Scenario-Out (LOSO) evaluation is not directly applicable.

Instead, I will use this a **Leave-One-Attack-Type-Out** protocol. For each fold, one attack category from `attack_cat` is held out for testing, while the model is trained on all remaining attack categories together with benign traffic. The test set therefore contains:
- all benign samples
- only the held-out attack category

This setup evaluates how well a model generalizes to an unseen attack type and serves as an UNSW-NB15 analogue to out-of-distribution evaluation. I think it is the closest I can simulate a "LOSO" on this dataset!

Because each held-out attack type defines one evaluation fold, the same fold structure can also be reused for:
- feature stability analysis across held-out attack-type folds
- transfer-utility analysis for features that remain important when the attack family changes

Conceptually this is different from IoT-23 scenario-level stability but it still measures whether a feature's contribution is robust under distribution shift in UNSW-NB15.

