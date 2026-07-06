"""Anton model layer: training, offline bid-optimizer evaluation, and serving.

This is STEP 3 of the pipeline. It consumes the leakage-safe **training tables**
produced by ``smarthub.feature_engineering`` (STEP 2) — it does NOT re-clean raw
leads — and trains ``P(won | bid, lead features)``, then uses that model to
recommend the profit-maximising bid.

Feature definitions (which columns, how they're derived and typed) live in
``smarthub.feature_engineering.features`` so training and serving stay in sync.
"""
