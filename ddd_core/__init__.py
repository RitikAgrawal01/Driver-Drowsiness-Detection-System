"""Shared drowsiness-feature code used by the training pipeline, the backend and the tests.

Training and serving import the same functions, so features cannot silently differ
between the two (the v1 training/serving skew).
"""
