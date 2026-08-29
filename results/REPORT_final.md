# AIDA-Twin: Final Honest Metrics Summary

## System Metrics & Scalability
Scalability measured up to N=100 twins on Intel64 Family 6 Model 154 Stepping 4, GenuineIntel (12 Cores), Windows 11, 7.68 GB RAM hardware.
The 100-twin concurrency achieved a per-decision response time of 944.5 ms and required 208.2 MB of Resident Memory.

## Communication Payload
Communication: raw 247.789 B/step vs event-driven 59.614 B/step (75.94% reduction, event rate 0.847).

## Cross-Domain Transfer (LODO)
Despite utilizing principled domain alignment (Z-score standardization on independent train splits) and a unified padded representation dimension, zero-shot transfer remains extremely poor across all disjoint datasets. 

For example, on Scenario B (Gas Turbine Target):
LODO zero-shot AUROC = 0.465 (no transfer); few-shot (k=25) AUROC = 0.610.

*Self-Check Note: Target-test indices were asserted to never overlap with any train/fine-tuning sets. The lack of zero-shot transfer is a genuine characteristic of these highly heterogeneous datasets.*
