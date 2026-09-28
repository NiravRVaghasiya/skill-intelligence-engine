---
name: data-privacy
display_name: Data Privacy & Anonymization
description: >
  Use when the user wants to protect personal data in a dataset or training
  pipeline. Trigger phrases: "remove PII from this dataset", "anonymize this
  data", "add differential privacy to training", "redact personal
  information", "how many people can I re-identify from this table",
  "k-anonymity for this table". NOT for fairness/bias metrics (see
  ai-ethics-fairness) or explaining model predictions (see explainability).
type: reference
domain: responsible-ai
level: intermediate
lifecycle: stable
risk_level: medium
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - k-anonymity
  - l-diversity
  - t-closeness
  - differential-privacy-budgeting
  - dp-sgd-training
  - quasi-identifier-detection
requires:
conflicts:
related:
  - ai-ethics-fairness
  - explainability
  - data-preprocessing
  - ai-ml-security
inputs: A dataset or training pipeline containing personal data, plus the intended release or deployment context.
outputs: An inventory of direct/quasi-identifiers and a recommended anonymization or differential-privacy technique matched to the threat model.
---

## Overview
Covers how personal data ends up identifiable even after "cleaning," and the
main technical tools for reducing that risk before or during modeling:
identifying PII and quasi-identifiers, classic anonymization guarantees
(k-anonymity and its extensions), and differential privacy for
statistics/ML training. The user walks away knowing which tool matches which
threat model, and why "we removed the names" is rarely sufficient.

## Key Concepts
- **Direct identifiers vs. quasi-identifiers.** *Direct identifiers* (name,
  email, SSN, phone) obviously identify someone and are easy to strip.
  *Quasi-identifiers* (zip code, birth date, gender, job title) don't
  identify anyone alone but **combine** to re-identify most of a population —
  Sweeney's classic result showed ~87% of the US population is uniquely
  identifiable from just {zip code, birth date, gender}. Any real privacy
  workflow must inventory quasi-identifiers, not just obvious PII columns.
- **De-identification vs. anonymization vs. pseudonymization.**
  *Pseudonymization* replaces identifiers with a reversible token/key — the
  mapping still exists somewhere, so under GDPR the data is still "personal
  data." *Anonymization* aims to be **irreversible**: no reasonable means
  exist to re-link records to individuals — this "reasonable means" bar is
  the EU/GDPR framing (Recital 26); other jurisdictions define the legal
  anonymization/de-identification line differently (e.g. US HIPAA's Safe
  Harbor and Expert Determination methods), so confirm which jurisdiction's
  specific standard actually applies before treating data as anonymized.
  *De-identification* (the NIST/US HIPAA term) sits between the two
  depending on the technique used — always check which guarantee a
  "de-identification" step actually provides.
- **k-anonymity, l-diversity, t-closeness.** *k-anonymity*: every combination
  of quasi-identifiers must be shared by at least \(k\) records (generalize
  zip codes, bucket ages, etc., until this holds). It doesn't protect against
  a *homogeneity attack* if all \(k\) records in a group share the same
  sensitive value (e.g. same diagnosis) — *l-diversity* fixes this by
  requiring \(\ell\) well-represented sensitive values per group.
  *t-closeness* further requires the sensitive-attribute distribution within
  each group to stay close to the overall distribution, defending against
  *skewness* and background-knowledge attacks.
- **Differential privacy (DP).** A mathematical guarantee that a
  statistic/model's output distribution changes by at most a bounded factor
  (controlled by privacy budget \(\varepsilon\)) whether or not any single
  individual's record is included. Achieved by adding calibrated noise
  (Laplace mechanism for bounded-sensitivity counts/sums, Gaussian mechanism
  for higher-dimensional queries). For ML training, **DP-SGD** clips
  per-example gradients and adds noise at each step so the trained model
  itself carries a DP guarantee, not just released statistics.
- **Privacy budget composition.** Each DP query/training run spends part of a
  finite \(\varepsilon\) budget; running many queries against the same
  dataset compounds the privacy loss (basic or advanced composition
  theorems), which is why production DP systems track and cap cumulative
  \(\varepsilon\) rather than treating each query independently.
- **Re-identification (linkage) attacks.** Combining an "anonymized" release
  with an external dataset that shares quasi-identifiers can re-identify
  individuals even when no direct identifier was ever released — the
  canonical cautionary examples are the AOL search-log release and the
  Netflix Prize dataset, both re-identified via linkage with public data.

## Gotchas
- **Pseudonymization is not anonymization.** Hashing or tokenizing a
  direct identifier is reversible if the mapping (or a rainbow table/salt-free
  hash of a small ID space) exists — regulators (GDPR) still treat
  pseudonymized data as personal data requiring the same protections.
- **Removing "obvious" PII columns misses quasi-identifiers.** Dropping name/
  SSN/email while leaving zip + birth date + gender in the clear still leaves
  most rows re-identifiable — always threat-model the quasi-identifier
  combination, not just a PII column checklist.
- **k-anonymity alone doesn't protect sensitive attributes.** Without
  l-diversity/t-closeness, a k-anonymous group can still leak a sensitive
  value via homogeneity (all k records share it) or background knowledge
  (attacker already knows the target isn't the one exception in the group).
- **DP noise erodes disproportionately for minority subgroups.** Adding
  uniform noise to satisfy a global \(\varepsilon\) hurts statistics computed
  on small subgroups far more (relative to their size) than on the majority
  group — DP and fairness goals can actively conflict and need to be
  evaluated together, not assumed compatible.
- **Privacy budget composes across queries/epochs.** Treating each DP query
  or each training epoch as "free" because \(\varepsilon\) looks small
  per-step ignores composition — cumulative \(\varepsilon\) across an entire
  pipeline or multi-epoch training run is what actually bounds the guarantee.

## References
- [NIST SP 800-188, De-Identification of Government Datasets](https://csrc.nist.gov/pubs/sp/800/188/final) — authoritative taxonomy of de-identification/re-identification risk and techniques.
- [Dwork & Roth, *The Algorithmic Foundations of Differential Privacy*](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) — the standard reference text for DP definitions, mechanisms, and composition theorems.
- [Google Differential Privacy library](https://github.com/google/differential-privacy) — open-source implementation of DP mechanisms (Laplace/Gaussian) for real aggregation queries.
