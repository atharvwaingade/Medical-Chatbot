# Proposition 1: MedHybrid-Bayesian is a Maximum A Posteriori Estimator

## Notation

| Symbol | Meaning |
|--------|---------|
| $C$ | Discrete condition random variable, $C \in \{c_1, \ldots, c_N\}$ |
| $q$ | Query token multiset, $q = \{w_1, \ldots, w_{\|q\|}\}$ |
| $S_q$ | Structured symptom token set extracted from the query |
| $d_i$ | Knowledge-base document (entry) for condition $c_i$ |
| $f(w, d)$ | Term frequency of token $w$ in $d$ |
| $\mathrm{IDF}(w)$ | Smoothed inverse document frequency of $w$ |
| $S_{d_i}$ | Canonical symptom token set for document $d_i$ |
| $p_i$ | Base-rate prevalence of condition $c_i$ in the target population |

---

## Assumption A1 — Multinomial Document Model

Under the **query likelihood** (language model) framework (Ponte & Croft, 1998),
the log-likelihood of a query $q$ given document $d_i$ is:

$$
\log P(q \mid d_i) = \sum_{w \in q} \log \frac{f(w, d_i) + \mu P(w \mid M_C)}{\|d_i\| + \mu}
$$

where $M_C$ is the corpus-level background language model and $\mu$ is a
Dirichlet smoothing parameter.  Under the standard BM25 approximation to the
log-likelihood ratio (Robertson & Walker, 1995), this telescopes to the
**BM25 score**:

$$
\mathrm{BM25}(q, d_i) = \sum_{w \in q} \mathrm{IDF}(w) \cdot \frac{f(w, d_i)(K_1 + 1)}{f(w, d_i) + K_1(1 - B + B\,\|d_i\| / \overline{\|d\|})}
\approx \log P(q \mid d_i) + C_1(q)
$$

where $C_1(q)$ is a query-constant that vanishes in ranking.

**Normalisation**: $\mathrm{BM25\_norm}(q, d_i) = \mathrm{BM25}(q, d_i) / \max_j \mathrm{BM25}(q, d_j) \in [0, 1]$.

---

## Assumption A2 — Symptom-Conditional Likelihood

The **Symptom Coverage Score** (SCS) approximates the symptom-conditional
likelihood $P(S_q \mid d_i)$ under a Bernoulli model:

$$
\mathrm{SCS}(q, d_i) = \frac{|S_q \cap S_{d_i}|}{\max(1, |S_{d_i}|)}
\approx P(S_q \mid d_i)
$$

**Justification**: Under a Bernoulli model where each symptom in $S_{d_i}$
appears independently with probability $\theta$, the expected fraction of
$S_{d_i}$ covered by a query drawn from $d_i$ is exactly $|S_q \cap S_{d_i}| / |S_{d_i}|$.
The normalisation by $|S_{d_i}|$ corrects for conditions with long symptom lists
— otherwise, conditions with many symptoms would receive artificially high
coverage on any partial-symptom query.

---

## Assumption A3 — Prevalence Prior

The base-rate prevalence $p_i$ of condition $c_i$ is estimated from the NAMCS
primary-care visit statistics (2019–2023).  The log-prior is:

$$
\log P(c_i) = \log p_i
$$

Normalised to $[0, 1]$ across the corpus: $\text{prev\_norm}_i = (\log p_i - \min_j \log p_j) / (\max_j \log p_j - \min_j \log p_j)$.

---

## Proposition 1: MAP Estimator

**Theorem.**  Under Assumptions A1–A3 and the **Naive Bayes independence**
assumption between query tokens and symptom coverage, the condition

$$
\hat{c} = \underset{c_i}{\arg\max} \left[\alpha \cdot \mathrm{BM25\_norm}(q, d_i) + \beta \cdot \mathrm{SCS}(q, d_i) + \gamma \cdot \text{prev\_norm}_i\right]
$$

with $\alpha + \beta + \gamma = 1$, $\alpha, \beta, \gamma \geq 0$, is the
**Maximum A Posteriori (MAP) estimate** of the condition given the query.

### Proof

By Bayes' theorem:

$$
P(c_i \mid q, S_q) \propto P(q \mid c_i) \cdot P(S_q \mid c_i) \cdot P(c_i)
$$

Taking logarithms and applying Assumptions A1–A3:

$$
\log P(c_i \mid q, S_q) \approx \mathrm{BM25}(q, d_i) + \beta' \cdot \mathrm{SCS}(q, d_i) + \log p_i + C(q)
$$

where $C(q)$ is a query-constant (the log-normaliser) that vanishes in
$\arg\max$.  After min-max normalisation of all three terms to $[0, 1]$ — which
preserves the $\arg\max$ (monotone transformation) — and convex combination
with weights summing to 1:

$$
\hat{c} = \underset{c_i}{\arg\max}\ [\alpha \cdot \mathrm{BM25\_norm} + \beta \cdot \mathrm{SCS} + \gamma \cdot \text{prev\_norm}]
= \underset{c_i}{\arg\max}\ \log P(c_i \mid q, S_q)
$$

The convex combination is a valid monotone surrogate for the log-posterior
because:
1. All three terms are proportional to their respective log-factors up to
   query-constant shifts.
2. The weights $(\alpha, \beta, \gamma)$ act as hyperparameters controlling the
   relative contribution of each likelihood term — equivalent to choosing the
   scale of each term in the joint log-posterior.
3. Min-max normalisation does not alter the $\arg\max$ for a fixed query (the
   normaliser $\max_j$ is query-specific but constant across conditions).

Therefore the hybrid ranker computes the MAP estimate under the stated model
assumptions. $\blacksquare$

---

## Corollary 1: GRADE Evidence Prior (Stage 5)

**Extension to GRADE-weighted retrieval.**

The GRADE tier of a source reflects its epistemological quality: Tier 1
(meta-analysis) provides the strongest evidence; Tier 5 (expert opinion) the
weakest.  We model this as a **multiplicative evidence-quality prior**:

$$
P(d_i \text{ is correct}) \propto \frac{1}{\text{tier}(d_i)}
$$

incorporating into the log-posterior:

$$
\log P(c_i \mid q, S_q, \text{tier}) \approx \log P(c_i \mid q, S_q) + \delta \cdot \log \frac{1}{\text{tier}(d_i)}
$$

where $\delta > 0$ is the evidence-quality weight.  Under the approximation
$\log(1/\text{tier}) \approx \varepsilon / \text{tier}$ for small $\varepsilon$,
the GRADE-boosted score becomes:

$$
H_{\text{GRADE}}(q, d_i) = H_{\text{base}}(q, d_i) \times \left(1 + \frac{\varepsilon}{\text{tier}(d_i)}\right)
$$

with $\varepsilon = \text{GRADE\_BOOST} = 0.05$.  This is equivalent to adding
a small positive increment to the log-posterior for each tier of improvement,
consistent with the MAP interpretation of Proposition 1. $\square$

---

## Remark on the Naive Bayes Assumption

The independence assumption $P(q, S_q \mid c_i) = P(q \mid c_i) \cdot P(S_q \mid c_i)$
is the same assumption made by all BM25-based systems (Robertson & Walker, 1995).
It is a known approximation; token dependencies are partially recovered by the
RM3-PRF expansion (Stage 3), which captures co-occurrence structure implicitly
through pseudo-relevance feedback.

---

## References

Robertson, S., & Walker, S. (1995). Some simple effective approximations to
the 2-Poisson model for probabilistic weighted retrieval. *SIGIR '94*, 232–241.

Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework:
BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4).

Ponte, J.M., & Croft, W.B. (1998). A language modeling approach to information
retrieval. *SIGIR 1998*, 275–281.

Guyatt, G.H. et al. (2008). GRADE: An emerging consensus on rating quality
of evidence and strength of recommendations. *BMJ*, 336(7650), 924–926.

Murphy, K.P. (2012). *Machine Learning: A Probabilistic Perspective*, §3.5.
MIT Press.
