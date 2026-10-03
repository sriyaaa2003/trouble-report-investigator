# From public bug reports to Cloud RAN Layer 1 trouble reports

This repository is a **proxy study**. It does not use, and cannot see, any Ericsson data. It takes the
problem described in a published thesis posting ("AI-assisted investigation of Trouble Reports for Cloud RAN
Layer 1") and builds and measures the same pipeline on a public look-alike: Mozilla Core bug reports, a large
C++ platform codebase organised into subsystem "components", with human-labelled duplicates and recorded fixes.

Everything below about Cloud RAN L1 is an **assumption drawn from the posting's wording**, not knowledge of
how the real system or its data look. The point of this note is to be explicit about what transfers, what
would have to change, and what I would need to learn.

## What the posting asks for, and what exists here

| Capability in the posting | In this repository | Evidence it works |
|---|---|---|
| Retrieve similar historical reports and fixes | `investigate.py`: hybrid BM25 + dense retrieval, each case shown with resolution, duplicate link and the recorded fix note | Duplicate-retrieval benchmark, [EVALUATION.md](EVALUATION.md) |
| Identify the likely affected subsystem | `models.py`: TF-IDF classifier, embedding classifier and a similar-case vote, combined | Subsystem-prediction benchmark, [EVALUATION.md](EVALUATION.md) |
| Present supporting evidence | Matched terms, shared code symbols, neighbour component/resolution counts, recurring tags, links to concrete earlier reports | Unit-tested; judged qualitatively only |
| Recommend investigation steps | Deterministic steps derived from the neighbours (read the closest case, start in the top subsystem, inspect the change that fixed a similar case, look at shared symbols) | Not evaluated against engineers |
| Generate root-cause hypotheses | `hypothesis.py`: optional LLM step, strictly validated, every hypothesis must cite provided cases | **Not run against a live model; quality not measured** |
| Analyse logs, traces, simulation results | Out of scope here (text of the report only) | none |
| Knowledge graph | Out of scope here; the neighbour/symbol evidence is the lightweight stand-in | none |

## What transfers directly
- **The evaluation design.** Strict time split (train and index only on the past), duplicates and
  recorded component as ground truth, baselines first (BM25 is hard to beat), paired bootstrap intervals,
  hyper-parameters chosen on a development period and reported once on a held-out period.
- **Code-aware text handling.** The tokenizer keeps full identifiers and also emits their parts
  (`namespace::Class::Method` -> the whole symbol, each segment, camel-case pieces). Fault reports for native
  software are dominated by symbols, function names and error strings, so exact-symbol matching matters.
- **Explainability by construction.** A prediction is backed by concrete earlier reports, which is what an
  engineer needs in order to trust or reject it.
- **Honest degradation.** If the closest historical case is only loosely similar, the tool says so.

## What would have to change for real Layer 1 data
1. **Inputs.** A Trouble Report is not just a title and description. The posting also lists logs, traces,
   simulation results and software changes. These need their own parsing and features (error-code
   histograms, timing anomalies, build identifiers, commit ranges), then fusion with the text signal. My
   `netops` project (anomaly detection over severity-tagged logs) is the closest thing I have built to the
   log side; it is far simpler than what real traces would require.
2. **Labels.** "Component" here is the owner at resolution time. At Ericsson the analogous label might be a
   subsystem, a team, or a function; its noise level and granularity determine how well any model can do.
3. **Confidentiality.** TR data will be restricted. That means on-premise embedding and generation models
   (no external API), access control on retrieval, and an audit trail. I built a separate project for the
   access-control side ([tenant-rag](https://github.com/sriyaaa2003/tenant-rag)); the two are not integrated.
4. **Domain vocabulary.** Layer 1 text will be full of terms a general-purpose embedding model has rarely seen.
   Lexical retrieval is robust to that (a reason the hybrid matters); a domain-adapted embedding model or
   fine-tuning on historical TR pairs would be an obvious thesis question.
5. **Ground truth for hypotheses.** Duplicates and components have labels. Root-cause *hypotheses* do not;
   evaluating them needs expert review (for example blinded ratings of grounded vs ungrounded outputs, and
   how often an engineer's eventual root cause appears among the proposed hypotheses).
6. **Scale and history.** A decade of TRs differs from a four-year public sample. Duplicate density, drift in
   terminology, and re-organised subsystems would all need handling; the time-split protocol is meant to expose
   exactly those effects.

## What I do not know, and would need to learn
- **5G NR and Layer 1 internals.** I have no working knowledge of the radio stack. I would not claim any.
- **C++ at the level the posting asks for.** This repository is Python. The *data* is about C++ software, but
  that is not evidence of C++ proficiency.
- **How Ericsson triages today,** which fields exist, how reliable they are, and what an engineer would
  actually find useful. A thesis would start by interviewing those engineers and looking at real examples.

## Questions I would want answered first
1. How are TR duplicates and related cases recorded today, and how reliable is that link?
2. Which fields are consistently filled, and which are free text?
3. Is the affected-subsystem field set at filing time or at resolution?
4. What is the acceptable cost of a wrong suggestion, and who reviews it?
5. Can models run on-premise, and what hardware is available?
