# Unattributed reference data

`X.npy` and `Y.npy`, 12 MB each, moved here from `simul/test/NB_t_test_data/`
during the 2026-09-10 restructure.

**What is known.** Nothing in the repository reads them: a grep for `.npy`
across every tracked file returns zero hits. They are the only tracked files
with no reader and no writer.

**What is guessed, and should not be relied on.** Their per-gene means are
around 100, which is consistent with `main_rSEQ.tex`'s "Densely Expressed Gene
Data" section (`\label{sec:dense_nb}`): mu_Y = 100, phi_Y = 0.1, mu_X = 100 + h
with h ~ N(15, 25), phi_X varying over {0.1, 0.2, 1.0}. That section produces
`figures/NB_experiments/dense_nb_lfcs.png`. The match is by parameter plausibility
only -- no code links the arrays to that figure, and no commit message does either.

**Why they are here rather than deleted.** A file is deleted only if we are certain it did not contribute to a manuscript
result, or if it is duplicated code. Neither holds: the parameter match is
evidence *for* contribution, not against, and there is no duplicate.

**Open question.** What produced these, and do they underlie `dense_nb_lfcs.png`?
If yes they are an input worth a generator; if no they are 24 MB of dead weight.
