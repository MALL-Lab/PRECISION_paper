# Reference values of tests/test_zph_exact.py: cox.zph(transform = "rank", terms = FALSE)
# of R survival on the synthetic data set of that test (no patient data). Frozen with
# R 4.5.2 (2025-10-31 ucrt) and survival 3.8.3. Rerun from the project root with
#   Rscript tests/zph_exact_reference.R
# and copy the printed coef, chisq, p and chisq_diag into R_REF of the test if survival
# changes.
#
# Four models, one per branch of zph_exact in scripts/64_proportional_hazards.py:
#   plain         x1 + x2, unpenalized, no strata
#   strata        x1 + x2 + strata(s), unpenalized
#   ridge         ridge(x1, x2, theta = 0.1 * n, scale = TRUE), the lifelines penalizer 0.1
#                 on every covariate, as in the canonical model of script 64
#   ridge_strata  ridge(x1, theta = 0.1 * n, scale = TRUE) + strata(s), one penalized
#                 coefficient and a stratum, as in the stratified penalized model of 64
# For the two penalized models the script also prints chisq_diag. It is the statistic of a
# copy of cox.zph in which the ridge second derivative goes only on the diagonal of the beta
# block, the convention of zph_exact. cox.zph itself assigns the vector fit$coxlist2$second
# (the diagonal of the ridge second derivative) to the nvar x nvar beta block, which R fills
# by recycling the vector column by column, so row i gets second[i] in every column. It also
# adds second to the gamma block. The copy changes those two lines and nothing else, so the
# score and information still come from the C code of survival (Czph1).
suppressPackageStartupMessages(library(survival))
d <- data.frame(
  time = c(7, 2, 25, 6, 2, 4, 6, 1, 8, 1, 25, 25, 25, 10, 8, 7, 25, 1, 4, 23,
           17, 12, 4, 12, 16, 25, 4, 10, 25, 1, 25, 1, 3, 8, 3, 12, 3, 10, 1, 25),
  status = c(0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
             1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1),
  x1 = c(0.5, 1.94, -0.93, 0.55, 1.09, -0.03, 0.44, 0.36, 0.3, 0.85,
         -1.03, -1.92, -0.49, -0.28, 1.56, 0.21, -0.13, 1.87, -0.66, 1.4,
         0.2, 1.53, -0.16, -1.43, 0.46, 0.57, 0.55, -0.48, 0.03, -0.39,
         -0.06, 1.18, -0.01, -0.61, 0.28, -0.64, -0.1, -0.51, 0.37, -1.38),
  x2 = c(1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1)
)
d$s <- rep(1:3, length.out = nrow(d))
n <- nrow(d)
ctl <- coxph.control(eps = 1e-12, toler.chol = 1e-14, iter.max = 100)

# cox.zph with the ridge second derivative on the diagonal of the beta block only
src <- deparse(body(cox.zph), width.cutoff = 500L)
beta_line <- "pmat[1:nvar, 1:nvar] <- fit$coxlist2$second"
gamma_line <- "pmat[1:nvar + nvar, 1:nvar + nvar] <- fit$coxlist2$second"
hit_beta <- which(trimws(src) == beta_line)
hit_gamma <- which(trimws(src) == gamma_line)
if (length(hit_beta) != 1 || length(hit_gamma) != 1) {
  stop("The penalty lines of cox.zph have changed, review chisq_diag before freezing it")
}
src[hit_beta] <- sub(beta_line, "pmat[1:nvar, 1:nvar] <- diag(fit$coxlist2$second, nvar)",
                     src[hit_beta], fixed = TRUE)
src <- src[-hit_gamma]
zph_diag <- cox.zph
body(zph_diag) <- parse(text = src)[[1]]

fits <- list(
  plain = coxph(Surv(time, status) ~ x1 + x2, data = d, ties = "efron", control = ctl),
  strata = coxph(Surv(time, status) ~ x1 + x2 + strata(s), data = d, ties = "efron", control = ctl),
  ridge = coxph(Surv(time, status) ~ ridge(x1, x2, theta = 0.1 * n, scale = TRUE),
                data = d, ties = "efron", control = ctl),
  ridge_strata = coxph(Surv(time, status) ~ ridge(x1, theta = 0.1 * n, scale = TRUE) + strata(s),
                       data = d, ties = "efron", control = ctl)
)

cat(R.version.string, "| survival", as.character(packageVersion("survival")), "\n")
for (name in names(fits)) {
  fit <- fits[[name]]
  cat(sprintf("[%s]\n", name))
  cat(sprintf("coef %s = %.12f\n", names(coef(fit)), coef(fit)), sep = "")
  tab <- cox.zph(fit, transform = "rank", terms = FALSE)$table
  if (!is.null(fit$coxlist2)) {
    tab_diag <- zph_diag(fit, transform = "rank", terms = FALSE)$table
  }
  for (r in setdiff(rownames(tab), "GLOBAL")) {
    line <- sprintf("%s chisq=%.15g df=%.15g p=%.15g", r, tab[r, "chisq"], tab[r, "df"], tab[r, "p"])
    if (!is.null(fit$coxlist2)) line <- sprintf("%s chisq_diag=%.15g", line, tab_diag[r, "chisq"])
    cat(line, "\n", sep = "")
  }
}
