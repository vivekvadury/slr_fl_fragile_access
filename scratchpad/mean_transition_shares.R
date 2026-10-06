# Scratch: observed mean transition share per modeled outcome, computed on the
# exact production estimation samples (approach arm, demographic_only, 5 km
# Conley run). Replays scripts/04_transition_models.R the same way
# scripts/08_final_draft_inference.R does, stopping right after model_specs is
# built (before any model fit). The only file the prefix writes (sample
# diagnostics) is redirected to scratchpad/.

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
})

OUT_DIR <- file.path("scratchpad", "mean_transition_shares")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

Sys.setenv(
  BRIDGE_ARM = "approach",
  MODEL_SPEC = "demographic_only",
  CONLEY_CUTOFF_KM = "5",
  TRANSITION_TABLE_DIR = OUT_DIR
)
Sys.unsetenv(c("AME_POP_WEIGHT", "TRANSITION_DATA_PATH"))

canonical <- file.path("scripts", "04_transition_models.R")
exprs <- parse(canonical, keep.source = FALSE)
txt <- vapply(exprs, function(e) paste(deparse(e, width.cutoff = 500L), collapse = " "), "")
stop_at <- which(startsWith(txt, "model_specs <- make_model_specs("))
stopifnot(length(stop_at) == 1L)

# 04 derives SCRIPT_DIR from this runner's --file= path; pin it to scripts/.
env <- new.env(parent = globalenv())
env$SCRIPT_DIR <- normalizePath("scripts")
for (i in seq_len(stop_at)) {
  if (startsWith(txt[i], "SCRIPT_DIR <-")) next
  eval(exprs[[i]], envir = env)
}
stopifnot(identical(env$ARM, "approach"), identical(env$MODEL_SPEC, "demographic_only"))

specs <- env$model_specs
rows <- lapply(names(specs), function(nm) {
  s <- specs[[nm]]
  d <- s$data
  y <- d[[s$outcome]]
  w <- d[[s$weights]]
  stopifnot(!anyNA(y), all(w > 0), all(y >= 0 & y <= 1))
  # Confirm the fitted model uses every row (no fixest drops).
  m <- get("fit_transition_model", envir = env)(s$outcome, d, s$weights, env$MODEL_SPEC)
  data.frame(
    transition = nm,
    outcome_column = s$outcome,
    denominator_column = s$weights,
    observations = nrow(d),
    block_groups = n_distinct(d$block_group_geoid),
    model_nobs = nobs(m),
    mean_share_unweighted = mean(y),
    pooled_share_weighted = sum(y * w) / sum(w)
  )
})
out <- bind_rows(rows) %>%
  mutate(
    mean_share_3dp = round(mean_share_unweighted, 3),
    mean_share_pct_1dp = round(100 * mean_share_unweighted, 1),
    pooled_share_pct_1dp = round(100 * pooled_share_weighted, 1),
    provenance = paste0(
      "approach arm; demographic_only; replay of ", canonical,
      " (", format(file.mtime(canonical), "%Y-%m-%d %H:%M"), ") on ",
      env$DATA_PATH, " (", format(file.mtime(env$DATA_PATH), "%Y-%m-%d %H:%M"),
      "); filters: complete z-covariates, slr_ft > 0, baseline_<origin>_n > 0"
    )
  )

# Composite check: Redundant -> Worse numerator equals the sum of its parts.
rd <- specs[["Redundant -> Worse"]]$data
comp_gap <- max(abs(rd$any_loss_of_redundancy -
  (rd$baseline_redundant_to_fragile + rd$baseline_redundant_to_isolated +
     rd$baseline_redundant_to_inundated)))
message("max |any_loss_of_redundancy - (R->F + R->Iso + R->Inund)| = ", comp_gap)

readr::write_csv(out, file.path(OUT_DIR, "mean_transition_shares.csv"))
print(out %>% select(transition, observations, block_groups, model_nobs,
                     mean_share_3dp, mean_share_pct_1dp, pooled_share_pct_1dp),
      row.names = FALSE)
