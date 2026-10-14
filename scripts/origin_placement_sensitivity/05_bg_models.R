# Block-group transition models across Monte Carlo origin replications.
#
# Exploratory (secondary question). Model preparation, risk sets, and fits are
# NOT re-implemented: like scripts/08_final_draft_inference.R, this evaluates
# scripts/04_transition_models.R verbatim up to (not including) its cluster
# bootstrap call, once per input dataset, with MODEL_SPEC=demographic_only and
# no Conley/population weighting. 04's own diagnostic CSVs go to a work dir via
# TRANSITION_TABLE_DIR, so outputs/tables is never touched.
#
# Inputs:  outputs/origin_placement_sensitivity/<run>/work_bg/bg_rep_XXX.csv
#          (written by 04_summarize.py; rep "canonical" = canonical states
#          aggregated by the same code, which must reproduce production fits)
# Outputs: outputs/origin_placement_sensitivity/<run>/bg_model_estimates.csv
#          coefficients with BG-clustered SEs and point AMEs (avg_slopes,
#          vcov = FALSE) for each replication. No inferential claims: the
#          clustered SE is reported descriptively only.
#
# Run from the repository root:
#   Rscript scripts/origin_placement_sensitivity/05_bg_models.R full

suppressPackageStartupMessages({
  library(tidyverse)
  library(fixest)
  library(marginaleffects)
})

args <- commandArgs(trailingOnly = TRUE)
RUN <- if (length(args) >= 1L) args[[1]] else "full"
RUN_DIR <- file.path("outputs", "origin_placement_sensitivity", RUN)
IN_DIR <- file.path(RUN_DIR, "work_bg")
MODEL_RUN_DIR <- file.path(IN_DIR, "model_run")
OUT_PATH <- file.path(RUN_DIR, "bg_model_estimates.csv")
CANONICAL_SCRIPT <- file.path("scripts", "04_transition_models.R")
dir.create(MODEL_RUN_DIR, recursive = TRUE, showWarnings = FALSE)

canonical_exprs <- parse(CANONICAL_SCRIPT, keep.source = FALSE)
expr_text <- vapply(
  canonical_exprs,
  function(e) paste(deparse(e, width.cutoff = 500L), collapse = " "),
  character(1)
)
stop_at <- which(startsWith(expr_text, "ame_boot_combined <- bootstrap_model_specs("))
if (length(stop_at) != 1L) {
  stop("Could not find exactly one bootstrap call in ", CANONICAL_SCRIPT)
}

fit_prefix <- function(data_path) {
  Sys.setenv(
    BRIDGE_ARM = "approach",
    MODEL_SPEC = "demographic_only",
    TRANSITION_TABLE_DIR = MODEL_RUN_DIR,
    TRANSITION_DATA_PATH = data_path
  )
  Sys.unsetenv(c("AME_POP_WEIGHT", "CONLEY_CUTOFF_KM"))
  env <- new.env(parent = globalenv())
  # 04 locates 04_shared_model_spec.R from --file= and reads CLI args; present
  # it with the arguments it would see when run as scripts/04_transition_models.R
  # with no CLI arguments (the dataset comes from TRANSITION_DATA_PATH).
  env$commandArgs <- function(trailingOnly = FALSE) {
    if (trailingOnly) character(0) else c("--file=scripts/04_transition_models.R")
  }
  for (e in canonical_exprs[seq_len(stop_at - 1L)]) {
    suppressMessages(eval(e, envir = env))
  }
  if (!identical(env$MODEL_SPEC, "demographic_only") || !identical(env$ARM, "approach") ||
      !identical(normalizePath(env$DATA_PATH), normalizePath(data_path))) {
    stop("Canonical prefix ran with the wrong spec, arm, or dataset.")
  }
  env
}

extract_estimates <- function(env, rep_label) {
  old <- getOption("marginaleffects_safe")
  on.exit(options(marginaleffects_safe = old), add = TRUE)
  options(marginaleffects_safe = FALSE)
  purrr::imap_dfr(env$transition_models, function(model, transition) {
    ct <- fixest::coeftable(model, vcov = ~block_group_geoid)
    coefs <- tibble(
      term = rownames(ct), coef = ct[, "Estimate"], coef_se_clustered = ct[, "Std. Error"]
    )
    ames <- suppressWarnings(avg_slopes(model, vcov = FALSE)) %>%
      as_tibble() %>%
      transmute(term, ame = estimate)
    spec <- env$model_specs[[transition]]
    n_bg <- dplyr::n_distinct(spec$data$block_group_geoid)
    risk <- sum(spec$data[[spec$weights]])
    events <- sum(spec$data[[spec$outcome]] * spec$data[[spec$weights]])
    coefs %>%
      left_join(ames, by = "term") %>%
      mutate(
        rep = rep_label, transition = transition, n_obs = stats::nobs(model),
        n_block_groups = n_bg, risk_set_blocks = risk, event_blocks = events
      )
  })
}

files <- sort(list.files(IN_DIR, pattern = "^bg_rep_.*\\.csv$", full.names = TRUE))
if (length(files) == 0L) stop("No replication datasets in ", IN_DIR)
message("Fitting ", length(files), " datasets from ", IN_DIR)

results <- list()
for (f in files) {
  label <- sub("^bg_rep_(.*)\\.csv$", "\\1", basename(f))
  t0 <- Sys.time()
  env <- tryCatch(fit_prefix(f), error = function(e) {
    message("FAILED ", label, ": ", conditionMessage(e))
    NULL
  })
  if (is.null(env)) {
    results[[label]] <- tibble(rep = label, transition = NA_character_, term = NA_character_,
                               fit_error = TRUE)
    next
  }
  results[[label]] <- extract_estimates(env, label) %>% mutate(fit_error = FALSE)
  message(sprintf("%s: %.1f s", label, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  rm(env)
  gc(verbose = FALSE)
}
est <- bind_rows(results)
readr::write_csv(est, OUT_PATH)
message("Wrote ", OUT_PATH)
# The reproduction check against outputs/tables/transition_model_coefficients_approach.csv
# (rep "canonical" must match production) is done in 07_compare_models.py.
