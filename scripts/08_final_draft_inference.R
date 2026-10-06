# Final-draft inference tables (analysis freeze for the October 2026 draft).
#
# Data preparation, QA assertions, risk sets, and model fits are NOT
# re-implemented here: this script evaluates scripts/04_transition_models.R
# verbatim up to (not including) its cluster-bootstrap call, once per model
# specification, so every model below is the production model. It then adds:
#   - delta-method AMEs under block-group-clustered and Conley (5/10/15 km)
#     covariance matrices;
#   - positive-definiteness checks on the raw (unrepaired) Conley matrices;
#   - a check that same-location rows (one block group observed in several SLR
#     scenarios) enter the Conley meat, and a hand-built reproduction of the
#     fixest Conley matrix;
#   - comparison against the existing 199-replication bootstrap tables in
#     outputs/tables (point estimates must match exactly);
#   - sample sizes, Pearson dispersion, Moran summaries, and descriptive tables.
#
# Outputs: outputs/final_draft/. The models' own diagnostic CSVs are redirected
# to outputs/final_draft/model_run via TRANSITION_TABLE_DIR, so nothing in
# outputs/tables is touched.
#
# Run from the repository root:
#   Rscript scripts/08_final_draft_inference.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(fixest)
  library(marginaleffects)
  library(openxlsx)
})

ARM <- "approach"
SPECS <- c("demographic_only", "with_physical")
PRIMARY_CUTOFF_KM <- 5
CUTOFFS_KM <- c(5, 10, 15)
OUT_DIR <- file.path("outputs", "final_draft")
MODEL_RUN_DIR <- file.path(OUT_DIR, "model_run")
CANONICAL_SCRIPT <- file.path("scripts", "04_transition_models.R")
dir.create(MODEL_RUN_DIR, recursive = TRUE, showWarnings = FALSE)

SOCIAL_TERMS <- c(
  z_pct_black_nh = "Black share",
  z_pct_hispanic = "Hispanic share",
  z_renter_share = "Renter share",
  z_log_median_income = "Log median income",
  z_pct_age_65plus = "Age 65+ share",
  z_no_vehicle_share = "No-vehicle household share"
)
PHYSICAL_TERMS <- c(
  z_elevation_m_mean = "Mean elevation",
  z_drainage_distance_km = "Distance to primary drainage"
)
TERM_LABELS <- c(SOCIAL_TERMS, PHYSICAL_TERMS)

# ---------------------------------------------------------------------------
# 1. Fit the production models by evaluating the canonical script's prefix.
# ---------------------------------------------------------------------------

canonical_exprs <- parse(CANONICAL_SCRIPT, keep.source = FALSE)
expr_text <- vapply(
  canonical_exprs,
  function(e) paste(deparse(e, width.cutoff = 500L), collapse = " "),
  character(1)
)
stop_at <- which(startsWith(expr_text, "ame_boot_combined <- bootstrap_model_specs("))
if (length(stop_at) != 1L) {
  stop(
    "Could not find exactly one bootstrap call in ", CANONICAL_SCRIPT,
    "; the canonical script changed. Update this runner deliberately."
  )
}

fit_production_models <- function(spec) {
  Sys.setenv(
    BRIDGE_ARM = ARM,
    MODEL_SPEC = spec,
    CONLEY_CUTOFF_KM = as.character(PRIMARY_CUTOFF_KM),
    TRANSITION_TABLE_DIR = MODEL_RUN_DIR
  )
  Sys.unsetenv(c("AME_POP_WEIGHT", "TRANSITION_DATA_PATH"))
  env <- new.env(parent = globalenv())
  for (e in canonical_exprs[seq_len(stop_at - 1L)]) {
    eval(e, envir = env)
  }
  required <- c("transition_models", "model_specs", "conley_vcovs", "sample_diagnostics")
  missing <- required[!vapply(required, exists, logical(1), envir = env, inherits = FALSE)]
  if (length(missing) > 0L) {
    stop("Canonical prefix did not create: ", paste(missing, collapse = ", "))
  }
  if (!identical(env$MODEL_SPEC, spec) || !identical(env$ARM, ARM)) {
    stop("Canonical prefix ran with the wrong spec or arm.")
  }
  env
}

fits <- setNames(lapply(SPECS, fit_production_models), SPECS)

# ---------------------------------------------------------------------------
# 2. Covariance matrices and delta-method AMEs.
# ---------------------------------------------------------------------------

# vcov_fix = FALSE returns the unrepaired matrix. fixest's argument handling
# (match.call) rejects forwarded dots, so the options are explicit.
raw_conley <- function(model, cutoff_km, distance = "triangular", ssc = NULL) {
  fixest::vcov_conley(
    model,
    lat = "centroid_lat", lon = "centroid_lon",
    cutoff = cutoff_km, distance = distance, ssc = ssc, vcov_fix = FALSE
  )
}

delta_ames <- function(model, vcov_matrix) {
  old <- getOption("marginaleffects_safe")
  on.exit(options(marginaleffects_safe = old), add = TRUE)
  options(marginaleffects_safe = FALSE)
  suppressWarnings(avg_slopes(model, vcov = vcov_matrix)) %>%
    as_tibble() %>%
    transmute(term, estimate, std_error = std.error, z = statistic,
              p_value = p.value, conf_low = conf.low, conf_high = conf.high)
}

ame_rows <- list()
pd_rows <- list()
for (spec in SPECS) {
  env <- fits[[spec]]
  for (transition in names(env$transition_models)) {
    model <- env$transition_models[[transition]]
    data <- env$model_specs[[transition]]$data
    vcovs <- list(clustered = vcov(model, vcov = ~block_group_geoid))
    for (k in CUTOFFS_KM) {
      raw <- raw_conley(model, k)
      repaired <- compute_conley_vcov(model, data, cutoff_km = k)
      eig <- eigen(raw, symmetric = TRUE, only.values = TRUE)$values
      pd_rows[[length(pd_rows) + 1L]] <- tibble(
        spec = spec, transition = transition, cutoff_km = k,
        min_eigenvalue_raw = min(eig),
        positive_definite_raw = min(eig) > 0,
        max_abs_repair = max(abs(repaired - raw))
      )
      vcovs[[paste0("conley_", k, "km")]] <- repaired
    }
    # The canonical prefix's 5 km Conley matrix must equal the one above.
    if (max(abs(env$conley_vcovs[[transition]] - vcovs$conley_5km)) > 1e-12) {
      stop("5 km Conley matrix differs from the canonical script's for ", transition)
    }
    family <- env$model_specs[[transition]]$family
    for (v in names(vcovs)) {
      ame_rows[[length(ame_rows) + 1L]] <- delta_ames(model, vcovs[[v]]) %>%
        mutate(spec = !!spec, transition = !!transition, vcov_type = !!v,
               risk_family = !!family, .before = 1)
    }
  }
}
ames <- bind_rows(ame_rows) %>%
  mutate(
    cutoff_km = suppressWarnings(as.numeric(str_extract(vcov_type, "[0-9]+"))),
    covariate = unname(TERM_LABELS[term]),
    estimate_pp = 100 * estimate,
    sig_05 = p_value < 0.05
  )
pd <- bind_rows(pd_rows)

# Point estimates do not depend on the covariance matrix.
est_spread <- ames %>%
  group_by(spec, transition, term) %>%
  summarise(spread = max(estimate) - min(estimate), .groups = "drop")
if (max(est_spread$spread) > 1e-12) {
  stop("AME point estimates differ across covariance types.")
}

# ---------------------------------------------------------------------------
# 3. Existing 199-replication bootstrap tables (supplementary comparison).
# ---------------------------------------------------------------------------

boot_paths <- c(
  demographic_only = file.path("outputs", "tables", "ame_bootstrap_results_approach.xlsx"),
  with_physical = file.path("outputs", "tables", "ame_bootstrap_results_approach_with_physical.xlsx")
)
boot <- imap_dfr(boot_paths, function(path, spec) {
  openxlsx::read.xlsx(path) %>%
    as_tibble() %>%
    transmute(spec = spec, transition, term,
              boot_estimate = estimate, boot_se = std.error, boot_p = p.value,
              boot_conf_low = conf.low, boot_conf_high = conf.high,
              n_boot, n_boot_fail)
})
boot_check <- ames %>%
  filter(vcov_type == "clustered") %>%
  inner_join(boot, by = c("spec", "transition", "term"))
if (nrow(boot_check) != nrow(filter(ames, vcov_type == "clustered"))) {
  stop("Bootstrap tables do not cover every spec/transition/term.")
}
boot_max_diff <- max(abs(boot_check$estimate - boot_check$boot_estimate))
if (boot_max_diff > 1e-10) {
  stop("Point estimates differ from the bootstrap tables by ", boot_max_diff,
       ": outputs/tables is stale relative to the canonical pipeline.")
}
if (any(boot$n_boot != 199L)) {
  stop("A bootstrap table does not contain 199 successful replications.")
}

# ---------------------------------------------------------------------------
# 4. Verification of the Conley implementation.
# ---------------------------------------------------------------------------

FIXEST_EARTH_RADIUS_KM <- 6376

haversine_km <- function(lat1, lon1, lat2, lon2, radius = 6371) {
  to_rad <- pi / 180
  dlat <- (lat2 - lat1) * to_rad
  dlon <- (lon2 - lon1) * to_rad
  a <- sin(dlat / 2)^2 + cos(lat1 * to_rad) * cos(lat2 * to_rad) * sin(dlon / 2)^2
  2 * radius * asin(pmin(1, sqrt(a)))
}

verify_rows <- list()
add_check <- function(check, value, passed, detail) {
  verify_rows[[length(verify_rows) + 1L]] <<- tibble(
    check = check, value = value, passed = passed, detail = detail
  )
}

env <- fits$demographic_only
check_transition <- "Redundant -> Worse"
model <- env$transition_models[[check_transition]]
data <- env$model_specs[[check_transition]]$data[fixest::obs(model), , drop = FALSE]

centroids <- distinct(data, block_group_geoid, centroid_lat, centroid_lon)
if (anyDuplicated(centroids$block_group_geoid)) stop("Conflicting centroids.")
dist_bg <- outer(seq_len(nrow(centroids)), seq_len(nrow(centroids)), function(i, j)
  haversine_km(centroids$centroid_lat[i], centroids$centroid_lon[i],
               centroids$centroid_lat[j], centroids$centroid_lon[j]))
diag(dist_bg) <- NA
min_distinct_km <- min(dist_bg, na.rm = TRUE)
nn_km <- apply(dist_bg, 1, min, na.rm = TRUE)
add_check("min distance between distinct block-group centroids (km)",
          min_distinct_km, min_distinct_km > 0,
          "Must be positive for the tiny-cutoff test to isolate same-location pairs.")
add_check("median nearest-neighbor centroid distance (km)", median(nn_km), TRUE,
          "Descriptive: scale of the 5 km cutoff relative to block-group spacing.")
neighbors_5km <- rowSums(dist_bg <= 5, na.rm = TRUE)
add_check("median other block groups within 5 km", median(neighbors_5km), TRUE,
          paste0("Range ", min(neighbors_5km), "-", max(neighbors_5km), "."))

# (a) With a cutoff below every between-block-group distance, the only pairs in
# the Conley meat are rows sharing a location, i.e. the same block group across
# SLR scenarios. If those pairs are included, the matrix equals the
# block-group-clustered matrix with no small-sample adjustments.
tiny_km <- min_distinct_km / 2
v_tiny <- raw_conley(model, tiny_km, distance = "spherical",
                     ssc = fixest::ssc(K.adj = FALSE))
v_cl0 <- vcov(model, vcov = ~block_group_geoid,
              ssc = fixest::ssc(K.adj = FALSE, G.adj = FALSE))
rel_tiny <- max(abs(v_tiny - v_cl0)) / max(abs(v_cl0))
add_check("Conley(tiny cutoff) vs unadjusted block-group clustered: max rel. diff",
          rel_tiny, rel_tiny < 1e-8,
          paste0("Cutoff ", signif(tiny_km, 3), " km. Equality means repeated ",
                 "scenario rows of one block group (distance 0) are inside the ",
                 "Conley kernel, so within-block-group dependence is covered."))

# (b) Hand-built Conley meat from the model's scores and Hessian.
scores <- model$scores
hessian <- model$hessian
if (is.null(scores) || is.null(hessian)) {
  add_check("hand-built Conley reproduction", NA_real_, NA,
            "fixest object lacks $scores or $hessian; not run.")
} else {
  bread <- solve(hessian)
  cluster_sum <- rowsum(scores, data$block_group_geoid)
  v_cl_manual <- bread %*% crossprod(cluster_sum) %*% bread
  rel_cl <- max(abs(v_cl_manual - v_cl0)) / max(abs(v_cl0))
  add_check("hand-built clustered vs fixest clustered (unadjusted): max rel. diff",
            rel_cl, rel_cl < 1e-6, "Validates the bread/score decomposition.")
  idx <- match(data$block_group_geoid, centroids$block_group_geoid)
  # fixest's spherical distance uses an Earth radius of 6,376 km (probed with
  # synthetic point pairs against its compiled routine), so rescale.
  within <- (dist_bg * FIXEST_EARTH_RADIUS_KM / 6371 <= PRIMARY_CUTOFF_KM)
  diag(within) <- TRUE
  # Meat = sum over block-group pairs within the cutoff of S_g S_h', where S_g
  # sums a block group's scores over its scenario rows.
  s_bg <- rowsum(scores, centroids$block_group_geoid[idx])
  s_bg <- s_bg[centroids$block_group_geoid, , drop = FALSE]
  meat <- t(s_bg) %*% (within * 1) %*% s_bg
  v_conley_manual <- bread %*% meat %*% bread
  v_conley_fixest <- raw_conley(model, PRIMARY_CUTOFF_KM, distance = "spherical",
                                ssc = fixest::ssc(K.adj = FALSE))
  rel_c <- max(abs(v_conley_manual - v_conley_fixest)) / max(abs(v_conley_fixest))
  add_check("hand-built 5 km Conley vs fixest (spherical, unadjusted): max rel. diff",
            rel_c, rel_c < 1e-8,
            "Uniform kernel; great-circle distances with fixest's 6,376 km Earth radius. Equality confirms the meat is the sum of S_g S_h' over block-group pairs within the cutoff.")
  v_tri <- raw_conley(model, PRIMARY_CUTOFF_KM, ssc = fixest::ssc(K.adj = FALSE))
  rel_tri <- max(abs(v_tri - v_conley_fixest)) / max(abs(v_conley_fixest))
  add_check("5 km Conley: triangular (production) vs spherical distance, max rel. diff",
            rel_tri, rel_tri < 1e-2, "Production uses fixest's default triangular distance.")
}
verification <- bind_rows(verify_rows)

# ---------------------------------------------------------------------------
# 5. Sample sizes and model diagnostics.
# ---------------------------------------------------------------------------

sample_sizes <- imap_dfr(fits, function(env, spec) {
  imap_dfr(env$transition_models, function(model, transition) {
    s <- env$model_specs[[transition]]
    d <- s$data[fixest::obs(model), , drop = FALSE]
    baseline <- distinct(d, block_group_geoid, .keep_all = TRUE)
    tibble(
      spec = spec, transition = transition, risk_family = s$family,
      weight_column = s$weights,
      observations = nobs(model),
      block_groups = n_distinct(d$block_group_geoid),
      slr_scenarios = n_distinct(d$slr_ft),
      baseline_risk_set_blocks = sum(baseline[[s$weights]]),
      block_scenario_trials = sum(d[[s$weights]]),
      transition_events = sum(round(d[[s$outcome]] * d[[s$weights]])),
      weighted_mean_share = weighted.mean(d[[s$outcome]], d[[s$weights]]),
      counties = n_distinct(d$county_name)
    )
  })
})
covariate_filters <- imap_dfr(fits, function(env, spec) {
  env$sample_diagnostics %>% mutate(spec = spec, .before = 1)
})

dispersion <- imap_dfr(fits, function(env, spec) {
  imap_dfr(env$transition_models, function(model, transition) {
    r <- as.numeric(stats::resid(model, type = "pearson"))
    df <- as.numeric(fixest::degrees_freedom(model, type = "resid"))
    d <- env$model_specs[[transition]]$data[fixest::obs(model), , drop = FALSE]
    top <- which.max(r^2)
    # A single row with a near-zero fitted probability and observed events can
    # dominate the Pearson statistic (it divides by p(1 - p)); report it.
    tibble(spec = spec, transition = transition, converged = isTRUE(model$convStatus),
           residual_df = df, pearson_dispersion = sum(r^2) / df,
           pearson_dispersion_excl_top_row = sum(r[-top]^2) / (df - 1),
           top_row_share_of_pearson = max(r^2) / sum(r^2),
           top_row = paste0(d$block_group_geoid[top], "@", d$slr_ft[top], "ft"),
           top_row_fitted_p = stats::fitted(model)[top],
           min_fitted_p = min(stats::fitted(model)))
  })
})

moran_paths <- c(
  demographic_only = file.path("outputs", "run_comparison", "moran_residual_diagnostics_approach.csv"),
  with_physical = file.path("outputs", "run_comparison", "moran_residual_diagnostics_approach_with_physical.csv")
)
moran <- imap_dfr(moran_paths, function(path, spec) {
  readr::read_csv(path, show_col_types = FALSE) %>% mutate(spec = spec)
}) %>%
  group_by(spec, transition) %>%
  summarise(moran_tests = n(), moran_significant_05 = sum(significant_0_05),
            moran_i_min = min(moran_i), moran_i_max = max(moran_i), .groups = "drop")

pd_wide <- pd %>%
  transmute(spec, transition, cutoff_km,
            value = if_else(positive_definite_raw, "PD", "not PD (eigen-repaired)")) %>%
  pivot_wider(names_from = cutoff_km, values_from = value, names_prefix = "conley_")

diagnostics <- sample_sizes %>%
  select(spec, transition, observations, block_groups) %>%
  left_join(dispersion, by = c("spec", "transition")) %>%
  left_join(moran, by = c("spec", "transition")) %>%
  left_join(pd_wide, by = c("spec", "transition"))

# ---------------------------------------------------------------------------
# 6. Output tables.
# ---------------------------------------------------------------------------

social <- ames %>% filter(term %in% names(SOCIAL_TERMS))
transition_order <- names(fits$demographic_only$transition_models)
order_rows <- function(df) {
  df %>% arrange(factor(transition, transition_order), factor(term, names(TERM_LABELS)))
}

main_cols <- c("spec", "transition", "risk_family", "term", "covariate", "estimate",
               "estimate_pp", "std_error", "z", "p_value", "conf_low", "conf_high", "sig_05")
main <- social %>%
  filter(vcov_type == "conley_5km") %>%
  select(all_of(main_cols)) %>%
  left_join(select(sample_sizes, spec, transition, observations, block_groups),
            by = c("spec", "transition")) %>%
  mutate(vcov = "Conley (1999), uniform kernel, 5 km cutoff; delta-method AME")

main_demo <- main %>% filter(spec == "demographic_only") %>% order_rows()
main_phys <- ames %>%
  filter(vcov_type == "conley_5km", spec == "with_physical") %>%
  select(all_of(main_cols)) %>% order_rows()
readr::write_csv(main_demo, file.path(OUT_DIR, "main_transition_ames_conley5.csv"))
readr::write_csv(main_phys, file.path(OUT_DIR, "transition_ames_conley5_with_physical.csv"))
openxlsx::write.xlsx(
  list(demographic_only = main_demo, with_physical = main_phys),
  file.path(OUT_DIR, "main_transition_ames_conley5.xlsx"), overwrite = TRUE
)

cutoff_sens <- ames %>%
  filter(str_starts(vcov_type, "conley")) %>%
  select(spec, transition, term, covariate, cutoff_km, estimate, std_error, p_value, sig_05) %>%
  left_join(select(pd, spec, transition, cutoff_km, positive_definite_raw, min_eigenvalue_raw),
            by = c("spec", "transition", "cutoff_km")) %>%
  arrange(spec, cutoff_km) %>% order_rows()
readr::write_csv(cutoff_sens, file.path(OUT_DIR, "conley_cutoff_sensitivity.csv"))

pick <- function(v) {
  ames %>% filter(vcov_type == v) %>%
    select(spec, transition, term, std_error, p_value) %>%
    rename_with(~ paste0(.x, "_", v), c(std_error, p_value))
}
cluster_vs_conley <- ames %>%
  filter(vcov_type == "conley_5km") %>%
  select(spec, transition, term, covariate, estimate) %>%
  left_join(pick("clustered"), by = c("spec", "transition", "term")) %>%
  left_join(pick("conley_5km"), by = c("spec", "transition", "term")) %>%
  left_join(select(boot, spec, transition, term, boot_se, boot_p, boot_conf_low, boot_conf_high),
            by = c("spec", "transition", "term")) %>%
  mutate(
    se_ratio_conley5_to_clustered = std_error_conley_5km / std_error_clustered,
    se_ratio_conley5_to_bootstrap = std_error_conley_5km / boot_se,
    sig05_clustered = p_value_clustered < 0.05,
    sig05_conley5 = p_value_conley_5km < 0.05,
    sig05_bootstrap = boot_p < 0.05,
    sig05_agreement = case_when(
      sig05_clustered & sig05_conley5 & sig05_bootstrap ~ "significant under all three",
      !sig05_clustered & !sig05_conley5 & !sig05_bootstrap ~ "not significant under any",
      TRUE ~ "differs"
    )
  ) %>% order_rows()
readr::write_csv(cluster_vs_conley, file.path(OUT_DIR, "cluster_vs_conley.csv"))

phys_cmp <- ames %>%
  filter(vcov_type == "conley_5km") %>%
  select(spec, transition, term, covariate, estimate, std_error, p_value) %>%
  pivot_wider(names_from = spec, values_from = c(estimate, std_error, p_value)) %>%
  mutate(
    change_pp = 100 * (estimate_with_physical - estimate_demographic_only),
    pct_change = 100 * (estimate_with_physical - estimate_demographic_only) /
      abs(estimate_demographic_only),
    sign_change = sign(estimate_with_physical) != sign(estimate_demographic_only),
    sig05_demographic_only = p_value_demographic_only < 0.05,
    sig05_with_physical = p_value_with_physical < 0.05
  ) %>% order_rows()
readr::write_csv(phys_cmp, file.path(OUT_DIR, "physical_controls_comparison.csv"))

readr::write_csv(sample_sizes, file.path(OUT_DIR, "model_sample_sizes.csv"))
readr::write_csv(covariate_filters, file.path(OUT_DIR, "model_covariate_filter_diagnostics.csv"))
readr::write_csv(diagnostics, file.path(OUT_DIR, "diagnostics_summary.csv"))
readr::write_csv(pd, file.path(OUT_DIR, "conley_positive_definiteness.csv"))
readr::write_csv(verification, file.path(OUT_DIR, "conley_verification.csv"))
readr::write_csv(ames %>% order_rows(), file.path(OUT_DIR, "all_ames_all_vcovs.csv"))

# Manuscript-style LaTeX table for the primary specification.
stars <- function(p) case_when(p < 0.001 ~ "***", p < 0.01 ~ "**", p < 0.05 ~ "*", TRUE ~ "")
headers <- c(
  "Redundant -> Fragile" = "Red.$\\to$Frag.", "Redundant -> Isolated" = "Red.$\\to$Iso.",
  "Redundant -> Inundated" = "Red.$\\to$Inund.", "Redundant -> Worse" = "Red.$\\to$Worse",
  "Fragile -> Isolated" = "Frag.$\\to$Iso.", "Fragile -> Inundated" = "Frag.$\\to$Inund.",
  "Fragile -> Worse" = "Frag.$\\to$Worse"
)
cell <- function(df, tr, tm, what) {
  r <- df[df$transition == tr & df$term == tm, ]
  if (what == "est") sprintf("%.3f%s", r$estimate_pp, stars(r$p_value))
  else sprintf("(%.3f)", 100 * r$std_error)
}
tex <- c(
  "% Auto-generated by scripts/08_final_draft_inference.R",
  "\\begin{table}[!htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{4pt}",
  "\\caption{Average marginal effects on transition probabilities (percentage points per 1 SD)}",
  "\\label{tab:ame_transition_probabilities}",
  paste0("\\begin{tabular}{l", strrep("c", length(transition_order)), "}"), "\\hline",
  paste0("Covariate & ", paste(headers[transition_order], collapse = " & "), " \\\\"), "\\hline"
)
for (tm in names(SOCIAL_TERMS)) {
  tex <- c(tex,
    paste0(SOCIAL_TERMS[[tm]], " & ", paste(vapply(transition_order, cell, "", df = main_demo, tm = tm, what = "est"), collapse = " & "), " \\\\"),
    paste0(" & ", paste(vapply(transition_order, cell, "", df = main_demo, tm = tm, what = "se"), collapse = " & "), " \\\\"))
}
ss <- sample_sizes %>% filter(spec == "demographic_only")
tex <- c(tex, "\\hline",
  paste0("Observations & ", paste(formatC(ss$observations[match(transition_order, ss$transition)], big.mark = ",", format = "d"), collapse = " & "), " \\\\"),
  paste0("Block groups & ", paste(formatC(ss$block_groups[match(transition_order, ss$transition)], big.mark = ",", format = "d"), collapse = " & "), " \\\\"),
  "\\hline",
  paste0("\\multicolumn{", length(transition_order) + 1L, "}{p{0.95\\linewidth}}{\\footnotesize Notes: ",
         "Average marginal effects in percentage points per one-standard-deviation increase. ",
         "Standard errors (in parentheses) are Conley (1999) spatial HAC with a uniform kernel and 5~km cutoff, ",
         "propagated to the AMEs by the delta method. All models include county and SLR-scenario fixed effects. ",
         "* $p<0.05$, ** $p<0.01$, *** $p<0.001$.}\\\\"),
  "\\hline", "\\end{tabular}", "\\end{table}")
writeLines(tex, file.path(OUT_DIR, "main_transition_ames_conley5.tex"))

# ---------------------------------------------------------------------------
# 7. Descriptive tables, assembled from the production population outputs with
#    consistency checks (these files come from scripts/05_population_figures.py).
# ---------------------------------------------------------------------------

status <- readr::read_csv(file.path("outputs", "tables", "fig4_status_population_by_slr_approach.csv"),
                          show_col_types = FALSE)
totals <- status %>% group_by(slr_ft) %>%
  summarise(blocks = sum(n_blocks), pop = sum(pop20),
            share_b = sum(share_of_blocks), share_p = sum(share_of_pop20), .groups = "drop")
if (any(totals$blocks != 68521L) || any(totals$pop != 6135688) ||
    any(abs(totals$share_b - 1) > 1e-9) || any(abs(totals$share_p - 1) > 1e-9)) {
  stop("Access-state totals are not constant at 68,521 blocks / 6,135,688 residents.")
}
if (any(status$n_blocks[status$scenario_status == "unclassified"] != 0)) {
  stop("Unclassified blocks present in the eligible universe.")
}
readr::write_csv(status, file.path(OUT_DIR, "descriptive_access_by_slr.csv"))

cum <- readr::read_csv(file.path("outputs", "tables", "fig4_cumulative_population_by_slr_approach.csv"),
                       show_col_types = FALSE)
cum_cols <- c("new_inundated_pop20", "new_isolated_or_inundated_pop20", "new_fragile_or_worse_pop20")
if (any(apply(cum[cum_cols], 2, diff) < 0)) stop("Cumulative population thresholds are not monotone.")
if (any(cum$new_inundated_pop20 > cum$new_isolated_or_inundated_pop20) ||
    any(cum$new_isolated_or_inundated_pop20 > cum$new_fragile_or_worse_pop20)) {
  stop("Population thresholds are not nested.")
}
# Manuscript Section 4.5 definitions: the five adverse transitions from
# baseline-redundant and baseline-fragile blocks; non-inundation share over
# population. transition_summary_by_slr_*.csv is not used because its
# "new inundated" also counts baseline-isolated blocks that become inundated.
thresholds <- cum %>%
  mutate(
    new_isolated_only_pop20 = new_isolated_or_inundated_pop20 - new_inundated_pop20,
    non_inundation_share_pop20 = 1 - new_inundated_pop20 / new_fragile_or_worse_pop20,
    non_inundation_share_blocks = 1 - new_inundated_blocks / new_fragile_or_worse_blocks
  )
readr::write_csv(thresholds, file.path(OUT_DIR, "population_thresholds_by_slr.csv"))

writeLines(capture.output(sessionInfo()), file.path(OUT_DIR, "r_session_info.txt"))
message("Final-draft inference tables written to ", OUT_DIR)
print(verification, width = 200)
