# Read-only verification of production inputs/models. Run from repository root.
# All writes are confined to this audit folder; no bootstrap is run.
suppressPackageStartupMessages({library(tidyverse); library(fixest); library(marginaleffects); library(openxlsx)})
OUT <- "outputs/elevation_drainage_audit_20261006"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
exprs <- parse("scripts/04_transition_models.R", keep.source = FALSE)
texts <- vapply(exprs, function(e) paste(deparse(e, width.cutoff=500L), collapse=" "), character(1))
stop_at <- which(startsWith(texts, "ame_boot_combined <- bootstrap_model_specs("))
stopifnot(length(stop_at)==1L)
fits <- list()
for (spec in c("demographic_only", "with_physical")) {
  Sys.setenv(BRIDGE_ARM="approach", MODEL_SPEC=spec, CONLEY_CUTOFF_KM="5",
             TRANSITION_TABLE_DIR=file.path(OUT,"model_run"))
  Sys.unsetenv(c("AME_POP_WEIGHT", "TRANSITION_DATA_PATH"))
  env <- new.env(parent=globalenv())
  for (j in seq_len(stop_at-1L)) {
    # The harness is outside scripts/: resolve only the shared-source directory.
    if (startsWith(texts[j], "source(file.path(SCRIPT_DIR")) env$SCRIPT_DIR <- normalizePath("scripts")
    eval(exprs[[j]], envir=env)
  }
  stopifnot(env$ARM=="approach",env$MODEL_SPEC==spec)
  fits[[spec]] <- env
}
ame_rows <- list(); sample_rows <- list(); pd_rows <- list(); check_rows <- list()
for (spec in names(fits)) {
  env <- fits[[spec]]
  for (transition in names(env$transition_models)) {
    message("AUDIT AMEs: ",spec," / ",transition)
    model <- env$transition_models[[transition]]
    s <- env$model_specs[[transition]]
    d <- s$data[fixest::obs(model),,drop=FALSE]
    baseline <- distinct(d,block_group_geoid,.keep_all=TRUE)
    sample_rows[[length(sample_rows)+1L]] <- tibble(spec,transition,
      observations=nobs(model),block_groups=n_distinct(d$block_group_geoid),
      baseline_risk_set_blocks=sum(baseline[[s$weights]]),
      block_scenario_trials=sum(d[[s$weights]]),
      transition_events=sum(round(d[[s$outcome]]*d[[s$weights]])),
      converged=isTRUE(model$convStatus),slr_scenarios=n_distinct(d$slr_ft),
      counties=n_distinct(d$county_name),weight_column=s$weights)
    raw <- fixest::vcov_conley(model,lat="centroid_lat",lon="centroid_lon",cutoff=5,vcov_fix=FALSE)
    eig <- eigen(raw,symmetric=TRUE,only.values=TRUE)$values
    pd_rows[[length(pd_rows)+1L]] <- tibble(spec,transition,min_eigenvalue_raw=min(eig),
      positive_definite=min(eig)>0,max_abs_repair=max(abs(raw-env$conley_vcovs[[transition]])))
    options(marginaleffects_safe=FALSE)
    a <- avg_slopes(model,vcov=env$conley_vcovs[[transition]]) %>% as_tibble() %>%
      transmute(spec=spec,transition=transition,term,estimate,std_error=std.error,
                z=statistic,p_value=p.value,conf_low=conf.low,conf_high=conf.high)
    ame_rows[[length(ame_rows)+1L]] <- a
    if (spec=="with_physical") {
      e0 <- fits$demographic_only
      m0 <- e0$transition_models[[transition]]
      s0 <- e0$model_specs[[transition]]
      d0 <- s0$data[fixest::obs(m0),,drop=FALSE]
      cols <- c("block_group_geoid","slr_ft","county_name",s$outcome,s$weights,
                get_model_covariates("demographic_only"),"centroid_lat","centroid_lon")
      # Compare actual columns, not the spec-specific diagnostic attributes
      # inherited by these data frames from prepare_transition_data().
      column_checks <- vapply(cols,function(k) identical(d[[k]],d0[[k]]),logical(1))
      same <- nrow(d)==nrow(d0) && all(column_checks)
      if (!same) print(cols[!column_checks])
      stopifnot(same)
      check_rows[[length(check_rows)+1L]] <- tibble(transition,
        identical_sample_rows_outcomes_denominators_social_zscores_FE_coordinates=same)
    }
  }
}
ames <- bind_rows(ame_rows)
readr::write_csv(ames,file.path(OUT,"reproduced_ames_conley5.csv"))
readr::write_csv(bind_rows(sample_rows),file.path(OUT,"verified_model_samples.csv"))
readr::write_csv(bind_rows(pd_rows),file.path(OUT,"verified_conley5_covariance.csv"))
readr::write_csv(bind_rows(check_rows),file.path(OUT,"sample_identity_checks.csv"))
saved <- readr::read_csv("outputs/final_draft/all_ames_all_vcovs.csv",show_col_types=FALSE) %>%
  filter(vcov_type=="conley_5km") %>% select(all_of(names(ames)))
cmp <- inner_join(ames,saved,by=c("spec","transition","term"),suffix=c("_current","_saved"))
stopifnot(nrow(cmp)==98L,nrow(ames)==98L,nrow(saved)==98L)
checks <- map_dfr(c("estimate","std_error","z","p_value","conf_low","conf_high"),function(field) {
  delta <- abs(cmp[[paste0(field,"_current")]]-cmp[[paste0(field,"_saved")]])
  tibble(field,max_abs_difference=max(delta),tolerance=1e-8,passed=max(delta)<1e-8)
})
readr::write_csv(checks,file.path(OUT,"saved_vs_current_checks.csv"))
writeLines(capture.output(sessionInfo()),file.path(OUT,"r_session_info.txt"))
print(checks)
stopifnot(all(checks$passed))
message("PASS: current models reproduce all 98 saved 5-km Conley AMEs and uncertainty fields.")
