# Verify the existing extreme-residual warning without changing production files.
suppressPackageStartupMessages({library(tidyverse); library(fixest); library(marginaleffects); library(openxlsx)})
OUT <- "outputs/elevation_drainage_audit_20261006"
Sys.setenv(BRIDGE_ARM="approach",MODEL_SPEC="with_physical",CONLEY_CUTOFF_KM="5",
           TRANSITION_TABLE_DIR=file.path(OUT,"model_run"))
Sys.unsetenv(c("AME_POP_WEIGHT","TRANSITION_DATA_PATH"))
exprs <- parse("scripts/04_transition_models.R",keep.source=FALSE)
texts <- vapply(exprs,function(e) paste(deparse(e,width.cutoff=500L),collapse=" "),character(1))
last <- which(startsWith(texts,"transition_models <- fit_model_specs("))
stopifnot(length(last)==1L)
env <- new.env(parent=globalenv())
for (j in seq_len(last)) {
  if (startsWith(texts[j],"source(file.path(SCRIPT_DIR")) env$SCRIPT_DIR <- normalizePath("scripts")
  eval(exprs[[j]],envir=env)
}
diagnostics <- imap_dfr(env$transition_models,function(model,transition) {
  s <- env$model_specs[[transition]]
  d <- s$data[fixest::obs(model),,drop=FALSE]
  r <- as.numeric(stats::resid(model,type="pearson"))
  top <- which.max(r^2)
  df <- as.numeric(fixest::degrees_freedom(model,type="resid"))
  tibble(spec="with_physical",transition,converged=isTRUE(model$convStatus),
    pearson_dispersion=sum(r^2)/df,top_row_share_of_pearson=max(r^2)/sum(r^2),
    top_block_group=d$block_group_geoid[top],top_slr_ft=d$slr_ft[top],
    top_row_fitted_probability=stats::fitted(model)[top],
    top_row_observed_share=d[[s$outcome]][top],top_row_denominator=d[[s$weights]][top],
    pearson_dispersion_excl_top_row=sum(r[-top]^2)/(df-1))
})
write_csv(diagnostics,file.path(OUT,"verified_adjusted_diagnostics.csv"))
saved <- read_csv("outputs/final_draft/diagnostics_summary.csv",show_col_types=FALSE) %>%
  filter(spec=="with_physical")
comparison <- inner_join(diagnostics,saved,by=c("spec","transition"),suffix=c("_fresh","_saved"))
for (field in c("pearson_dispersion","top_row_share_of_pearson","pearson_dispersion_excl_top_row")) {
  stopifnot(max(abs(comparison[[paste0(field,"_fresh")]]-comparison[[paste0(field,"_saved")]]))<1e-7)
}
print(diagnostics)
message("PASS: adjusted fit diagnostics reproduced from current inputs.")
