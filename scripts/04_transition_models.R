# Manuscript transition models and Table 2 exports.
#
# This script estimates the grouped binomial transition models described in
# the manuscript and exports the cluster-bootstrap AME table used as Table 2.
#
# Bootstrap controls:
# - AME_BOOT_REPS sets successful bootstrap replications; default is 199.
# - AME_BOOT_SEED sets the base seed; default is 20260411.
# - AME_BOOT_MAX_ATTEMPTS sets the retry cap for failed bootstrap fits.

library(tidyverse)
library(fixest)
library(marginaleffects)
library(openxlsx)

DATA_PATH <- "data/processed/analysis/block_group_analysis_dataset.csv"
TABLE_DIR <- file.path("outputs", "tables")
AME_EXCEL_PATH <- file.path(TABLE_DIR, "ame_bootstrap_results.xlsx")
AME_LATEX_PATH <- file.path(TABLE_DIR, "ame_bootstrap_transition_table.tex")

CORE_COVARIATES <- c(
  "pct_black_nh",
  "pct_hispanic",
  "renter_share",
  "log_median_income",
  "pct_age_65plus",
  "no_vehicle_share"
)

MODEL_COVARIATES <- c(
  "z_pct_black_nh",
  "z_pct_hispanic",
  "z_renter_share",
  "z_log_median_income",
  "z_pct_age_65plus",
  "z_no_vehicle_share"
)

MODEL_RHS <- paste(MODEL_COVARIATES, collapse = " + ")

read_analysis_data <- function(path = DATA_PATH) {
  readr::read_csv(
    path,
    show_col_types = FALSE,
    col_types = cols(
      block_group_geoid = col_character(),
      tract_geoid = col_character(),
      county_fips = col_character()
    )
  ) %>%
    select(-any_of("poverty_rate"))
}

prepare_transition_data <- function(dat) {
  base_counts <- dat %>%
    filter(slr_ft == 0) %>%
    transmute(
      block_group_geoid,
      baseline_total_blocks = total_blocks,
      baseline_redundant_n = block_centroid_redundant,
      baseline_fragile_n = block_centroid_fragile,
      baseline_isolated_n = block_centroid_isolated,
      baseline_inundated_n = block_centroid_inundated
    )

  dat %>%
    left_join(base_counts, by = "block_group_geoid") %>%
    mutate(
      slr_ft_f = factor(slr_ft),
      prop_red_to_worse = if_else(
        baseline_redundant_n > 0,
        any_loss_of_redundancy / baseline_redundant_n,
        NA_real_
      ),
      prop_red_to_fragile = if_else(
        baseline_redundant_n > 0,
        baseline_redundant_to_fragile / baseline_redundant_n,
        NA_real_
      ),
      prop_red_to_isolated = if_else(
        baseline_redundant_n > 0,
        baseline_redundant_to_isolated / baseline_redundant_n,
        NA_real_
      ),
      prop_red_to_inundated = if_else(
        baseline_redundant_n > 0,
        baseline_redundant_to_inundated / baseline_redundant_n,
        NA_real_
      ),
      prop_fragile_to_isolated = if_else(
        baseline_fragile_n > 0,
        baseline_fragile_to_isolated / baseline_fragile_n,
        NA_real_
      ),
      prop_fragile_to_inundated = if_else(
        baseline_fragile_n > 0,
        baseline_fragile_to_inundated / baseline_fragile_n,
        NA_real_
      )
    ) %>%
    mutate(
      across(
        all_of(CORE_COVARIATES),
        ~ as.numeric(scale(.x)),
        .names = "z_{.col}"
      )
    ) %>%
    drop_na(all_of(MODEL_COVARIATES))
}

fit_transition_model <- function(outcome, data, weight_var) {
  formula <- as.formula(
    paste0(outcome, " ~ ", MODEL_RHS, " | county_name + slr_ft_f")
  )
  feglm(
    formula,
    data = data,
    family = binomial(),
    weights = as.formula(paste0("~ ", weight_var)),
    vcov = ~ block_group_geoid
  )
}

get_int_env <- function(env_name, default) {
  value <- suppressWarnings(as.integer(Sys.getenv(env_name, unset = as.character(default))))
  if (is.na(value) || value <= 0) {
    return(default)
  }
  value
}

AME_BOOT_REPS <- get_int_env("AME_BOOT_REPS", 199L)
AME_BOOT_SEED <- get_int_env("AME_BOOT_SEED", 20260411L)
AME_BOOT_MAX_ATTEMPTS <- get_int_env("AME_BOOT_MAX_ATTEMPTS", AME_BOOT_REPS + 50L)

bootstrap_avg_slopes <- function(
    model,
    data,
    outcome,
    weight_var,
    cluster = "block_group_geoid",
    reps = AME_BOOT_REPS,
    seed = AME_BOOT_SEED,
    max_attempts = AME_BOOT_MAX_ATTEMPTS,
    conf_level = 0.95,
    label = deparse(formula(model)[[2]])
) {
  point_estimates <- avg_slopes(model, vcov = FALSE) %>%
    as_tibble() %>%
    select(term, estimate)

  cluster_ids <- unique(as.character(data[[cluster]]))
  split_data <- split(data, as.character(data[[cluster]]), drop = TRUE)
  n_clusters <- length(cluster_ids)

  if (n_clusters == 0) {
    stop("No clusters were available for bootstrap resampling.")
  }

  boot_draws <- matrix(
    NA_real_,
    nrow = reps,
    ncol = nrow(point_estimates),
    dimnames = list(NULL, point_estimates$term)
  )

  set.seed(seed)
  success <- 0L
  attempts <- 0L

  message(sprintf(
    "Bootstrap AME SEs for %s (%d successful reps requested)...",
    label,
    reps
  ))

  while (success < reps && attempts < max_attempts) {
    attempts <- attempts + 1L
    sampled_clusters <- sample(cluster_ids, size = n_clusters, replace = TRUE)
    boot_data <- bind_rows(split_data[sampled_clusters])

    boot_formula <- as.formula(
      paste0(outcome, " ~ ", MODEL_RHS, " | county_name + slr_ft_f")
    )
    boot_weights <- as.formula(paste0("~ ", weight_var))

    boot_model <- tryCatch(
      suppressWarnings(
        feglm(
          boot_formula,
          data = boot_data,
          family = binomial(),
          weights = boot_weights,
          vcov = "iid",
          notes = FALSE
        )
      ),
      error = function(e) NULL
    )
    if (is.null(boot_model)) {
      next
    }

    boot_ame <- tryCatch(
      suppressWarnings(avg_slopes(boot_model, vcov = FALSE) %>% as_tibble()),
      error = function(e) NULL
    )
    if (is.null(boot_ame)) {
      next
    }

    success <- success + 1L
    boot_draws[success, match(boot_ame$term, point_estimates$term)] <- boot_ame$estimate
  }

  if (success == 0L) {
    stop(sprintf("All bootstrap replications failed for %s.", label))
  }

  if (success < reps) {
    warning(sprintf(
      "Only %d of %d requested bootstrap replications succeeded for %s.",
      success,
      reps,
      label
    ))
  }

  boot_draws <- boot_draws[seq_len(success), , drop = FALSE]
  alpha <- (1 - conf_level) / 2

  se <- apply(boot_draws, 2, sd, na.rm = TRUE)
  conf_low <- apply(boot_draws, 2, quantile, probs = alpha, na.rm = TRUE, names = FALSE)
  conf_high <- apply(boot_draws, 2, quantile, probs = 1 - alpha, na.rm = TRUE, names = FALSE)

  point_estimates %>%
    mutate(
      std.error = unname(se[term]),
      statistic = if_else(!is.na(std.error) & std.error > 0, estimate / std.error, NA_real_),
      p.value = if_else(!is.na(statistic), 2 * pnorm(abs(statistic), lower.tail = FALSE), NA_real_),
      conf.low = unname(conf_low[term]),
      conf.high = unname(conf_high[term]),
      n_boot = success,
      n_boot_fail = attempts - success,
      conf.level = conf_level
    )
}

format_ame_estimate <- function(estimate, p_value, digits = 3) {
  stars <- case_when(
    is.na(p_value) ~ "",
    p_value < 0.001 ~ "***",
    p_value < 0.01 ~ "**",
    p_value < 0.05 ~ "*",
    TRUE ~ ""
  )
  ifelse(
    is.na(estimate),
    "",
    paste0(formatC(estimate, digits = digits, format = "f"), stars)
  )
}

format_ame_se <- function(std_error, digits = 3) {
  ifelse(
    is.na(std_error),
    "",
    paste0("(", formatC(std_error, digits = digits, format = "f"), ")")
  )
}

latex_row <- function(x) {
  paste0(paste(x, collapse = " & "), " \\\\")
}

make_model_specs <- function(redrisk_dat, fragrisk_dat) {
  list(
    "Redundant -> Fragile" = list(
      data = redrisk_dat,
      outcome = "prop_red_to_fragile",
      weights = "baseline_redundant_n"
    ),
    "Redundant -> Isolated" = list(
      data = redrisk_dat,
      outcome = "prop_red_to_isolated",
      weights = "baseline_redundant_n"
    ),
    "Redundant -> Inundated" = list(
      data = redrisk_dat,
      outcome = "prop_red_to_inundated",
      weights = "baseline_redundant_n"
    ),
    "Redundant -> Worse" = list(
      data = redrisk_dat,
      outcome = "prop_red_to_worse",
      weights = "baseline_redundant_n"
    ),
    "Fragile -> Isolated" = list(
      data = fragrisk_dat,
      outcome = "prop_fragile_to_isolated",
      weights = "baseline_fragile_n"
    ),
    "Fragile -> Inundated" = list(
      data = fragrisk_dat,
      outcome = "prop_fragile_to_inundated",
      weights = "baseline_fragile_n"
    ),
    "Fragile -> Worse" = list(
      data = fragrisk_dat,
      outcome = "prop_fragile_to_worse",
      weights = "baseline_fragile_n"
    )
  )
}

fit_model_specs <- function(specs) {
  purrr::map(
    specs,
    ~ fit_transition_model(.x$outcome, .x$data, .x$weights)
  )
}

bootstrap_model_specs <- function(models, specs) {
  purrr::imap_dfr(
    models,
    function(model, transition) {
      transition_index <- match(transition, names(models))
      bootstrap_avg_slopes(
        model,
        specs[[transition]]$data,
        outcome = specs[[transition]]$outcome,
        weight_var = specs[[transition]]$weights,
        label = transition,
        seed = AME_BOOT_SEED + transition_index
      ) %>%
        mutate(transition = transition, .before = 1)
    }
  ) %>%
    select(
      transition,
      term,
      estimate,
      std.error,
      statistic,
      p.value,
      conf.low,
      conf.high,
      conf.level,
      n_boot,
      n_boot_fail
    )
}

build_ame_table <- function(ame_boot_combined, transition_order, term_labels) {
  empty_transition_cells <- as.list(rep("", length(transition_order)))
  names(empty_transition_cells) <- transition_order

  ame_table_long <- ame_boot_combined %>%
    filter(
      term %in% names(term_labels),
      transition %in% transition_order
    ) %>%
    mutate(
      term = factor(term, levels = names(term_labels)),
      transition = factor(transition, levels = transition_order),
      estimate_cell = format_ame_estimate(estimate, p.value),
      se_cell = format_ame_se(std.error)
    ) %>%
    arrange(term, transition)

  estimate_wide <- ame_table_long %>%
    select(term, transition, estimate_cell) %>%
    pivot_wider(
      names_from = transition,
      values_from = estimate_cell,
      values_fill = ""
    )

  se_wide <- ame_table_long %>%
    select(term, transition, se_cell) %>%
    pivot_wider(
      names_from = transition,
      values_from = se_cell,
      values_fill = ""
    )

  build_rows <- function(term_name) {
    est_row <- estimate_wide %>% filter(term == term_name)
    se_row <- se_wide %>% filter(term == term_name)

    est_cells <- if (nrow(est_row) == 0) {
      empty_transition_cells
    } else {
      as.list(est_row[1, transition_order, drop = FALSE])
    }

    se_cells <- if (nrow(se_row) == 0) {
      empty_transition_cells
    } else {
      as.list(se_row[1, transition_order, drop = FALSE])
    }

    bind_rows(
      tibble(Covariate = unname(term_labels[term_name]), !!!est_cells),
      tibble(Covariate = "", !!!se_cells)
    )
  }

  purrr::map_dfr(names(term_labels), build_rows)
}

build_diagnostic_rows <- function(models, specs, transition_order) {
  diagnostic_values <- tibble(
    Covariate = c(
      "Mean transition share",
      "Observations",
      "Block groups",
      "County FE",
      "SLR-scenario FE"
    )
  )

  for (transition in transition_order) {
    spec <- specs[[transition]]
    model_data <- spec$data[obs(models[[transition]]), , drop = FALSE]
    diagnostic_values[[transition]] <- c(
      formatC(
        weighted.mean(model_data[[spec$outcome]], model_data[[spec$weights]], na.rm = TRUE),
        digits = 3,
        format = "f"
      ),
      formatC(nobs(models[[transition]]), format = "d", big.mark = ","),
      formatC(n_distinct(model_data$block_group_geoid), format = "d", big.mark = ","),
      "Yes",
      "Yes"
    )
  }

  diagnostic_values
}

write_ame_outputs <- function(ame_boot_combined, models, specs) {
  dir.create(TABLE_DIR, showWarnings = FALSE, recursive = TRUE)
  write.xlsx(ame_boot_combined, file = AME_EXCEL_PATH, overwrite = TRUE)

  term_labels <- c(
    z_pct_black_nh = "Black share (z)",
    z_pct_hispanic = "Hispanic share (z)",
    z_renter_share = "Renter share (z)",
    z_log_median_income = "Log median income (z)",
    z_pct_age_65plus = "Age 65+ share (z)",
    z_no_vehicle_share = "No-vehicle hh share (z)"
  )

  transition_order <- names(specs)
  transition_headers <- c(
    "Redundant -> Fragile" = "Red. $\\to$ Frag.",
    "Redundant -> Isolated" = "Red. $\\to$ Iso.",
    "Redundant -> Inundated" = "Red. $\\to$ Inund.",
    "Redundant -> Worse" = "Red. $\\to$ Worse",
    "Fragile -> Isolated" = "Frag. $\\to$ Iso.",
    "Fragile -> Inundated" = "Frag. $\\to$ Inund.",
    "Fragile -> Worse" = "Frag. $\\to$ Worse"
  )

  ame_table <- build_ame_table(ame_boot_combined, transition_order, term_labels)
  colnames(ame_table) <- c(
    "Covariate",
    unname(transition_headers[transition_order])
  )
  ame_table[is.na(ame_table)] <- ""

  diagnostic_rows <- build_diagnostic_rows(models, specs, transition_order)
  colnames(diagnostic_rows) <- colnames(ame_table)

  latex_lines <- c(
    "% Auto-generated by scripts/04_manuscript_transition_models.R",
    "\\begin{table}[!htbp]",
    "\\centering",
    "\\scriptsize",
    "\\setlength{\\tabcolsep}{4pt}",
    "\\caption{Average marginal effects for all transition probabilities}",
    "\\label{tab:ame_transition_probabilities}",
    paste0("\\begin{tabular}{l", paste(rep("c", length(transition_order)), collapse = ""), "}"),
    "\\hline",
    latex_row(colnames(ame_table)),
    "\\hline"
  )

  for (i in seq_len(nrow(ame_table))) {
    latex_lines <- c(
      latex_lines,
      latex_row(unlist(ame_table[i, ], use.names = FALSE))
    )
  }

  latex_lines <- c(latex_lines, "\\hline")

  for (i in seq_len(nrow(diagnostic_rows))) {
    latex_lines <- c(
      latex_lines,
      latex_row(unlist(diagnostic_rows[i, ], use.names = FALSE))
    )
  }

  latex_lines <- c(
    latex_lines,
    "\\hline",
    paste0(
      "\\multicolumn{", ncol(ame_table),
      "}{p{0.95\\linewidth}}{\\footnotesize Notes: Entries are average marginal effects. ",
      "Standard errors are from a ", AME_BOOT_REPS,
      "-replication cluster bootstrap by block group. County and SLR-scenario fixed effects are included in all models. ",
      "Significance stars are based on ",
      "bootstrapped p-values: * $p<0.05$, ** $p<0.01$, *** $p<0.001$. Abbreviations: W = Worse.}\\\\"
    ),
    "\\hline",
    "\\end{tabular}",
    "\\end{table}"
  )

  writeLines(latex_lines, con = AME_LATEX_PATH)
  message("Saved AME Excel results to: ", AME_EXCEL_PATH)
  message("Saved manuscript LaTeX table to: ", AME_LATEX_PATH)
}

analysis_dat <- read_analysis_data() %>%
  prepare_transition_data()

trans_dat <- analysis_dat %>%
  filter(slr_ft > 0)

redrisk_dat <- trans_dat %>%
  filter(baseline_redundant_n > 0) %>%
  mutate(
    prop_red_to_worse = any_loss_of_redundancy / baseline_redundant_n,
    prop_red_to_fragile = baseline_redundant_to_fragile / baseline_redundant_n,
    prop_red_to_isolated = baseline_redundant_to_isolated / baseline_redundant_n,
    prop_red_to_inundated = baseline_redundant_to_inundated / baseline_redundant_n
  )

fragrisk_dat <- trans_dat %>%
  filter(baseline_fragile_n > 0) %>%
  mutate(
    fragile_to_worse_n = baseline_fragile_to_isolated + baseline_fragile_to_inundated,
    prop_fragile_to_worse = fragile_to_worse_n / baseline_fragile_n,
    prop_fragile_to_isolated = baseline_fragile_to_isolated / baseline_fragile_n,
    prop_fragile_to_inundated = baseline_fragile_to_inundated / baseline_fragile_n
  )

model_specs <- make_model_specs(redrisk_dat, fragrisk_dat)
transition_models <- fit_model_specs(model_specs)
ame_boot_combined <- bootstrap_model_specs(transition_models, model_specs)
write_ame_outputs(ame_boot_combined, transition_models, model_specs)
