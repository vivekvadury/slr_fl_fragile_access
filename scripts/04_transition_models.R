# Manuscript transition models and Table 2 exports.
#
# This script estimates the grouped binomial transition models described in
# the manuscript and exports the cluster-bootstrap AME table used as Table 2.
#
# Bootstrap controls:
# - AME_BOOT_REPS sets successful bootstrap replications; default is 199.
# - AME_BOOT_SEED sets the base seed; default is 20260411.
# - AME_BOOT_MAX_ATTEMPTS sets the retry cap for failed bootstrap fits.
#
# Runtime controls:
# - BRIDGE_ARM selects approach, intersect, or retain; default is approach.
# - TRANSITION_DATA_PATH overrides the arm-tagged input dataset.
# - The input path may also be supplied positionally or with --data; --arm
#   overrides BRIDGE_ARM. Examples:
#     Rscript scripts/04_transition_models.R --arm retain
#     Rscript scripts/04_transition_models.R --arm retain --data path/to/data.csv

library(tidyverse)
library(fixest)
library(marginaleffects)
library(openxlsx)

VALID_ARMS <- c("intersect", "approach", "retain")

parse_runtime_options <- function(args = commandArgs(trailingOnly = TRUE)) {
  env_arm <- Sys.getenv("BRIDGE_ARM", unset = "")
  arm <- if (nzchar(env_arm)) env_arm else "approach"
  arm_is_explicit <- nzchar(env_arm)
  data_path <- Sys.getenv("TRANSITION_DATA_PATH", unset = "")
  cli_data_path <- ""
  positional <- character()

  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (identical(arg, "--arm")) {
      if (i == length(args)) {
        stop("--arm requires a value.")
      }
      i <- i + 1L
      arm <- args[[i]]
      arm_is_explicit <- TRUE
    } else if (startsWith(arg, "--arm=")) {
      arm <- substring(arg, nchar("--arm=") + 1L)
      arm_is_explicit <- TRUE
    } else if (identical(arg, "--data")) {
      if (i == length(args)) {
        stop("--data requires a value.")
      }
      i <- i + 1L
      cli_data_path <- args[[i]]
    } else if (startsWith(arg, "--data=")) {
      cli_data_path <- substring(arg, nchar("--data=") + 1L)
    } else if (startsWith(arg, "--")) {
      stop("Unknown command-line option: ", arg)
    } else {
      positional <- c(positional, arg)
    }
    i <- i + 1L
  }

  if (length(positional) > 1L) {
    stop("At most one positional dataset path may be supplied.")
  }
  if (length(positional) == 1L && nzchar(cli_data_path)) {
    stop("Supply the dataset path either positionally or with --data, not both.")
  }
  if (length(positional) == 1L) {
    cli_data_path <- positional[[1]]
  }
  if (nzchar(cli_data_path)) {
    data_path <- cli_data_path
  }

  if (nzchar(data_path)) {
    arm_match <- regexec(
      "block_group_analysis_dataset_(intersect|approach|retain)\\.csv$",
      basename(data_path)
    )
    arm_parts <- regmatches(basename(data_path), arm_match)[[1]]
    if (length(arm_parts) == 2L) {
      inferred_arm <- arm_parts[[2]]
      if (arm_is_explicit && !identical(arm, inferred_arm)) {
        stop(
          "Arm '", arm, "' conflicts with dataset filename arm '",
          inferred_arm, "'."
        )
      }
      if (!arm_is_explicit) {
        arm <- inferred_arm
      }
    }
  }

  if (!(arm %in% VALID_ARMS)) {
    stop(
      "Invalid arm '", arm, "'. Expected one of: ",
      paste(VALID_ARMS, collapse = ", "), "."
    )
  }
  if (!nzchar(data_path)) {
    data_path <- file.path(
      "data",
      "processed",
      "analysis",
      sprintf("block_group_analysis_dataset_%s.csv", arm)
    )
  }

  list(arm = arm, data_path = data_path)
}

RUN_OPTIONS <- parse_runtime_options()
ARM <- RUN_OPTIONS$arm
DATA_PATH <- RUN_OPTIONS$data_path
TABLE_DIR <- file.path("outputs", "tables")
AME_EXCEL_PATH <- file.path(
  TABLE_DIR,
  sprintf("ame_bootstrap_results_%s.xlsx", ARM)
)
AME_LATEX_PATH <- file.path(
  TABLE_DIR,
  sprintf("ame_bootstrap_transition_table_%s.tex", ARM)
)
SAMPLE_DIAGNOSTICS_PATH <- file.path(
  TABLE_DIR,
  sprintf("transition_sample_diagnostics_%s.csv", ARM)
)

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

STATE_COUNT_COLUMNS <- c(
  "block_centroid_unclassified",
  "block_centroid_inundated",
  "block_centroid_isolated",
  "block_centroid_fragile",
  "block_centroid_redundant"
)

TRANSITION_COUNT_COLUMNS <- c(
  "any_loss_of_redundancy",
  "baseline_redundant_to_fragile",
  "baseline_redundant_to_isolated",
  "baseline_redundant_to_inundated",
  "baseline_fragile_to_isolated",
  "baseline_fragile_to_inundated"
)

read_analysis_data <- function(path = DATA_PATH) {
  if (!file.exists(path)) {
    stop(
      "Analysis dataset does not exist: ", path,
      ". Run notebook 03 for arm '", ARM, "' first."
    )
  }
  message("Arm: ", ARM)
  message("Reading analysis dataset: ", path)
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

assert_eligible_state_partition <- function(dat) {
  required_columns <- c(
    "block_group_geoid",
    "slr_ft",
    "total_blocks",
    STATE_COUNT_COLUMNS,
    TRANSITION_COUNT_COLUMNS
  )
  missing_columns <- setdiff(required_columns, names(dat))
  if (length(missing_columns) > 0L) {
    stop(
      "The analysis dataset is missing required eligible-universe columns: ",
      paste(missing_columns, collapse = ", "), "."
    )
  }

  duplicate_keys <- dat %>%
    count(block_group_geoid, slr_ft, name = "n") %>%
    filter(n != 1L)
  if (nrow(duplicate_keys) > 0L) {
    stop(
      "Expected one row per (block_group_geoid, slr_ft); found ",
      nrow(duplicate_keys), " duplicate keys."
    )
  }

  state_matrix <- as.matrix(dat[, STATE_COUNT_COLUMNS, drop = FALSE])
  if (anyNA(state_matrix) || anyNA(dat$total_blocks)) {
    stop("Eligible-universe state counts and total_blocks must not be missing.")
  }
  if (
    any(!is.finite(state_matrix)) ||
      any(state_matrix < 0) ||
      any(state_matrix != floor(state_matrix))
  ) {
    stop("Eligible-universe state counts must be finite, nonnegative integers.")
  }

  eligible_risk_set_n <- rowSums(state_matrix)
  bad_partition <- which(eligible_risk_set_n != dat$total_blocks)
  if (length(bad_partition) > 0L) {
    example_rows <- head(bad_partition, 5L)
    examples <- tibble(
      key = paste0(
        dat$block_group_geoid[example_rows], "@",
        dat$slr_ft[example_rows], "ft"
      ),
      total_blocks = dat$total_blocks[example_rows],
      state_sum = eligible_risk_set_n[example_rows]
    )
    stop(
      "Five-state counts do not sum to the eligible risk set in ",
      length(bad_partition), " block-group/SLR rows. Examples: ",
      paste0(
        examples$key, " (total=", examples$total_blocks,
        ", states=", examples$state_sum, ")",
        collapse = "; "
      )
    )
  }

  message(
    "Eligible-universe state partition passed for ", nrow(dat),
    " block-group/SLR rows."
  )
  dat %>% mutate(eligible_risk_set_n = .env$eligible_risk_set_n)
}

make_filter_diagnostic <- function(data, keep, filter_name) {
  if (length(keep) != nrow(data) || anyNA(keep)) {
    stop("Invalid keep vector for sample diagnostic: ", filter_name)
  }
  dropped_block_groups <- data$block_group_geoid[!keep]
  retained_block_groups <- data$block_group_geoid[keep]
  tibble(
    arm = ARM,
    filter = filter_name,
    input_rows = nrow(data),
    input_block_groups = n_distinct(data$block_group_geoid),
    dropped_rows = sum(!keep),
    dropped_block_groups = n_distinct(dropped_block_groups),
    retained_rows = sum(keep),
    retained_block_groups = n_distinct(retained_block_groups)
  )
}

prepare_transition_data <- function(dat) {
  dat <- assert_eligible_state_partition(dat)

  base_counts <- dat %>%
    filter(slr_ft == 0) %>%
    transmute(
      block_group_geoid,
      baseline_total_blocks = total_blocks,
      baseline_eligible_risk_set_n = eligible_risk_set_n,
      baseline_unclassified_n = block_centroid_unclassified,
      baseline_redundant_n = block_centroid_redundant,
      baseline_fragile_n = block_centroid_fragile,
      baseline_isolated_n = block_centroid_isolated,
      baseline_inundated_n = block_centroid_inundated
    )

  prepared <- dat %>%
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
    )

  baseline_columns <- c(
    "baseline_total_blocks",
    "baseline_eligible_risk_set_n",
    "baseline_unclassified_n",
    "baseline_redundant_n",
    "baseline_fragile_n",
    "baseline_isolated_n",
    "baseline_inundated_n"
  )
  if (anyNA(prepared[, baseline_columns, drop = FALSE])) {
    stop("At least one block group lacks a unique 0-ft eligible baseline row.")
  }

  baseline_state_sum <- with(
    prepared,
    baseline_unclassified_n + baseline_inundated_n + baseline_isolated_n +
      baseline_fragile_n + baseline_redundant_n
  )
  bad_baseline_partition <- which(
    baseline_state_sum != prepared$baseline_eligible_risk_set_n |
      prepared$baseline_eligible_risk_set_n != prepared$baseline_total_blocks
  )
  if (length(bad_baseline_partition) > 0L) {
    stop(
      "Baseline five-state counts do not equal baseline_total_blocks in ",
      length(bad_baseline_partition), " block-group/SLR rows."
    )
  }

  unstable_universe <- which(
    prepared$eligible_risk_set_n != prepared$baseline_eligible_risk_set_n |
      prepared$total_blocks != prepared$baseline_total_blocks
  )
  if (length(unstable_universe) > 0L) {
    stop(
      "The eligible block risk set changes with SLR in ",
      length(unstable_universe), " block-group/SLR rows."
    )
  }

  transition_matrix <- as.matrix(
    prepared[, TRANSITION_COUNT_COLUMNS, drop = FALSE]
  )
  if (
    anyNA(transition_matrix) ||
      any(!is.finite(transition_matrix)) ||
      any(transition_matrix < 0) ||
      any(transition_matrix != floor(transition_matrix))
  ) {
    stop("Transition counts must be finite, nonnegative integers.")
  }
  redundant_event_sum <- with(
    prepared,
    baseline_redundant_to_fragile + baseline_redundant_to_isolated +
      baseline_redundant_to_inundated
  )
  fragile_event_sum <- with(
    prepared,
    baseline_fragile_to_isolated + baseline_fragile_to_inundated
  )
  bad_transition_risk_set <- which(
    redundant_event_sum != prepared$any_loss_of_redundancy |
      redundant_event_sum > prepared$baseline_redundant_n |
      fragile_event_sum > prepared$baseline_fragile_n
  )
  if (length(bad_transition_risk_set) > 0L) {
    stop(
      "Transition events do not fit their eligible baseline state risk sets in ",
      length(bad_transition_risk_set), " block-group/SLR rows."
    )
  }

  # The grouped-binomial weights below are explicit state-specific risk sets
  # inside the validated eligible universe, not total block counts.
  scaled <- prepared %>%
    mutate(
      across(
        all_of(CORE_COVARIATES),
        ~ as.numeric(scale(.x)),
        .names = "z_{.col}"
      )
    )

  complete_covariates <- complete.cases(
    scaled[, MODEL_COVARIATES, drop = FALSE]
  )
  covariate_diagnostic <- make_filter_diagnostic(
    scaled,
    complete_covariates,
    "drop_na(all_of(MODEL_COVARIATES))"
  )
  message(
    "Covariate completeness filter dropped ",
    covariate_diagnostic$dropped_block_groups, " block groups (",
    covariate_diagnostic$dropped_rows, " block-group/SLR rows)."
  )

  output <- scaled[complete_covariates, , drop = FALSE]
  attr(output, "covariate_filter_diagnostic") <- covariate_diagnostic
  output
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
    paste0(
      "% Auto-generated by scripts/04_transition_models.R for arm: ",
      ARM
    ),
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

analysis_dat <- read_analysis_data() %>% prepare_transition_data()
covariate_filter_diagnostic <- attr(
  analysis_dat,
  "covariate_filter_diagnostic"
)

trans_dat <- analysis_dat %>%
  filter(slr_ft > 0)

redundant_risk_keep <- trans_dat$baseline_redundant_n > 0
fragile_risk_keep <- trans_dat$baseline_fragile_n > 0

sample_diagnostics <- bind_rows(
  covariate_filter_diagnostic,
  make_filter_diagnostic(
    trans_dat,
    redundant_risk_keep,
    "baseline_redundant_n > 0"
  ),
  make_filter_diagnostic(
    trans_dat,
    fragile_risk_keep,
    "baseline_fragile_n > 0"
  )
)

dir.create(TABLE_DIR, showWarnings = FALSE, recursive = TRUE)
readr::write_csv(sample_diagnostics, SAMPLE_DIAGNOSTICS_PATH)
purrr::pwalk(
  sample_diagnostics,
  function(
      arm,
      filter,
      input_rows,
      input_block_groups,
      dropped_rows,
      dropped_block_groups,
      retained_rows,
      retained_block_groups
  ) {
    message(
      "[", arm, "] ", filter, ": dropped ", dropped_block_groups, " of ",
      input_block_groups, " block groups and ", dropped_rows, " of ",
      input_rows, " rows; retained ", retained_block_groups,
      " block groups and ", retained_rows, " rows."
    )
  }
)
message("Saved sample diagnostics to: ", SAMPLE_DIAGNOSTICS_PATH)

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
