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
# - MODEL_SPEC selects demographic_only (default) or with_physical.
# - CONLEY_CUTOFF_KM optionally adds Conley/HAC standard errors while retaining
#   the existing block-group-clustered and cluster-bootstrap standard errors.
# - AME_POP_WEIGHT, when set (1/true, or an explicit column name; the shorthand
#   resolves to eligible_pop20), additionally writes a population-weighted AME
#   table to *_popweighted.{xlsx,tex}. The model fit is unchanged; only the
#   avg_slopes() averaging is weighted so each block group counts in proportion
#   to its eligible population rather than once. It does not replace Table 2.
# - TRANSITION_DATA_PATH overrides the arm-tagged input dataset.
# - The input path may also be supplied positionally or with --data; --arm
#   overrides BRIDGE_ARM. Examples:
#     Rscript scripts/04_transition_models.R --arm retain
#     Rscript scripts/04_transition_models.R --arm retain --data path/to/data.csv

library(tidyverse)
library(fixest)
library(marginaleffects)
library(openxlsx)

script_file_argument <- grep(
  "^--file=", commandArgs(trailingOnly = FALSE), value = TRUE
)
SCRIPT_DIR <- if (length(script_file_argument) == 1L) {
  dirname(normalizePath(sub("^--file=", "", script_file_argument)))
} else {
  normalizePath("scripts")
}
source(file.path(SCRIPT_DIR, "04_shared_model_spec.R"))

VALID_ARMS <- c("intersect", "approach", "retain")

parse_runtime_options <- function(args = commandArgs(trailingOnly = TRUE)) {
  env_arm <- Sys.getenv("BRIDGE_ARM", unset = "")
  arm <- if (nzchar(env_arm)) env_arm else "approach"
  arm_is_explicit <- nzchar(env_arm)
  data_path <- Sys.getenv("TRANSITION_DATA_PATH", unset = "")
  cli_data_path <- ""
  positional <- character()
  model_spec <- Sys.getenv("MODEL_SPEC", unset = "demographic_only")
  conley_text <- Sys.getenv("CONLEY_CUTOFF_KM", unset = "")
  pop_weight_text <- Sys.getenv("AME_POP_WEIGHT", unset = "")

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
  model_spec <- validate_model_spec(model_spec)
  conley_cutoff_km <- NULL
  if (nzchar(conley_text)) {
    conley_cutoff_km <- suppressWarnings(as.numeric(conley_text))
    if (
      length(conley_cutoff_km) != 1L || is.na(conley_cutoff_km) ||
        !is.finite(conley_cutoff_km) || conley_cutoff_km <= 0
    ) {
      stop("CONLEY_CUTOFF_KM must be a positive finite number when set.")
    }
  }
  pop_weight_var <- NULL
  if (nzchar(pop_weight_text)) {
    pop_weight_var <- if (
      tolower(pop_weight_text) %in% c("1", "true", "yes", "on")
    ) {
      "eligible_pop20"
    } else {
      pop_weight_text
    }
  }
  if (!nzchar(data_path)) {
    data_path <- file.path(
      "data",
      "processed",
      "analysis",
      sprintf("block_group_analysis_dataset_%s.csv", arm)
    )
  }

  list(
    arm = arm,
    data_path = data_path,
    model_spec = model_spec,
    conley_cutoff_km = conley_cutoff_km,
    pop_weight_var = pop_weight_var
  )
}

RUN_OPTIONS <- parse_runtime_options()
ARM <- RUN_OPTIONS$arm
DATA_PATH <- RUN_OPTIONS$data_path
MODEL_SPEC <- RUN_OPTIONS$model_spec
CONLEY_CUTOFF_KM <- RUN_OPTIONS$conley_cutoff_km
POP_WEIGHT_VAR <- RUN_OPTIONS$pop_weight_var
TABLE_DIR <- file.path("outputs", "tables")
PHYSICAL_COVARIATES_PATH <- file.path(
  "data", "processed", "analysis", "block_group_physical_covariates.csv"
)
BLOCK_GROUP_GPKG_PATH <- file.path(
  "outputs", "spatial",
  sprintf("slr_block_group_analysis_%s.gpkg", ARM)
)

spec_output_name <- function(stem, extension, spec_name = MODEL_SPEC,
                             extra_tag = NULL) {
  spec_name <- validate_model_spec(spec_name)
  suffix <- if (identical(spec_name, "demographic_only")) {
    ARM
  } else {
    paste(ARM, spec_name, sep = "_")
  }
  if (!is.null(extra_tag) && nzchar(extra_tag)) {
    suffix <- paste(suffix, extra_tag, sep = "_")
  }
  sprintf("%s_%s.%s", stem, suffix, extension)
}

AME_EXCEL_PATH <- file.path(
  TABLE_DIR,
  spec_output_name("ame_bootstrap_results", "xlsx")
)
AME_LATEX_PATH <- file.path(
  TABLE_DIR,
  spec_output_name("ame_bootstrap_transition_table", "tex")
)
SAMPLE_DIAGNOSTICS_PATH <- file.path(
  TABLE_DIR,
  spec_output_name("transition_sample_diagnostics", "csv")
)
COEFFICIENT_DIAGNOSTICS_PATH <- file.path(
  TABLE_DIR,
  spec_output_name("transition_model_coefficients", "csv")
)
SPEC_COMPARISON_PATH <- file.path(
  TABLE_DIR,
  sprintf("transition_model_spec_comparison_%s.csv", ARM)
)

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
  message("Model specification: ", MODEL_SPEC)
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

attach_physical_covariates <- function(
    dat,
    path = PHYSICAL_COVARIATES_PATH
) {
  if (!file.exists(path)) {
    stop(
      "Physical-covariate file does not exist: ", path,
      ". Run scripts/03b_join_elevation_drainage.py first."
    )
  }
  message("Reading physical covariates: ", path)
  physical <- readr::read_csv(
    path,
    show_col_types = FALSE,
    col_types = cols(block_group_geoid = col_character())
  )
  required <- c(
    "block_group_geoid", "elevation_m_mean", "elevation_m_median",
    "drainage_distance_km"
  )
  missing <- setdiff(required, names(physical))
  if (length(missing) > 0L) {
    stop(
      "Physical-covariate file is missing required columns: ",
      paste(missing, collapse = ", "), "."
    )
  }
  physical <- physical %>% select(all_of(required))
  if (
    anyNA(physical$block_group_geoid) ||
      any(!nzchar(physical$block_group_geoid)) ||
      anyDuplicated(physical$block_group_geoid)
  ) {
    stop("Physical covariates must contain one nonmissing row per GEOID.")
  }
  numeric_columns <- setdiff(required, "block_group_geoid")
  if (
    any(!vapply(physical[numeric_columns], is.numeric, logical(1))) ||
      anyNA(physical[numeric_columns]) ||
      any(!is.finite(as.matrix(physical[numeric_columns])))
  ) {
    stop("Physical covariates must be numeric, finite, and nonmissing.")
  }
  input_rows <- nrow(dat)
  output <- dat %>% left_join(physical, by = "block_group_geoid")
  if (nrow(output) != input_rows) {
    stop("Physical-covariate join changed the analysis row count.")
  }
  unmatched <- output %>%
    filter(if_any(all_of(SPEC_WITH_PHYSICAL[7:8]), is.na)) %>%
    distinct(block_group_geoid) %>%
    pull(block_group_geoid)
  if (length(unmatched) > 0L) {
    stop(
      "Physical covariates are missing for ", length(unmatched),
      " analysis block groups. Examples: ",
      paste(head(unmatched, 20L), collapse = ", "), "."
    )
  }
  message(
    "Physical-covariate join passed for ",
    n_distinct(output$block_group_geoid), " block groups."
  )
  output
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

prepare_transition_data <- function(dat, spec_name = MODEL_SPEC) {
  spec_name <- validate_model_spec(spec_name)
  core_covariates <- if (identical(spec_name, "demographic_only")) {
    SPEC_DEMOGRAPHIC_ONLY
  } else {
    SPEC_WITH_PHYSICAL
  }
  model_covariates <- get_model_covariates(spec_name)
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
        all_of(core_covariates),
        ~ as.numeric(scale(.x)),
        .names = "z_{.col}"
      )
    )

  complete_covariates <- complete.cases(
    scaled[, model_covariates, drop = FALSE]
  )
  diagnostic_label <- if (identical(spec_name, "demographic_only")) {
    "drop_na(all_of(MODEL_COVARIATES))"
  } else {
    "drop_na(all_of(get_model_covariates(MODEL_SPEC)))"
  }
  covariate_diagnostic <- make_filter_diagnostic(
    scaled,
    complete_covariates,
    diagnostic_label
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
    spec_name = MODEL_SPEC,
    cluster = "block_group_geoid",
    reps = AME_BOOT_REPS,
    seed = AME_BOOT_SEED,
    max_attempts = AME_BOOT_MAX_ATTEMPTS,
    conf_level = 0.95,
    label = deparse(formula(model)[[2]]),
    pop_weight_var = NULL
) {
  # pop_weight_var == NULL reproduces the block-group-count-weighted AME exactly:
  # avg_slopes() is called with no wts argument, byte-for-byte the prior code
  # path. When set, each block group's contribution to the average marginal
  # effect is proportional to that column; the grouped-binomial fit is unchanged.
  if (!is.null(pop_weight_var)) {
    if (!pop_weight_var %in% names(data)) {
      stop(
        "Population-weight column '", pop_weight_var,
        "' is not present in the risk-set data for ", label, "."
      )
    }
    if (
      !is.numeric(data[[pop_weight_var]]) ||
        anyNA(data[[pop_weight_var]]) ||
        any(!is.finite(data[[pop_weight_var]])) ||
        any(data[[pop_weight_var]] < 0)
    ) {
      stop(
        "Population-weight column '", pop_weight_var,
        "' must be numeric, finite and nonnegative."
      )
    }
  }
  # When pop_weight_var is NULL the call is byte-for-byte the prior code path.
  # Otherwise the weight is supplied as a numeric vector aligned to the model's
  # retained rows (fixest::obs()), which does not depend on marginaleffects
  # being able to recover the column from the fitted object.
  average_slopes <- function(fitted, fit_data) {
    if (is.null(pop_weight_var)) {
      return(avg_slopes(fitted, vcov = FALSE))
    }
    row_weights <- fit_data[[pop_weight_var]][fixest::obs(fitted)]
    if (length(row_weights) != stats::nobs(fitted) || anyNA(row_weights)) {
      stop("Population weights did not align with the fitted model's rows.")
    }
    avg_slopes(fitted, vcov = FALSE, wts = row_weights)
  }

  point_estimates <- average_slopes(model, data) %>%
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

    boot_formula <- make_transition_formula(outcome, spec_name)
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
      suppressWarnings(average_slopes(boot_model, boot_data) %>% as_tibble()),
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

fit_model_specs <- function(specs) {
  purrr::map(
    specs,
    ~ fit_transition_model(
      .x$outcome, .x$data, .x$weights, spec_name = MODEL_SPEC
    )
  )
}

collect_coefficient_diagnostics <- function(
    models,
    conley_vcovs = NULL
) {
  output <- purrr::imap_dfr(
    models,
    function(model, transition) {
      coefficient_table <- as.data.frame(fixest::coeftable(model))
      if (ncol(coefficient_table) < 4L) {
        stop("Unexpected coefficient-table schema for transition: ", transition)
      }
      tibble(
        arm = ARM,
        transition = transition,
        term = rownames(coefficient_table),
        estimate = coefficient_table[[1]],
        std.error = coefficient_table[[2]],
        statistic = coefficient_table[[3]],
        p.value = coefficient_table[[4]]
      )
    }
  )
  if (!is.null(conley_vcovs)) {
    conley_rows <- purrr::imap_dfr(
      conley_vcovs,
      function(vcov_matrix, transition) {
        tibble(
          transition = transition,
          term = rownames(vcov_matrix),
          conley_se = sqrt(diag(vcov_matrix)),
          conley_cutoff_km = CONLEY_CUTOFF_KM
        )
      }
    )
    output <- output %>%
      left_join(conley_rows, by = c("transition", "term"))
    if (anyNA(output$conley_se)) {
      stop("Conley standard errors did not match all coefficient rows.")
    }
  }
  output
}

compute_conley_ame_table <- function(models, conley_vcovs) {
  previous_safety_option <- getOption("marginaleffects_safe")
  on.exit(options(marginaleffects_safe = previous_safety_option), add = TRUE)
  options(marginaleffects_safe = FALSE)
  purrr::imap_dfr(
    models,
    function(model, transition) {
      suppressWarnings(
        avg_slopes(model, vcov = conley_vcovs[[transition]]) %>%
          as_tibble()
      ) %>%
        transmute(
          transition = transition,
          term,
          conley_se = std.error,
          conley_cutoff_km = CONLEY_CUTOFF_KM
        )
    }
  )
}

bootstrap_model_specs <- function(models, specs, conley_vcovs = NULL,
                                  pop_weight_var = NULL) {
  output <- purrr::imap_dfr(
    models,
    function(model, transition) {
      transition_index <- match(transition, names(models))
      bootstrap_avg_slopes(
        model,
        specs[[transition]]$data,
        outcome = specs[[transition]]$outcome,
        weight_var = specs[[transition]]$weights,
        spec_name = MODEL_SPEC,
        label = transition,
        seed = AME_BOOT_SEED + transition_index,
        pop_weight_var = pop_weight_var
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
  if (!is.null(conley_vcovs)) {
    conley_ames <- compute_conley_ame_table(models, conley_vcovs)
    output <- output %>%
      left_join(conley_ames, by = c("transition", "term"))
    if (anyNA(output$conley_se)) {
      stop("Conley AME standard errors did not match all bootstrap AME rows.")
    }
  }
  output
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

write_ame_outputs <- function(ame_boot_combined, models, specs,
                              file_tag = NULL, caption_note = NULL) {
  dir.create(TABLE_DIR, showWarnings = FALSE, recursive = TRUE)
  excel_path <- if (is.null(file_tag)) {
    AME_EXCEL_PATH
  } else {
    file.path(
      TABLE_DIR,
      spec_output_name("ame_bootstrap_results", "xlsx", MODEL_SPEC, file_tag)
    )
  }
  latex_path <- if (is.null(file_tag)) {
    AME_LATEX_PATH
  } else {
    file.path(
      TABLE_DIR,
      spec_output_name(
        "ame_bootstrap_transition_table", "tex", MODEL_SPEC, file_tag
      )
    )
  }
  write.xlsx(ame_boot_combined, file = excel_path, overwrite = TRUE)

  term_labels <- c(
    z_pct_black_nh = "Black share (z)",
    z_pct_hispanic = "Hispanic share (z)",
    z_renter_share = "Renter share (z)",
    z_log_median_income = "Log median income (z)",
    z_pct_age_65plus = "Age 65+ share (z)",
    z_no_vehicle_share = "No-vehicle hh share (z)"
  )
  if (identical(MODEL_SPEC, "with_physical")) {
    term_labels <- c(
      term_labels,
      z_elevation_m_mean = "Mean elevation (z)",
      z_drainage_distance_km = "Distance to primary drainage (z)"
    )
  }

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
      if (is.null(caption_note)) "" else paste0(caption_note, " "),
      "Significance stars are based on ",
      "bootstrapped p-values: * $p<0.05$, ** $p<0.01$, *** $p<0.001$. Abbreviations: W = Worse.}\\\\"
    ),
    "\\hline",
    "\\end{tabular}",
    "\\end{table}"
  )

  writeLines(latex_lines, con = latex_path)
  message("Saved AME Excel results to: ", excel_path)
  message("Saved manuscript LaTeX table to: ", latex_path)
}

compare_specifications <- function(arm = ARM) {
  demographic_path <- file.path(
    TABLE_DIR,
    spec_output_name(
      "transition_model_coefficients", "csv", "demographic_only"
    )
  )
  physical_path <- file.path(
    TABLE_DIR,
    spec_output_name(
      "transition_model_coefficients", "csv", "with_physical"
    )
  )
  if (!file.exists(demographic_path) || !file.exists(physical_path)) {
    message(
      "Specification comparison not written: both coefficient files do ",
      "not yet exist for arm '", arm, "'."
    )
    return(invisible(NULL))
  }

  read_coefficients <- function(path, suffix) {
    input <- readr::read_csv(path, show_col_types = FALSE)
    required <- c("arm", "transition", "term", "estimate", "std.error")
    missing <- setdiff(required, names(input))
    if (length(missing) > 0L) {
      stop(
        "Coefficient comparison input is missing columns: ",
        paste(missing, collapse = ", "), "."
      )
    }
    if (anyDuplicated(input[, c("transition", "term")])) {
      stop("Coefficient comparison input has duplicate transition/term rows.")
    }
    optional <- intersect(
      c("conley_se", "conley_cutoff_km"), names(input)
    )
    input %>%
      select(transition, term, estimate, std.error, all_of(optional)) %>%
      rename_with(
        ~ paste0(.x, "_", suffix),
        -c(transition, term)
      )
  }

  demographic <- read_coefficients(demographic_path, "demographic_only")
  physical <- read_coefficients(physical_path, "with_physical")
  comparison <- full_join(
    demographic, physical, by = c("transition", "term")
  ) %>%
    mutate(
      arm = arm,
      racial_coefficient = term %in% c(
        "z_pct_black_nh", "z_pct_hispanic"
      ),
      percent_change_vs_demographic = if_else(
        racial_coefficient &
          !is.na(estimate_demographic_only) &
          estimate_demographic_only != 0 &
          !is.na(estimate_with_physical),
        100 * (
          estimate_with_physical - estimate_demographic_only
        ) / estimate_demographic_only,
        NA_real_
      )
    ) %>%
    select(arm, transition, term, everything()) %>%
    arrange(transition, term)

  readr::write_csv(comparison, SPEC_COMPARISON_PATH)
  message("Saved specification comparison to: ", SPEC_COMPARISON_PATH)
  invisible(comparison)
}

attach_conley_coordinates <- function(dat, gpkg_path = BLOCK_GROUP_GPKG_PATH) {
  message("Deriving block-group centroids for Conley covariance: ", gpkg_path)
  centroids <- load_block_group_centroids(gpkg_path)
  input_rows <- nrow(dat)
  output <- dat %>% left_join(centroids, by = "block_group_geoid")
  if (nrow(output) != input_rows) {
    stop("Centroid join changed the analysis row count.")
  }
  missing_coordinates <- output %>%
    filter(is.na(centroid_lat) | is.na(centroid_lon)) %>%
    distinct(block_group_geoid) %>%
    pull(block_group_geoid)
  if (length(missing_coordinates) > 0L) {
    stop(
      "Centroid coordinates are missing for ",
      length(missing_coordinates), " block groups. Examples: ",
      paste(head(missing_coordinates, 20L), collapse = ", "), "."
    )
  }
  output
}

raw_analysis_dat <- read_analysis_data()
if (identical(MODEL_SPEC, "with_physical")) {
  raw_analysis_dat <- attach_physical_covariates(raw_analysis_dat)
}
if (!is.null(CONLEY_CUTOFF_KM)) {
  raw_analysis_dat <- attach_conley_coordinates(raw_analysis_dat)
  message(
    "Conley/HAC standard errors enabled at cutoff ",
    CONLEY_CUTOFF_KM, " km; clustered SEs remain the primary columns."
  )
}
analysis_dat <- prepare_transition_data(raw_analysis_dat, MODEL_SPEC)
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
message("All seven transition models converged with complete coefficients.")
conley_vcovs <- NULL
if (!is.null(CONLEY_CUTOFF_KM)) {
  conley_vcovs <- purrr::imap(
    transition_models,
    ~ compute_conley_vcov(
      .x,
      model_specs[[.y]]$data,
      cutoff_km = CONLEY_CUTOFF_KM
    )
  )
}
coefficient_diagnostics <- collect_coefficient_diagnostics(
  transition_models,
  conley_vcovs
)
readr::write_csv(coefficient_diagnostics, COEFFICIENT_DIAGNOSTICS_PATH)
message("Saved model coefficient diagnostics to: ", COEFFICIENT_DIAGNOSTICS_PATH)
compare_specifications()
ame_boot_combined <- bootstrap_model_specs(
  transition_models,
  model_specs,
  conley_vcovs
)
write_ame_outputs(ame_boot_combined, transition_models, model_specs)

if (!is.null(POP_WEIGHT_VAR)) {
  # Additional, non-replacing table: the manuscript's Table 2 AMEs weight each
  # block group once; this variant weights each block group's contribution by
  # its total eligible population, so the reported average reflects residents
  # rather than block groups. The model fit is identical; only avg_slopes()
  # aggregation changes. Runs its own cluster bootstrap with the same seeds and
  # therefore the same resampled block groups as the table above.
  message(
    "Computing population-weighted AME table (avg_slopes wts = '",
    POP_WEIGHT_VAR, "'); grouped-binomial model fit unchanged."
  )
  ame_boot_popweighted <- bootstrap_model_specs(
    transition_models,
    model_specs,
    conley_vcovs = NULL,
    pop_weight_var = POP_WEIGHT_VAR
  )
  write_ame_outputs(
    ame_boot_popweighted,
    transition_models,
    model_specs,
    file_tag = "popweighted",
    caption_note = paste0(
      "Average marginal effects are population-weighted: each block group's ",
      "contribution is proportional to its eligible 2020 population ",
      "(\\texttt{", POP_WEIGHT_VAR, "}), so every resident counts once ",
      "regardless of their own characteristics. The grouped-binomial model ",
      "fit is unchanged (still weighted by baseline state counts)."
    )
  )
}
