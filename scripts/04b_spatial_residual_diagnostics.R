# Spatial residual diagnostics for the approach-arm transition models.
#
# This script sources model definitions from 04_shared_model_spec.R rather than
# sourcing 04_transition_models.R (which would also run its bootstrap and
# manuscript exports). It then tests Pearson residuals for spatial
# autocorrelation at each positive SLR scenario.
#
# The four redundant-risk models and three fragile-risk models have different
# estimation samples. One full block-group adjacency graph is constructed,
# then spdep's nb subset method induces one weights list for each risk set.
# No model residual is dropped, padded, or positionally aligned.
#
# Outputs (--spec demographic_only, the default):
#   outputs/run_comparison/moran_residual_diagnostics_approach.csv
#   outputs/run_comparison/moran_spatial_weights_diagnostics_approach.csv
#   outputs/run_comparison/moran_residual_diagnostics_approach.md
#   outputs/run_comparison/final_methods_verification.md (Sections 4-5 only)
#
# --spec with_physical writes spec-suffixed siblings
# (moran_residual_diagnostics_approach_with_physical.csv, ...) and never
# rewrites final_methods_verification.md. --spec both runs both specs and also
# writes outputs/run_comparison/moran_residual_diagnostics_spec_comparison.csv,
# a 42-row paired table with a column counting how many tests lose 0.05
# significance under the physical-covariate spec.
#
# The Pearson dispersion range quoted in report Sections 4-5 is computed live
# from this script's own fitted models, so it stays correct when the covariate
# set changes.
#
# Usage:
#   Rscript scripts/04b_spatial_residual_diagnostics.R
#   Rscript scripts/04b_spatial_residual_diagnostics.R --spec with_physical
#   Rscript scripts/04b_spatial_residual_diagnostics.R --spec both
#   Rscript scripts/04b_spatial_residual_diagnostics.R --no-update-report
#   Rscript scripts/04b_spatial_residual_diagnostics.R --help

resolve_diagnostic_script_path <- function() {
  command_args <- commandArgs(trailingOnly = FALSE)
  file_args <- grep("^--file=", command_args, value = TRUE)
  command_path <- if (length(file_args) > 0L) {
    sub("^--file=", "", file_args[[1L]])
  } else {
    character()
  }
  frame_paths <- unlist(lapply(sys.frames(), function(frame) {
    if (!is.null(frame$ofile)) frame$ofile else character()
  }), use.names = FALSE)
  candidates <- unique(c(
    command_path,
    rev(frame_paths),
    file.path(getwd(), "scripts", "04b_spatial_residual_diagnostics.R"),
    file.path(getwd(), "04b_spatial_residual_diagnostics.R")
  ))
  candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
  if (length(candidates) == 0L) {
    stop(
      "Could not locate 04b_spatial_residual_diagnostics.R to source its shared specification.",
      call. = FALSE
    )
  }
  normalizePath(candidates[[1L]], winslash = "/", mustWork = TRUE)
}

DIAGNOSTIC_SCRIPT_PATH <- resolve_diagnostic_script_path()
SHARED_MODEL_SPEC_PATH <- file.path(
  dirname(DIAGNOSTIC_SCRIPT_PATH), "04_shared_model_spec.R"
)
if (!file.exists(SHARED_MODEL_SPEC_PATH)) {
  stop(
    "Shared model specification does not exist: ", SHARED_MODEL_SPEC_PATH,
    call. = FALSE
  )
}
source(SHARED_MODEL_SPEC_PATH, local = FALSE, encoding = "UTF-8")

REQUIRED_PACKAGES <- c("fixest", "sf", "spdep")
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
SLR_SCENARIOS <- 1:6
SIGNIFICANCE_LEVEL <- 0.05
KNN_FALLBACK_K <- 6L
VALID_DIAGNOSTIC_SPECS <- c(VALID_MODEL_SPECS, "both")
DEFAULT_PHYSICAL_COVARIATES_PATH <- file.path(
  "data", "processed", "analysis", "block_group_physical_covariates.csv"
)

usage <- function() {
  cat(paste(
    "Usage: Rscript scripts/04b_spatial_residual_diagnostics.R [options]",
    "",
    "Options:",
    "  --data PATH             Approach-arm block-group analysis CSV.",
    "  --gpkg PATH             Approach-arm block-group GeoPackage.",
    "  --physical-covariates PATH",
    "                          Block-group elevation/drainage CSV.",
    "  --spec SPEC             demographic_only (default), with_physical,",
    "                          or both (paired 42-test comparison).",
    "  --output-dir PATH       Directory for CSV and Markdown outputs.",
    "  --report PATH           Default-spec verification report updated in place.",
    "  --no-update-report      Do not replace Sections 4-5 in the report.",
    "  -h, --help              Show this help message.",
    sep = "\n"
  ))
}

parse_options <- function(args = commandArgs(trailingOnly = TRUE)) {
  options <- list(
    data = file.path(
      "data", "processed", "analysis",
      "block_group_analysis_dataset_approach.csv"
    ),
    gpkg = file.path(
      "outputs", "spatial", "slr_block_group_analysis_approach.gpkg"
    ),
    physical_covariates = DEFAULT_PHYSICAL_COVARIATES_PATH,
    spec = "demographic_only",
    output_dir = file.path("outputs", "run_comparison"),
    report = file.path(
      "outputs", "run_comparison", "final_methods_verification.md"
    ),
    update_report = TRUE
  )

  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    } else if (identical(arg, "--no-update-report")) {
      options$update_report <- FALSE
    } else if (arg %in% c(
      "--data", "--gpkg", "--physical-covariates", "--spec",
      "--output-dir", "--report"
    )) {
      if (i == length(args)) {
        stop(arg, " requires a value.")
      }
      i <- i + 1L
      key <- sub("^--", "", arg)
      key <- gsub("-", "_", key, fixed = TRUE)
      options[[key]] <- args[[i]]
    } else if (grepl(
      "^--(data|gpkg|physical-covariates|spec|output-dir|report)=", arg
    )) {
      parts <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
      key <- gsub("-", "_", parts[[1]], fixed = TRUE)
      options[[key]] <- paste(parts[-1], collapse = "=")
    } else {
      stop("Unknown option: ", arg)
    }
    i <- i + 1L
  }
  if (!(options$spec %in% VALID_DIAGNOSTIC_SPECS)) {
    stop(
      "Invalid --spec value '", options$spec, "'. Expected one of: ",
      paste(VALID_DIAGNOSTIC_SPECS, collapse = ", "), ".",
      call. = FALSE
    )
  }
  options
}

check_dependencies <- function() {
  missing <- REQUIRED_PACKAGES[
    !vapply(REQUIRED_PACKAGES, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))
  ]
  if (length(missing) > 0L) {
    stop(
      "Missing required R package(s): ", paste(missing, collapse = ", "),
      ". Install them before rerunning; for example: ",
      "install.packages(c(",
      paste(sprintf("\"%s\"", missing), collapse = ", "), "))."
    )
  }
  message(
    "Resolved R/package versions: R ", getRversion(),
    "; fixest ", packageVersion("fixest"),
    "; sf ", packageVersion("sf"),
    "; spdep ", packageVersion("spdep")
  )
}

require_columns <- function(data, columns, label) {
  missing <- setdiff(columns, names(data))
  if (length(missing) > 0L) {
    stop(label, " is missing required columns: ", paste(missing, collapse = ", "))
  }
}

read_analysis_data <- function(path) {
  if (!file.exists(path)) {
    stop("Analysis dataset does not exist: ", path)
  }
  message("Reading approach-arm analysis data: ", normalizePath(path))
  data <- read.csv(
    path,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    colClasses = c(block_group_geoid = "character")
  )
  required <- unique(c(
    "block_group_geoid", "county_name", "slr_ft", "total_blocks",
    STATE_COUNT_COLUMNS, TRANSITION_COUNT_COLUMNS, SPEC_DEMOGRAPHIC_ONLY
  ))
  require_columns(data, required, "Analysis dataset")
  data$block_group_geoid <- as.character(data$block_group_geoid)
  if (
    anyNA(data$block_group_geoid) ||
      any(!nzchar(data$block_group_geoid))
  ) {
    stop("Analysis dataset has missing or empty block_group_geoid values.")
  }
  data
}

join_physical_covariates <- function(data, path) {
  if (!file.exists(path)) {
    stop(
      "Physical-covariate file does not exist: ", path,
      ". Run scripts/03b_join_elevation_drainage.py first."
    )
  }
  message("Reading block-group physical covariates: ", normalizePath(path))
  physical <- read.csv(
    path,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    colClasses = c(block_group_geoid = "character")
  )
  physical_columns <- setdiff(
    SPEC_WITH_PHYSICAL, SPEC_DEMOGRAPHIC_ONLY
  )
  require_columns(
    physical,
    c("block_group_geoid", physical_columns),
    "Physical-covariate file"
  )
  physical$block_group_geoid <- as.character(physical$block_group_geoid)
  if (
    anyNA(physical$block_group_geoid) ||
      any(!nzchar(physical$block_group_geoid)) ||
      anyDuplicated(physical$block_group_geoid)
  ) {
    stop(
      "Physical-covariate file must contain one row per nonmissing, nonempty block_group_geoid."
    )
  }
  overlapping_columns <- intersect(physical_columns, names(data))
  if (length(overlapping_columns) > 0L) {
    stop(
      "Analysis dataset already contains physical covariate column(s): ",
      paste(overlapping_columns, collapse = ", "),
      ". Refusing to overwrite them during the physical-covariate join."
    )
  }

  join_index <- match(data$block_group_geoid, physical$block_group_geoid)
  if (anyNA(join_index)) {
    missing_geoids <- unique(data$block_group_geoid[is.na(join_index)])
    stop(
      length(missing_geoids),
      " analysis block group(s) are absent from the physical-covariate file; examples: ",
      paste(head(missing_geoids, 20L), collapse = ", "), "."
    )
  }
  for (column in physical_columns) {
    if (!is.numeric(physical[[column]])) {
      stop("Physical covariate must be numeric: ", column, ".")
    }
    joined_values <- physical[[column]][join_index]
    if (anyNA(joined_values) || any(!is.finite(joined_values))) {
      bad_geoids <- unique(data$block_group_geoid[
        is.na(joined_values) | !is.finite(joined_values)
      ])
      stop(
        "Physical covariate '", column,
        "' is missing or non-finite for analysis block group(s); examples: ",
        paste(head(bad_geoids, 20L), collapse = ", "), "."
      )
    }
    data[[column]] <- joined_values
  }
  message(
    "Physical-covariate left join retained all ", nrow(data),
    " analysis rows in their original order."
  )
  data
}

prepare_transition_data <- function(data, spec_name) {
  spec_name <- validate_model_spec(spec_name)
  raw_covariates <- sub("^z_", "", get_model_covariates(spec_name))
  model_covariates <- get_model_covariates(spec_name)
  require_columns(data, raw_covariates, paste0("Analysis data for ", spec_name))

  state_matrix <- as.matrix(data[, STATE_COUNT_COLUMNS, drop = FALSE])
  if (
    anyNA(state_matrix) || any(!is.finite(state_matrix)) ||
      any(state_matrix < 0) || any(state_matrix != floor(state_matrix))
  ) {
    stop("Five-state counts must be finite, nonnegative integers.")
  }
  if (any(rowSums(state_matrix) != data$total_blocks)) {
    stop("Five-state counts do not equal total_blocks.")
  }

  baseline <- data[data$slr_ft == 0, c(
    "block_group_geoid", "total_blocks", STATE_COUNT_COLUMNS
  ), drop = FALSE]
  if (nrow(baseline) == 0L || anyDuplicated(baseline$block_group_geoid)) {
    stop("Expected exactly one 0-ft row per block group.")
  }
  baseline_index <- match(data$block_group_geoid, baseline$block_group_geoid)
  if (anyNA(baseline_index)) {
    stop("At least one block group lacks a 0-ft baseline row.")
  }

  data$baseline_total_blocks <- baseline$total_blocks[baseline_index]
  data$baseline_unclassified_n <- baseline$block_centroid_unclassified[baseline_index]
  data$baseline_redundant_n <- baseline$block_centroid_redundant[baseline_index]
  data$baseline_fragile_n <- baseline$block_centroid_fragile[baseline_index]
  data$baseline_isolated_n <- baseline$block_centroid_isolated[baseline_index]
  data$baseline_inundated_n <- baseline$block_centroid_inundated[baseline_index]
  data$baseline_eligible_risk_set_n <- with(
    data,
    baseline_unclassified_n + baseline_redundant_n + baseline_fragile_n +
      baseline_isolated_n + baseline_inundated_n
  )
  if (
    any(data$baseline_eligible_risk_set_n != data$baseline_total_blocks) ||
      any(data$total_blocks != data$baseline_total_blocks)
  ) {
    stop("Baseline state partition or stable eligible universe assertion failed.")
  }

  transition_matrix <- as.matrix(
    data[, TRANSITION_COUNT_COLUMNS, drop = FALSE]
  )
  if (
    anyNA(transition_matrix) || any(!is.finite(transition_matrix)) ||
      any(transition_matrix < 0) ||
      any(transition_matrix != floor(transition_matrix))
  ) {
    stop("Transition counts must be finite, nonnegative integers.")
  }
  redundant_event_sum <- with(
    data,
    baseline_redundant_to_fragile + baseline_redundant_to_isolated +
      baseline_redundant_to_inundated
  )
  fragile_event_sum <- with(
    data,
    baseline_fragile_to_isolated + baseline_fragile_to_inundated
  )
  if (
    any(redundant_event_sum != data$any_loss_of_redundancy) ||
      any(redundant_event_sum > data$baseline_redundant_n) ||
      any(fragile_event_sum > data$baseline_fragile_n)
  ) {
    stop("Transition counts do not fit their baseline-state risk sets.")
  }

  # This mirrors 04_transition_models.R: scale across the full 0-6 ft data,
  # then apply one complete-case filter for all covariates in the selected spec.
  for (i in seq_along(raw_covariates)) {
    data[[model_covariates[[i]]]] <- as.numeric(
      scale(data[[raw_covariates[[i]]]])
    )
  }
  data$slr_ft_f <- factor(data$slr_ft)
  complete_covariates <- complete.cases(
    data[, model_covariates, drop = FALSE]
  )
  message(
    "Complete-covariate filter for ", spec_name, ": retained ",
    sum(complete_covariates),
    " of ", nrow(data), " block-group/SLR rows."
  )
  data[complete_covariates, , drop = FALSE]
}

build_transition_risk_sets <- function(prepared) {
  transition_data <- prepared[prepared$slr_ft > 0, , drop = FALSE]

  redundant_data <- transition_data[
    transition_data$baseline_redundant_n > 0, , drop = FALSE
  ]
  redundant_data$prop_red_to_worse <- with(
    redundant_data, any_loss_of_redundancy / baseline_redundant_n
  )
  redundant_data$prop_red_to_fragile <- with(
    redundant_data, baseline_redundant_to_fragile / baseline_redundant_n
  )
  redundant_data$prop_red_to_isolated <- with(
    redundant_data, baseline_redundant_to_isolated / baseline_redundant_n
  )
  redundant_data$prop_red_to_inundated <- with(
    redundant_data, baseline_redundant_to_inundated / baseline_redundant_n
  )

  fragile_data <- transition_data[
    transition_data$baseline_fragile_n > 0, , drop = FALSE
  ]
  fragile_data$prop_fragile_to_worse <- with(
    fragile_data,
    (baseline_fragile_to_isolated + baseline_fragile_to_inundated) /
      baseline_fragile_n
  )
  fragile_data$prop_fragile_to_isolated <- with(
    fragile_data, baseline_fragile_to_isolated / baseline_fragile_n
  )
  fragile_data$prop_fragile_to_inundated <- with(
    fragile_data, baseline_fragile_to_inundated / baseline_fragile_n
  )

  list(redrisk_dat = redundant_data, fragrisk_dat = fragile_data)
}

fit_models_and_residuals <- function(specs, spec_name) {
  spec_name <- validate_model_spec(spec_name)
  models <- list()
  residual_rows <- list()
  for (transition in names(specs)) {
    spec <- specs[[transition]]
    model <- fit_transition_model(
      outcome = spec$outcome,
      data = spec$data,
      weight_var = spec$weights,
      spec_name = spec_name
    )
    observation_index <- fixest::obs(model)
    rows <- spec$data[
      observation_index, c("block_group_geoid", "slr_ft"), drop = FALSE
    ]
    rows$block_group_geoid <- as.character(rows$block_group_geoid)
    rows$pearson_residual <- as.numeric(
      stats::resid(model, type = "pearson")
    )
    if (nrow(rows) != length(rows$pearson_residual)) {
      stop("Residual-row length mismatch: ", transition)
    }
    models[[transition]] <- model
    residual_rows[[transition]] <- rows
  }
  list(models = models, residual_rows = residual_rows)
}

compute_pearson_dispersions <- function(models) {
  rows <- lapply(names(models), function(transition) {
    model <- models[[transition]]
    pearson_residuals <- as.numeric(stats::resid(model, type = "pearson"))
    residual_df <- as.numeric(
      fixest::degrees_freedom(model, type = "resid")
    )
    if (
      length(residual_df) != 1L || is.na(residual_df) ||
        !is.finite(residual_df) || residual_df <= 0
    ) {
      stop("Invalid residual degrees of freedom for ", transition, ".")
    }
    pearson_chisq <- sum(pearson_residuals^2)
    data.frame(
      transition = transition,
      observations = stats::nobs(model),
      residual_df = residual_df,
      pearson_chisq = pearson_chisq,
      dispersion = pearson_chisq / residual_df,
      stringsAsFactors = FALSE
    )
  })
  output <- do.call(rbind, rows)
  rownames(output) <- NULL
  if (
    nrow(output) != 7L || anyNA(output$dispersion) ||
      any(!is.finite(output$dispersion))
  ) {
    stop("Pearson-dispersion table failed its seven-model assertion.")
  }
  output
}

validate_family_supports <- function(specs, residual_rows) {
  supports <- list()
  diagnostics <- list()
  for (family in c("redundant", "fragile")) {
    family_models <- names(specs)[vapply(
      specs, function(x) identical(x$family, family), logical(1)
    )]
    expected <- NULL
    for (transition in family_models) {
      rows <- residual_rows[[transition]]
      scenario_sets <- lapply(SLR_SCENARIOS, function(slr_ft) {
        scenario_rows <- rows[rows$slr_ft == slr_ft, , drop = FALSE]
        if (anyDuplicated(scenario_rows$block_group_geoid)) {
          stop("Duplicate residual GEOIDs: ", transition, ", ", slr_ft, " ft.")
        }
        sort(unique(scenario_rows$block_group_geoid))
      })
      reference <- scenario_sets[[1]]
      mismatched <- SLR_SCENARIOS[!vapply(
        scenario_sets, function(x) identical(x, reference), logical(1)
      )]
      if (length(mismatched) > 0L) {
        stop(
          "Scenario support mismatch for ", transition, " at SLR levels: ",
          paste(mismatched, collapse = ", "), "."
        )
      }
      if (is.null(expected)) {
        expected <- reference
      } else if (!identical(reference, expected)) {
        stop("Model support mismatch within the ", family, "-risk family.")
      }
    }
    supports[[family]] <- expected
    diagnostics[[family]] <- data.frame(
      risk_family = family,
      n_models = length(family_models),
      n_scenarios = length(SLR_SCENARIOS),
      n_block_groups = length(expected),
      fixed_across_scenarios = TRUE,
      stringsAsFactors = FALSE
    )
    message(
      "Fixed ", family, "-risk support: ", length(expected),
      " block groups across all six scenarios."
    )
  }
  list(
    supports = supports,
    diagnostics = do.call(rbind, diagnostics)
  )
}

build_full_neighbours <- function(gpkg_path) {
  if (!file.exists(gpkg_path)) {
    stop("Spatial GeoPackage does not exist: ", gpkg_path)
  }
  message("Reading block-group geometry: ", normalizePath(gpkg_path))
  geometry <- sf::st_read(gpkg_path, layer = "slr_0ft", quiet = TRUE)
  require_columns(geometry, "block_group_geoid", "Spatial layer slr_0ft")
  geometry$block_group_geoid <- as.character(geometry$block_group_geoid)
  if (anyNA(geometry$block_group_geoid) || anyDuplicated(geometry$block_group_geoid)) {
    stop("Spatial layer has missing or duplicate block_group_geoid values.")
  }
  geometry <- geometry[order(geometry$block_group_geoid), ]
  geometry_ids <- geometry$block_group_geoid

  queen_nb <- spdep::poly2nb(
    geometry, queen = TRUE, row.names = geometry_ids
  )
  queen_islands <- sum(spdep::card(queen_nb) == 0L)
  queen_components <- spdep::n.comp.nb(queen_nb)$nc
  full_nb <- queen_nb
  method <- "queen"
  queen_listw_attempt <- tryCatch(
    spdep::nb2listw(full_nb, style = "W", zero.policy = FALSE),
    error = function(error) error
  )

  if (inherits(queen_listw_attempt, "error")) {
    projected <- sf::st_transform(geometry, 5070)
    points <- suppressWarnings(sf::st_point_on_surface(projected))
    coordinates <- sf::st_coordinates(points)
    full_nb <- spdep::knn2nb(
      spdep::knearneigh(coordinates, k = KNN_FALLBACK_K),
      row.names = geometry_ids,
      sym = TRUE
    )
    method <- paste0("symmetric_k", KNN_FALLBACK_K, "_fallback")
  }
  if (!identical(attr(full_nb, "region.id"), geometry_ids)) {
    stop("Full-neighbour graph GEOID order does not match the geometry order.")
  }
  message(
    "Full adjacency: ", length(full_nb), " units; method = ", method,
    "; queen islands = ", queen_islands, "."
  )
  list(
    nb = full_nb,
    ids = geometry_ids,
    method = method,
    queen_islands = queen_islands,
    queen_components = queen_components,
    full_islands = sum(spdep::card(full_nb) == 0L),
    full_components = spdep::n.comp.nb(full_nb)$nc
  )
}

build_risk_weights <- function(family, support_ids, full_graph) {
  missing_geometry <- setdiff(support_ids, full_graph$ids)
  if (length(missing_geometry) > 0L) {
    stop(
      family, "-risk residual GEOIDs absent from geometry: ",
      paste(head(missing_geometry, 20L), collapse = ", ")
    )
  }
  keep <- full_graph$ids %in% support_ids
  subset_nb <- subset(full_graph$nb, keep)
  subset_ids <- attr(subset_nb, "region.id")
  if (
    length(subset_ids) != length(support_ids) ||
      !setequal(subset_ids, support_ids)
  ) {
    stop(family, "-risk subset-neighbour GEOID mismatch.")
  }
  islands <- sum(spdep::card(subset_nb) == 0L)
  listw <- spdep::nb2listw(
    subset_nb, style = "W", zero.policy = TRUE
  )
  if (!identical(attr(listw$neighbours, "region.id"), subset_ids)) {
    stop(family, "-risk listw GEOID order mismatch.")
  }
  components <- spdep::n.comp.nb(subset_nb)$nc
  message(
    family, "-risk weights: ", length(subset_nb), " units; ", islands,
    " zero-neighbour islands; ", components, " connected components."
  )
  list(
    nb = subset_nb,
    listw = listw,
    ids = subset_ids,
    islands = islands,
    components = components
  )
}

run_moran_tests <- function(specs, residual_rows, weights) {
  results <- list()
  for (transition in names(specs)) {
    family <- specs[[transition]]$family
    family_weights <- weights[[family]]
    rows <- residual_rows[[transition]]
    for (slr_ft in SLR_SCENARIOS) {
      scenario_rows <- rows[
        rows$slr_ft == slr_ft,
        c("block_group_geoid", "pearson_residual"),
        drop = FALSE
      ]
      if (anyDuplicated(scenario_rows$block_group_geoid)) {
        stop("Duplicate residual GEOID: ", transition, ", ", slr_ft, " ft.")
      }

      # The merge is a key check, not an invitation to positional alignment.
      joined <- merge(
        data.frame(
          block_group_geoid = family_weights$ids,
          stringsAsFactors = FALSE
        ),
        scenario_rows,
        by = "block_group_geoid",
        all = TRUE,
        sort = FALSE
      )
      if (nrow(joined) != length(family_weights$ids)) {
        stop(
          "Joined row mismatch for ", transition, " at ", slr_ft,
          " ft: expected ", length(family_weights$ids),
          ", got ", nrow(joined), "."
        )
      }
      if (anyNA(joined$pearson_residual)) {
        stop(
          "Missing residuals after GEOID join for ", transition, " at ",
          slr_ft, " ft: ", sum(is.na(joined$pearson_residual)), "."
        )
      }
      residual_only <- setdiff(
        joined$block_group_geoid, family_weights$ids
      )
      if (length(residual_only) > 0L) {
        stop(
          "Residual-only GEOIDs for ", transition, " at ", slr_ft,
          " ft: ", paste(head(residual_only, 20L), collapse = ", ")
        )
      }
      order_index <- match(
        family_weights$ids, joined$block_group_geoid
      )
      if (anyNA(order_index)) {
        stop("At least one weights GEOID is absent after the residual join.")
      }
      joined <- joined[order_index, , drop = FALSE]
      if (!identical(joined$block_group_geoid, family_weights$ids)) {
        stop("Final residual/listw GEOID order mismatch.")
      }

      test <- spdep::moran.test(
        joined$pearson_residual,
        family_weights$listw,
        zero.policy = TRUE
      )
      results[[length(results) + 1L]] <- data.frame(
        transition = transition,
        risk_family = family,
        slr_ft = slr_ft,
        n = nrow(joined),
        moran_i = unname(test$estimate[["Moran I statistic"]]),
        expectation = unname(test$estimate[["Expectation"]]),
        variance = unname(test$estimate[["Variance"]]),
        p_value = test$p.value,
        alternative = test$alternative,
        significant_0_05 = test$p.value < SIGNIFICANCE_LEVEL,
        stringsAsFactors = FALSE
      )
    }
  }
  output <- do.call(rbind, results)
  rownames(output) <- NULL
  if (
    nrow(output) != 42L ||
      anyDuplicated(output[, c("transition", "slr_ft")]) ||
      anyNA(output$moran_i) || anyNA(output$p_value) ||
      any(output$p_value < 0 | output$p_value > 1)
  ) {
    stop("The completed Moran result table failed its 42-test assertion.")
  }
  output
}

format_integer <- function(value) {
  format(value, big.mark = ",", scientific = FALSE, trim = TRUE)
}

format_p_value <- function(value) {
  if (identical(value, 0)) {
    return("0 (underflow)")
  }
  if (value < 0.001) {
    return(formatC(value, format = "e", digits = 3))
  }
  sprintf("%.6f", value)
}

markdown_table <- function(results) {
  lines <- c(
    "| Transition/outcome | Risk family | SLR scenario | N | Moran's I | p-value | Significant at 0.05 |",
    "|---|---|---:|---:|---:|---:|---|"
  )
  for (i in seq_len(nrow(results))) {
    transition <- gsub(" -> ", " → ", results$transition[[i]], fixed = TRUE)
    family <- paste0(
      toupper(substr(results$risk_family[[i]], 1L, 1L)),
      substring(results$risk_family[[i]], 2L)
    )
    lines <- c(lines, paste0(
      "| ", transition,
      " | ", family,
      " | ", results$slr_ft[[i]], " ft",
      " | ", format_integer(results$n[[i]]),
      " | ", sprintf("%.6f", results$moran_i[[i]]),
      " | ", format_p_value(results$p_value[[i]]),
      " | ", if (results$significant_0_05[[i]]) "Yes" else "No",
      " |"
    ))
  }
  lines
}

describe_nonsignificant_tests <- function(results) {
  nonsignificant <- results[!results$significant_0_05, , drop = FALSE]
  if (nrow(nonsignificant) == 0L) {
    return("No test was nonsignificant at 0.05.")
  }
  labels <- paste0(
    nonsignificant$transition, " at ", nonsignificant$slr_ft, " ft"
  )
  paste0(
    "The nonsignificant result",
    if (length(labels) == 1L) " is " else "s are ",
    paste(labels, collapse = "; "), "."
  )
}

make_report_sections <- function(
    results,
    weights,
    full_graph,
    dispersions,
    spec_name
) {
  spec_name <- validate_model_spec(spec_name)
  n_significant <- sum(results$significant_0_05)
  family_significant <- tapply(
    results$significant_0_05, results$risk_family, sum
  )
  family_total <- table(results$risk_family)
  dispersion_min <- min(dispersions$dispersion)
  dispersion_max <- max(dispersions$dispersion)
  dispersion_range <- paste0(
    sprintf("%.3f", dispersion_min), "\u2013", sprintf("%.3f", dispersion_max)
  )
  dispersion_text <- if (all(dispersions$dispersion > 1)) {
    paste0(
      "All seven grouped-binomial fits are overdispersed; live Pearson ",
      "dispersion values range from ", dispersion_range,
      " (`sum(Pearson residual^2) / residual df`)."
    )
  } else {
    paste0(
      "Live Pearson dispersion across the seven grouped-binomial fits ranges ",
      "from ", dispersion_range,
      " (`sum(Pearson residual^2) / residual df`)."
    )
  }
  specification_text <- if (identical(spec_name, "demographic_only")) {
    paste0(
      "the unchanged demographic-only specification (six standardized ",
      "demographic covariates)"
    )
  } else {
    paste0(
      "the explicitly alternate with-physical specification (the same six ",
      "standardized demographic covariates plus standardized mean elevation ",
      "and drainage distance)"
    )
  }
  section_title_suffix <- if (identical(spec_name, "demographic_only")) {
    "resolved"
  } else {
    "with physical covariates"
  }

  environment_line <- paste0(
    "This diagnostic ran entirely locally in R ", getRversion(),
    " with `spdep` ", packageVersion("spdep"), ", `sf` ",
    packageVersion("sf"), ", and `fixest` ", packageVersion("fixest"),
    "; neither `dplyr` nor `tidyr` was used."
  )
  full_method_text <- if (identical(full_graph$method, "queen")) {
    "Queen contiguity produced a usable full weights graph."
  } else {
    paste0(
      "The full ", format_integer(length(full_graph$ids)),
      "-unit queen graph had ", full_graph$queen_islands,
      " zero-neighbor unit, so the prescribed symmetric k = ",
      KNN_FALLBACK_K, " fallback was used once for the full adjacency."
    )
  }

  section4 <- c(
    paste0(
      "## 4. Step 2b \u2014 Moran's I of approach-arm Pearson residuals (",
      section_title_suffix, ")"
    ),
    "",
    environment_line,
    "",
    paste0(
      "The seven approach-arm models were refit with ", specification_text, ". ",
      "The redundant-risk estimation sample contained the same ",
      format_integer(length(weights$redundant$ids)),
      " block groups at every SLR level for all four redundant-risk outcomes; ",
      "the fragile-risk sample contained the same ",
      format_integer(length(weights$fragile$ids)),
      " block groups at every level for all three fragile-risk outcomes. ",
      full_method_text, " The two estimation-sample graphs were then induced ",
      "with `subset.nb()` from that full topology rather than rebuilding ",
      "neighbors from subset geometries. Each Pearson-residual vector was ",
      "joined by `block_group_geoid` and explicitly reordered to its risk-set ",
      "`listw` GEOID order before testing."
    ),
    "",
    dispersion_text,
    "",
    paste0(
      "**Zero-neighbor islands after risk-set subsetting: redundant-risk = ",
      weights$redundant$islands, "; fragile-risk = ", weights$fragile$islands,
      ".** Both lists were nevertheless constructed and tested with ",
      "`zero.policy = TRUE` as specified. The subset graphs contained ",
      weights$redundant$components, " and ", weights$fragile$components,
      " connected components, respectively."
    ),
    "",
    paste0(
      "The tests use the default one-sided `greater` alternative for positive ",
      "spatial autocorrelation. “0 (underflow)” means R returned a numerical ",
      "p-value of exactly zero at double precision."
    ),
    "",
    markdown_table(results),
    "",
    paste0(
      format_integer(n_significant), " of the 42 tests reject spatial ",
      "randomness at 0.05: ", family_significant[["redundant"]], " of ",
      family_total[["redundant"]], " in the redundant-risk family and ",
      family_significant[["fragile"]], " of ", family_total[["fragile"]],
      " in the fragile-risk family."
    ),
    describe_nonsignificant_tests(results)
  )

  if (n_significant > 0L) {
    verdict_text <- paste0(
      "**For the `", spec_name,
      "` specification, `vcov = block_group_geoid` plus block-group ",
      "cluster-bootstrap uncertainty approach is not sufficient on its own ",
      "for a defensible Methods section.** Pearson dispersion across the seven ",
      "current fits ranges from ", dispersion_range, ", and ",
      n_significant, " of 42 Moran tests show significant positive spatial ",
      "autocorrelation, including ", family_significant[["redundant"]], " of ",
      family_total[["redundant"]], " redundant-risk tests and ",
      family_significant[["fragile"]], " of ", family_total[["fragile"]],
      " fragile-risk tests. Clustering by block group handles repeated ",
      "observations of the same unit across SLR scenarios, but it does not ",
      "address dependence between neighboring block groups. The point ",
      "specification need not change, but the reported uncertainty must be ",
      "supplemented with a spatially robust procedure before the inferential ",
      "claims are defensible; documenting spatial dependence only as a ",
      "limitation is not enough."
    )
  } else {
    verdict_text <- paste0(
      "**No Moran test detected significant positive spatial autocorrelation.** ",
      "The block-group-clustered variance and cluster bootstrap therefore do ",
      "not need a spatially robust supplement on the evidence of this ",
      "diagnostic. Pearson dispersion across the seven current fits ranges ",
      "from ", dispersion_range, " and must still be reported and addressed ",
      "in the uncertainty discussion."
    )
  }
  verdict <- c("## 5. Verdict", "", verdict_text)
  list(section4 = section4, verdict = verdict)
}

replace_report_sections <- function(path, sections) {
  if (!file.exists(path)) {
    stop("Verification report does not exist: ", path)
  }
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  section4_start <- grep("^## 4\\. Step 2b", lines)
  section5_start <- grep("^## 5\\. Verdict$", lines)
  if (length(section4_start) != 1L || length(section5_start) != 1L) {
    stop("Expected exactly one Step 2b section and one Section 5 verdict.")
  }
  if (section4_start >= section5_start) {
    stop("Verification report section order is invalid.")
  }
  later_headings <- grep("^## [0-9]+\\.", lines)
  later_headings <- later_headings[later_headings > section5_start]
  suffix <- if (length(later_headings) > 0L) {
    lines[min(later_headings):length(lines)]
  } else {
    character()
  }
  prefix <- lines[seq_len(section4_start - 1L)]
  output <- c(
    prefix,
    sections$section4,
    "",
    sections$verdict,
    if (length(suffix) > 0L) c("", suffix) else character()
  )
  writeLines(output, path, useBytes = TRUE)
  message("Updated verification report Sections 4-5: ", normalizePath(path))
}

# The default demographic-only spec keeps the historical un-suffixed filenames
# and owns Sections 4-5 of the verification report. The explicitly alternate
# with_physical spec writes spec-suffixed siblings and never rewrites the
# report, so a --spec with_physical run can never silently displace the
# canonical demographic-only diagnostic.
spec_diagnostic_path <- function(output_dir, stem, extension, spec_name) {
  spec_name <- validate_model_spec(spec_name)
  suffix <- if (identical(spec_name, "demographic_only")) {
    "approach"
  } else {
    paste0("approach_", spec_name)
  }
  file.path(output_dir, sprintf("%s_%s.%s", stem, suffix, extension))
}

write_outputs <- function(
    results,
    support_diagnostics,
    weights,
    full_graph,
    sections,
    options,
    spec_name,
    update_report
) {
  spec_name <- validate_model_spec(spec_name)
  dir.create(options$output_dir, recursive = TRUE, showWarnings = FALSE)
  results_path <- spec_diagnostic_path(
    options$output_dir, "moran_residual_diagnostics", "csv", spec_name
  )
  weights_path <- spec_diagnostic_path(
    options$output_dir, "moran_spatial_weights_diagnostics", "csv", spec_name
  )
  markdown_path <- spec_diagnostic_path(
    options$output_dir, "moran_residual_diagnostics", "md", spec_name
  )

  write.csv(results, results_path, row.names = FALSE, quote = TRUE)
  weights_diagnostics <- support_diagnostics
  weights_diagnostics$weights_method <- full_graph$method
  weights_diagnostics$full_graph_units <- length(full_graph$ids)
  weights_diagnostics$full_queen_islands <- full_graph$queen_islands
  weights_diagnostics$full_graph_islands <- full_graph$full_islands
  weights_diagnostics$subset_islands <- c(
    weights$redundant$islands, weights$fragile$islands
  )
  weights_diagnostics$subset_components <- c(
    weights$redundant$components, weights$fragile$components
  )
  write.csv(weights_diagnostics, weights_path, row.names = FALSE, quote = TRUE)
  writeLines(c(sections$section4, "", sections$verdict), markdown_path, useBytes = TRUE)

  message("[", spec_name, "] Saved 42-test CSV: ", normalizePath(results_path))
  message("[", spec_name, "] Saved weights diagnostics: ", normalizePath(weights_path))
  message("[", spec_name, "] Saved Markdown table and verdict: ", normalizePath(markdown_path))
  if (isTRUE(update_report)) {
    replace_report_sections(options$report, sections)
  } else if (identical(spec_name, "demographic_only")) {
    message("Skipped verification-report update (--no-update-report).")
  } else {
    message(
      "Verification-report Sections 4-5 describe the demographic-only ",
      "specification and are left unchanged by the ", spec_name, " run."
    )
  }
}

write_spec_comparison <- function(
    demographic_results,
    physical_results,
    output_dir
) {
  key_columns <- c("transition", "slr_ft")
  select_spec <- function(results, spec_name) {
    keep <- results[, c(
      key_columns, "risk_family", "n", "moran_i", "p_value", "significant_0_05"
    )]
    renamed <- setdiff(names(keep), key_columns)
    names(keep)[match(renamed, names(keep))] <- paste0(renamed, "_", spec_name)
    keep
  }
  comparison <- merge(
    select_spec(demographic_results, "demographic_only"),
    select_spec(physical_results, "with_physical"),
    by = key_columns,
    all = TRUE,
    sort = FALSE
  )
  if (
    nrow(comparison) != 42L ||
      anyNA(comparison$moran_i_demographic_only) ||
      anyNA(comparison$moran_i_with_physical) ||
      !identical(
        comparison$risk_family_demographic_only,
        comparison$risk_family_with_physical
      ) ||
      !identical(comparison$n_demographic_only, comparison$n_with_physical)
  ) {
    stop("Paired spec comparison failed its 42-test alignment assertion.")
  }
  comparison$risk_family <- comparison$risk_family_demographic_only
  comparison$n <- comparison$n_demographic_only
  for (dropped in c(
    "risk_family_demographic_only", "risk_family_with_physical",
    "n_demographic_only", "n_with_physical"
  )) {
    comparison[[dropped]] <- NULL
  }

  comparison$moran_i_change_with_physical <-
    comparison$moran_i_with_physical - comparison$moran_i_demographic_only
  comparison$lost_significance_under_physical <-
    comparison$significant_0_05_demographic_only &
      !comparison$significant_0_05_with_physical
  comparison$gained_significance_under_physical <-
    !comparison$significant_0_05_demographic_only &
      comparison$significant_0_05_with_physical
  n_lost <- sum(comparison$lost_significance_under_physical)
  n_gained <- sum(comparison$gained_significance_under_physical)
  comparison$n_tests_losing_significance_under_physical <- n_lost
  comparison$n_tests_gaining_significance_under_physical <- n_gained

  ordered_columns <- c(
    "transition", "risk_family", "slr_ft", "n",
    "moran_i_demographic_only", "moran_i_with_physical",
    "moran_i_change_with_physical",
    "p_value_demographic_only", "p_value_with_physical",
    "significant_0_05_demographic_only", "significant_0_05_with_physical",
    "lost_significance_under_physical", "gained_significance_under_physical",
    "n_tests_losing_significance_under_physical",
    "n_tests_gaining_significance_under_physical"
  )
  comparison <- comparison[
    order(comparison$risk_family, comparison$transition, comparison$slr_ft),
    ordered_columns,
    drop = FALSE
  ]
  rownames(comparison) <- NULL

  path <- file.path(
    output_dir, "moran_residual_diagnostics_spec_comparison.csv"
  )
  write.csv(comparison, path, row.names = FALSE, quote = TRUE)
  message(sprintf(
    paste0(
      "Saved paired Moran comparison (%d tests; %d lose and %d gain ",
      "significance at 0.05 under with_physical): %s"
    ),
    nrow(comparison), n_lost, n_gained, normalizePath(path)
  ))
  comparison
}

run_single_spec <- function(spec_name, raw_data, full_graph, options) {
  spec_name <- validate_model_spec(spec_name)
  message("\n=== Spatial residual diagnostic: ", spec_name, " specification ===")

  data <- raw_data
  if (identical(spec_name, "with_physical")) {
    data <- join_physical_covariates(data, options$physical_covariates)
  }
  prepared <- prepare_transition_data(data, spec_name)
  risk_sets <- build_transition_risk_sets(prepared)
  specs <- make_model_specs(risk_sets$redrisk_dat, risk_sets$fragrisk_dat)
  fitted <- fit_models_and_residuals(specs, spec_name)
  dispersions <- compute_pearson_dispersions(fitted$models)
  support_check <- validate_family_supports(specs, fitted$residual_rows)
  weights <- list(
    redundant = build_risk_weights(
      "redundant", support_check$supports$redundant, full_graph
    ),
    fragile = build_risk_weights(
      "fragile", support_check$supports$fragile, full_graph
    )
  )
  results <- run_moran_tests(specs, fitted$residual_rows, weights)
  sections <- make_report_sections(
    results, weights, full_graph, dispersions, spec_name
  )

  list(
    spec_name = spec_name,
    results = results,
    dispersions = dispersions,
    support_diagnostics = support_check$diagnostics,
    weights = weights,
    sections = sections
  )
}

report_spec_console_summary <- function(run) {
  results <- run$results
  weights <- run$weights
  family_counts <- tapply(results$significant_0_05, results$risk_family, sum)
  dispersion_range <- sprintf(
    "%.3f-%.3f",
    min(run$dispersions$dispersion), max(run$dispersions$dispersion)
  )
  cat("\n[", run$spec_name, "] Spatial residual diagnostic complete.\n", sep = "")
  cat("[", run$spec_name, "] Tests: 42\n", sep = "")
  cat(
    "[", run$spec_name, "] Significant at 0.05: ",
    sum(results$significant_0_05), "\n", sep = ""
  )
  cat(
    "[", run$spec_name, "] Redundant-risk significant: ",
    family_counts[["redundant"]], "/24\n", sep = ""
  )
  cat(
    "[", run$spec_name, "] Fragile-risk significant: ",
    family_counts[["fragile"]], "/18\n", sep = ""
  )
  cat(
    "[", run$spec_name, "] Live Pearson dispersion range: ",
    dispersion_range, "\n", sep = ""
  )
  cat(
    "[", run$spec_name, "] Zero-neighbour islands (redundant/fragile): ",
    weights$redundant$islands, "/", weights$fragile$islands, "\n", sep = ""
  )
}

main <- function() {
  options <- parse_options()
  check_dependencies()
  raw_data <- read_analysis_data(options$data)
  # The block-group adjacency topology does not depend on the model
  # specification, so it is built once and reused for every spec.
  full_graph <- build_full_neighbours(options$gpkg)

  specs_to_run <- if (identical(options$spec, "both")) {
    c("demographic_only", "with_physical")
  } else {
    options$spec
  }

  runs <- list()
  for (spec_name in specs_to_run) {
    run <- run_single_spec(spec_name, raw_data, full_graph, options)
    update_report <- isTRUE(options$update_report) &&
      identical(spec_name, "demographic_only")
    write_outputs(
      run$results,
      run$support_diagnostics,
      run$weights,
      full_graph,
      run$sections,
      options,
      spec_name,
      update_report
    )
    report_spec_console_summary(run)
    runs[[spec_name]] <- run
  }

  if (identical(options$spec, "both")) {
    comparison <- write_spec_comparison(
      runs[["demographic_only"]]$results,
      runs[["with_physical"]]$results,
      options$output_dir
    )
    cat(
      "\n[both] Paired Moran comparison: ",
      sum(comparison$lost_significance_under_physical),
      " of 42 tests lose significance at 0.05 under with_physical; ",
      sum(comparison$gained_significance_under_physical), " gain it.\n",
      sep = ""
    )
  }
}

main()
