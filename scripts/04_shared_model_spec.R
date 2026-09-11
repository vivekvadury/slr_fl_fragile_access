# Shared transition-model specifications and fitting utilities.
#
# This file contains definitions only. It is safe to source from both
# 04_transition_models.R and 04b_spatial_residual_diagnostics.R without
# triggering model fits, bootstraps, diagnostics, or output writes.

SPEC_DEMOGRAPHIC_ONLY <- c(
  "pct_black_nh",
  "pct_hispanic",
  "renter_share",
  "log_median_income",
  "pct_age_65plus",
  "no_vehicle_share"
)

SPEC_WITH_PHYSICAL <- c(
  SPEC_DEMOGRAPHIC_ONLY,
  "elevation_m_mean",
  "drainage_distance_km"
)

VALID_MODEL_SPECS <- c("demographic_only", "with_physical")

CONLEY_LATITUDE_COLUMN <- "centroid_lat"
CONLEY_LONGITUDE_COLUMN <- "centroid_lon"

validate_model_spec <- function(spec_name) {
  if (
    length(spec_name) != 1L ||
      is.na(spec_name) ||
      !is.character(spec_name) ||
      !(spec_name %in% VALID_MODEL_SPECS)
  ) {
    stop(
      "Invalid model specification '", paste(spec_name, collapse = ", "),
      "'. Expected one of: ", paste(VALID_MODEL_SPECS, collapse = ", "), ".",
      call. = FALSE
    )
  }
  spec_name
}

get_model_covariates <- function(spec_name) {
  spec_name <- validate_model_spec(spec_name)
  raw_covariates <- switch(
    spec_name,
    demographic_only = SPEC_DEMOGRAPHIC_ONLY,
    with_physical = SPEC_WITH_PHYSICAL
  )
  paste0("z_", raw_covariates)
}

make_transition_formula <- function(outcome, spec_name) {
  if (
    length(outcome) != 1L ||
      is.na(outcome) ||
      !is.character(outcome) ||
      !nzchar(outcome)
  ) {
    stop("outcome must be one non-empty column name.", call. = FALSE)
  }
  model_rhs <- paste(get_model_covariates(spec_name), collapse = " + ")
  stats::as.formula(
    paste0(outcome, " ~ ", model_rhs, " | county_name + slr_ft_f"),
    env = parent.frame()
  )
}

fit_transition_model <- function(
    outcome,
    data,
    weight_var,
    spec_name
) {
  spec_name <- validate_model_spec(spec_name)
  if (!is.data.frame(data)) {
    stop("data must be a data frame.", call. = FALSE)
  }
  if (
    length(weight_var) != 1L ||
      is.na(weight_var) ||
      !is.character(weight_var) ||
      !nzchar(weight_var)
  ) {
    stop("weight_var must be one non-empty column name.", call. = FALSE)
  }

  model_covariates <- get_model_covariates(spec_name)
  required_columns <- unique(c(
    outcome,
    weight_var,
    model_covariates,
    "county_name",
    "slr_ft_f",
    "block_group_geoid"
  ))
  missing_columns <- setdiff(required_columns, names(data))
  if (length(missing_columns) > 0L) {
    stop(
      "Cannot fit ", outcome, " under specification '", spec_name,
      "': missing required column(s): ",
      paste(missing_columns, collapse = ", "), ".",
      call. = FALSE
    )
  }

  formula <- make_transition_formula(outcome, spec_name)
  model <- fixest::feglm(
    formula,
    data = data,
    family = stats::binomial(),
    weights = stats::as.formula(paste0("~ ", weight_var)),
    vcov = ~ block_group_geoid
  )

  if (!isTRUE(model$convStatus)) {
    stop(
      "Transition model did not converge for outcome '", outcome,
      "' under specification '", spec_name, "'.",
      call. = FALSE
    )
  }

  model_coefficients <- stats::coef(model)
  missing_terms <- setdiff(model_covariates, names(model_coefficients))
  if (length(missing_terms) > 0L) {
    stop(
      "Transition model omitted requested covariate(s) for outcome '",
      outcome, "' under specification '", spec_name, "': ",
      paste(missing_terms, collapse = ", "),
      ". Check collinearity or separation diagnostics.",
      call. = FALSE
    )
  }
  if (anyNA(model_coefficients) || any(!is.finite(model_coefficients))) {
    stop(
      "Transition model returned missing or non-finite coefficient(s) for ",
      "outcome '", outcome, "' under specification '", spec_name, "'.",
      call. = FALSE
    )
  }

  model
}

make_model_specs <- function(redrisk_dat, fragrisk_dat) {
  if (!is.data.frame(redrisk_dat) || !is.data.frame(fragrisk_dat)) {
    stop("Both risk-set inputs must be data frames.", call. = FALSE)
  }

  list(
    "Redundant -> Fragile" = list(
      data = redrisk_dat,
      outcome = "prop_red_to_fragile",
      weights = "baseline_redundant_n",
      family = "redundant"
    ),
    "Redundant -> Isolated" = list(
      data = redrisk_dat,
      outcome = "prop_red_to_isolated",
      weights = "baseline_redundant_n",
      family = "redundant"
    ),
    "Redundant -> Inundated" = list(
      data = redrisk_dat,
      outcome = "prop_red_to_inundated",
      weights = "baseline_redundant_n",
      family = "redundant"
    ),
    "Redundant -> Worse" = list(
      data = redrisk_dat,
      outcome = "prop_red_to_worse",
      weights = "baseline_redundant_n",
      family = "redundant"
    ),
    "Fragile -> Isolated" = list(
      data = fragrisk_dat,
      outcome = "prop_fragile_to_isolated",
      weights = "baseline_fragile_n",
      family = "fragile"
    ),
    "Fragile -> Inundated" = list(
      data = fragrisk_dat,
      outcome = "prop_fragile_to_inundated",
      weights = "baseline_fragile_n",
      family = "fragile"
    ),
    "Fragile -> Worse" = list(
      data = fragrisk_dat,
      outcome = "prop_fragile_to_worse",
      weights = "baseline_fragile_n",
      family = "fragile"
    )
  )
}

load_block_group_centroids <- function(
    gpkg_path = file.path(
      "outputs", "spatial", "slr_block_group_analysis_approach.gpkg"
    ),
    layer = "slr_0ft"
) {
  if (!requireNamespace("sf", quietly = TRUE)) {
    stop(
      "Package 'sf' is required to derive block-group centroids.",
      call. = FALSE
    )
  }
  if (
    length(gpkg_path) != 1L ||
      is.na(gpkg_path) ||
      !is.character(gpkg_path) ||
      !nzchar(gpkg_path) ||
      !file.exists(gpkg_path)
  ) {
    stop("Block-group GeoPackage does not exist: ", gpkg_path, call. = FALSE)
  }
  if (
    length(layer) != 1L || is.na(layer) ||
      !is.character(layer) || !nzchar(layer)
  ) {
    stop("layer must be one non-empty layer name.", call. = FALSE)
  }

  available_layers <- sf::st_layers(gpkg_path)$name
  if (!(layer %in% available_layers)) {
    stop(
      "GeoPackage layer '", layer, "' does not exist in ", gpkg_path,
      ". Available layers: ", paste(available_layers, collapse = ", "), ".",
      call. = FALSE
    )
  }

  block_groups <- sf::st_read(gpkg_path, layer = layer, quiet = TRUE)
  if (!("block_group_geoid" %in% names(block_groups))) {
    stop(
      "GeoPackage layer '", layer,
      "' is missing block_group_geoid.",
      call. = FALSE
    )
  }
  block_groups$block_group_geoid <- as.character(
    block_groups$block_group_geoid
  )
  if (
    anyNA(block_groups$block_group_geoid) ||
      any(!nzchar(block_groups$block_group_geoid)) ||
      anyDuplicated(block_groups$block_group_geoid)
  ) {
    stop(
      "Block-group geometry has missing, empty, or duplicate GEOIDs.",
      call. = FALSE
    )
  }
  if (nrow(block_groups) == 0L) {
    stop("Block-group geometry layer is empty.", call. = FALSE)
  }
  if (is.na(sf::st_crs(block_groups))) {
    stop("Block-group geometry has no coordinate reference system.", call. = FALSE)
  }
  if (any(sf::st_is_empty(block_groups))) {
    stop("Block-group geometry contains empty features.", call. = FALSE)
  }

  centroid_geometry <- block_groups
  if (isTRUE(sf::st_is_longlat(centroid_geometry))) {
    # A projected equal-area CRS avoids taking polygon centroids in angular
    # coordinates if a future GeoPackage is supplied in longitude/latitude.
    centroid_geometry <- sf::st_transform(centroid_geometry, 5070)
  }
  centroids <- suppressWarnings(sf::st_centroid(centroid_geometry))
  centroids <- sf::st_transform(centroids, 4326)
  coordinates <- sf::st_coordinates(centroids)
  if (nrow(coordinates) != nrow(block_groups)) {
    stop("Centroid coordinate count does not match the block-group count.", call. = FALSE)
  }

  output <- data.frame(
    block_group_geoid = block_groups$block_group_geoid,
    centroid_lat = as.numeric(coordinates[, "Y"]),
    centroid_lon = as.numeric(coordinates[, "X"]),
    stringsAsFactors = FALSE
  )
  if (
    anyNA(output$centroid_lat) || anyNA(output$centroid_lon) ||
      any(!is.finite(output$centroid_lat)) ||
      any(!is.finite(output$centroid_lon)) ||
      any(output$centroid_lat < -90 | output$centroid_lat > 90) ||
      any(output$centroid_lon < -180 | output$centroid_lon > 180)
  ) {
    stop("Derived centroid longitude/latitude values are invalid.", call. = FALSE)
  }

  output[order(output$block_group_geoid), , drop = FALSE]
}

compute_conley_vcov <- function(model, block_group_data, cutoff_km) {
  if (!inherits(model, "fixest")) {
    stop("model must be a fitted fixest model.", call. = FALSE)
  }
  if (!is.data.frame(block_group_data)) {
    stop("block_group_data must be a data frame.", call. = FALSE)
  }
  if (
    length(cutoff_km) != 1L || is.na(cutoff_km) ||
      !is.numeric(cutoff_km) || !is.finite(cutoff_km) || cutoff_km <= 0
  ) {
    stop("cutoff_km must be one positive, finite number.", call. = FALSE)
  }

  coordinate_columns <- c(
    "block_group_geoid",
    CONLEY_LATITUDE_COLUMN,
    CONLEY_LONGITUDE_COLUMN
  )
  missing_columns <- setdiff(coordinate_columns, names(block_group_data))
  if (length(missing_columns) > 0L) {
    stop(
      "Conley covariance data is missing required column(s): ",
      paste(missing_columns, collapse = ", "), ".",
      call. = FALSE
    )
  }
  if (nrow(block_group_data) == 0L) {
    stop("Conley covariance data is empty.", call. = FALSE)
  }

  geoids <- as.character(block_group_data$block_group_geoid)
  latitude <- block_group_data[[CONLEY_LATITUDE_COLUMN]]
  longitude <- block_group_data[[CONLEY_LONGITUDE_COLUMN]]
  if (
    anyNA(geoids) || any(!nzchar(geoids)) ||
      !is.numeric(latitude) || !is.numeric(longitude) ||
      anyNA(latitude) || anyNA(longitude) ||
      any(!is.finite(latitude)) || any(!is.finite(longitude)) ||
      any(latitude < -90 | latitude > 90) ||
      any(longitude < -180 | longitude > 180)
  ) {
    stop(
      "Conley covariance data has invalid GEOIDs or longitude/latitude values.",
      call. = FALSE
    )
  }

  coordinate_map <- unique(data.frame(
    block_group_geoid = geoids,
    centroid_lat = latitude,
    centroid_lon = longitude,
    stringsAsFactors = FALSE
  ))
  if (anyDuplicated(coordinate_map$block_group_geoid)) {
    stop(
      "At least one block-group GEOID maps to conflicting centroid coordinates.",
      call. = FALSE
    )
  }

  observation_index <- fixest::obs(model)
  if (
    (!is.null(model$nobs_origin) && nrow(block_group_data) != model$nobs_origin) ||
    length(observation_index) != stats::nobs(model) ||
      anyNA(observation_index) ||
      any(observation_index < 1L) ||
      any(observation_index > nrow(block_group_data))
  ) {
    stop(
      "Model observation indices do not align with block_group_data. Pass the ",
      "same data frame used to fit the model.",
      call. = FALSE
    )
  }

  conley_vcov <- tryCatch(
    fixest::vcov_conley(
      model,
      lat = CONLEY_LATITUDE_COLUMN,
      lon = CONLEY_LONGITUDE_COLUMN,
      cutoff = as.numeric(cutoff_km)
    ),
    error = function(error) {
      stop(
        "Conley covariance estimation failed. Centroid columns must be joined ",
        "to the model's estimation data before fit_transition_model() is ",
        "called. Original error: ", conditionMessage(error),
        call. = FALSE
      )
    }
  )

  coefficient_names <- names(stats::coef(model))
  if (
    !is.matrix(conley_vcov) ||
      !identical(dim(conley_vcov), c(length(coefficient_names), length(coefficient_names))) ||
      anyNA(conley_vcov) || any(!is.finite(conley_vcov)) ||
      !setequal(rownames(conley_vcov), coefficient_names) ||
      !setequal(colnames(conley_vcov), coefficient_names)
  ) {
    stop("Conley covariance matrix failed its integrity checks.", call. = FALSE)
  }

  attr(conley_vcov, "conley_cutoff_km") <- as.numeric(cutoff_km)
  conley_vcov
}
