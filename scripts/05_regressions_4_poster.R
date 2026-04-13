# ============================================================
# 0. Packages
# ============================================================
library(tidyverse)
library(fixest)
library(marginaleffects)
library(modelsummary)

# ============================================================
# 1. Read data
# ============================================================
dat <- readr::read_csv(
  "data/processed/analysis/block_group_analysis_dataset.csv",
  show_col_types = FALSE,
  col_types = cols(
    block_group_geoid = col_character(),
    tract_geoid = col_character(),
    county_fips = col_character()
  )) %>% 
  select(-poverty_rate) # Unable to pull from census at the block group geography

# ============================================================
# 2. Build baseline denominators from the 0 ft reference rows
# ============================================================
base_counts <- dat %>%
  filter(slr_ft == 0) %>%
  transmute(
    block_group_geoid,
    baseline_total_blocks   = total_blocks,
    baseline_redundant_n    = block_centroid_redundant,
    baseline_fragile_n      = block_centroid_fragile,
    baseline_isolated_n     = block_centroid_isolated,
    baseline_inundated_n    = block_centroid_inundated,
    baseline_nonisolated_n  = total_blocks - block_centroid_isolated,
    baseline_noninundated_n = total_blocks - block_centroid_inundated
  )

dat2 <- dat %>%
  left_join(base_counts, by = "block_group_geoid") %>%
  mutate(
    slr_ft_f = factor(slr_ft),
    
    # Calculating transition shares
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
    ),
    
    # Detour burden
    log_detour = if_else(
      !is.na(mean_path_inflation_ratio) & mean_path_inflation_ratio > 0,
      log(mean_path_inflation_ratio),
      NA_real_
    )
  ) %>%
  mutate(
    across(
      c(pct_black_nh, pct_hispanic, renter_share, log_median_income, median_age),
      ~ as.numeric(scale(.x)), # Z-scoring for comparability
      .names = "z_{.col}"
    )
  )

# Keep only rows with the core covariates observed
analysis_dat <- dat2 %>%
  drop_na(
    z_pct_black_nh,
    z_pct_hispanic,
    z_renter_share,
    z_log_median_income,
    z_median_age
  )

# ============================================================
# 5. Transition models --
#    Among baseline-redundant blocks, who becomes fragile / isolated / inundated?
# ============================================================
trans_dat <- analysis_dat %>% filter(slr_ft > 0)

redrisk_dat <- trans_dat %>%
  filter(baseline_redundant_n > 0) %>% # Filtering to block groups that had one baseline redundant glock
  mutate(
    prop_red_to_worse     = any_loss_of_redundancy / baseline_redundant_n,
    prop_red_to_fragile   = baseline_redundant_to_fragile / baseline_redundant_n,
    prop_red_to_isolated  = baseline_redundant_to_isolated / baseline_redundant_n,
    prop_red_to_inundated = baseline_redundant_to_inundated / baseline_redundant_n
  )

m_red_to_fragile <- feglm(
  prop_red_to_fragile ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name + slr_ft_f,
  data = redrisk_dat,
  family = binomial(),
  weights = ~ baseline_redundant_n,
  vcov = ~ block_group_geoid
)

m_red_to_isolated <- feglm(
  prop_red_to_isolated ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name + slr_ft_f,
  data = redrisk_dat,
  family = binomial(),
  weights = ~ baseline_redundant_n,
  vcov = ~ block_group_geoid
)

m_red_to_inundated <- feglm(
  prop_red_to_inundated ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name + slr_ft_f,
  data = redrisk_dat,
  family = binomial(),
  weights = ~ baseline_redundant_n,
  vcov = ~ block_group_geoid
)

# Broadest transition outcome: any downgrade from baseline redundancy
m_red_to_worse <- feglm(
  prop_red_to_worse ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name + slr_ft_f,
  data = redrisk_dat,
  family = binomial(),
  weights = ~ baseline_redundant_n,
  vcov = ~ block_group_geoid
)
etable(
  m_red_to_fragile,
  m_red_to_isolated,
  m_red_to_inundated, 
  m_red_to_worse,
  headers = c("Redundant → Fragile", "Redundant → Isolated", 
              "Redundant → Inundated", "Redundant → Worse"
              ),
  dict = c(
    z_pct_black_nh = "Black share (z)",
    z_pct_hispanic = "Hispanic share (z)",
    z_renter_share = "Renter share (z)",
    z_log_median_income = "Log median income (z)",
    z_median_age = "Median age (z)"
  ),
  fitstat = ~ n + ll,
  digits = 3
)


# ============================================================
# 6. What happens to already-fragile places? -- looking at one level down.
# ============================================================
fragrisk_dat <- trans_dat  %>%
  filter(baseline_fragile_n > 0) %>%
  mutate(
    fragile_to_worse_n        = baseline_fragile_to_isolated + baseline_fragile_to_inundated,
    prop_fragile_to_worse     = fragile_to_worse_n / baseline_fragile_n,
    prop_fragile_to_isolated  = baseline_fragile_to_isolated / baseline_fragile_n,
    prop_fragile_to_inundated = baseline_fragile_to_inundated / baseline_fragile_n
  )

m_fragile_to_worse <- feglm(
  prop_fragile_to_worse ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name + slr_ft_f,
  data = fragrisk_dat,
  family = binomial(),
  weights = ~ baseline_fragile_n,
  vcov = ~ block_group_geoid
)

m_fragile_to_isolated <- feglm(
  prop_fragile_to_isolated ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name + slr_ft_f,
  data = fragrisk_dat,
  family = binomial(),
  weights = ~ baseline_fragile_n,
  vcov = ~ block_group_geoid
)

m_fragile_to_inundated <- feglm(
  prop_fragile_to_inundated ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name + slr_ft_f,
  data = fragrisk_dat,
  family = binomial(),
  weights = ~ baseline_fragile_n,
  vcov = ~ block_group_geoid
)
etable(
  m_fragile_to_isolated,
  m_fragile_to_inundated,
  m_fragile_to_worse,
  headers = c(
    "Fragile → Isolated",
    "Fragile → Inundated",
    "Fragile → Worse"
    
  ), 
  
  dict = c(
    z_pct_black_nh = "Black share (z)",
    z_pct_hispanic = "Hispanic share (z)",
    z_renter_share = "Renter share (z)",
    z_log_median_income = "Log median income (z)",
    z_median_age = "Median age (z)"
  ),
  fitstat = ~ n + ll,
  digits = 3
  
  )


# Looking at the last transition!
isolrisk_dat <- trans_dat %>%
  filter(baseline_isolated_n > 0) %>%
  mutate(
    prop_isolated_to_inundated =
      baseline_isolated_to_inundated / baseline_isolated_n
  )


m_isolated_to_inundated <- feglm(
  prop_isolated_to_inundated ~
    z_pct_black_nh +
    z_pct_hispanic +
    z_renter_share +
    z_log_median_income +
    z_median_age |
    county_name + slr_ft_f,
  data = isolrisk_dat,
  family = binomial(),
  weights = ~ baseline_isolated_n,
  vcov = ~ block_group_geoid
)

etable(
  m_fragile_to_isolated,
  m_fragile_to_inundated,
  m_fragile_to_worse,
  m_isolated_to_inundated,
  headers = c(
    "Fragile → Isolated",
    "Fragile → Inundated",
    "Fragile → Worse",
    "Isolated → Inundated"
    
  ), 
  
  dict = c(
    z_pct_black_nh = "Black share (z)",
    z_pct_hispanic = "Hispanic share (z)",
    z_renter_share = "Renter share (z)",
    z_log_median_income = "Log median income (z)",
    z_median_age = "Median age (z)"
  ),
  fitstat = ~ n + ll,
  digits = 3
  
)

# -------------------------------------------------------------------------------
# BUILDING A TABLE WITH ALL TRANSITION PROBABILTIES -----------------------------
# -------------------------------------------------------------------------------

etable(
  m_red_to_fragile,
  m_red_to_isolated,
  m_red_to_inundated, 
  m_red_to_worse,
  m_fragile_to_isolated,
  m_fragile_to_inundated,
  m_fragile_to_worse,
  m_isolated_to_inundated,
  headers = c(
    "Redundant → Fragile", 
    "Redundant → Isolated", 
    "Redundant → Inundated", 
    "Redundant → Worse",
    "Fragile → Isolated",
    "Fragile → Inundated",
    "Fragile → Worse",
    "Isolated → Inundated"
    
  ), 
  
  dict = c(
    z_pct_black_nh = "Black share (z)",
    z_pct_hispanic = "Hispanic share (z)",
    z_renter_share = "Renter share (z)",
    z_log_median_income = "Log median income (z)",
    z_median_age = "Median age (z)"
  ),
  fitstat = ~ n + ll,
  digits = 3
  
)

# ============================================================
# 7. Detour burden among still-connected places -- drop?
# ============================================================
detour_dat <- trans_dat %>%
  filter(!is.na(log_detour))

m_detour <- feols(
  log_detour ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name + slr_ft_f,
  data = detour_dat,
  weights = ~ total_blocks,
  vcov = ~ block_group_geoid
)

summary(m_detour)

hist(detour_dat$log_detour) # There is not much variation 

# ============================================================
# 8. Does inequality steepen as SLR increases?
# ============================================================
m_loss_interact <- feols(
  share_lost_redundancy ~
    (z_pct_black_nh + z_pct_hispanic + z_renter_share +
       z_log_median_income + z_median_age) * i(slr_ft, ref = 1) |
    county_name,
  data = trans_dat,
  weights = ~ total_blocks,
  vcov = ~ block_group_geoid
)

summary(m_loss_interact)

# ============================================================
# 9. Tighter geographic control: tract FE robustness
# ============================================================
m_loss_tractfe <- feols(
  share_lost_redundancy ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    tract_geoid + slr_ft_f,
  data = trans_dat,
  weights = ~ total_blocks,
  vcov = ~ block_group_geoid
)

summary(m_loss_tractfe) # This does not mean much, inequality can be much more fine-grained. 

# ============================================================
# 9b. Helpers for bootstrap AME uncertainty
# ============================================================
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

    raw_call <- as.list(model[["call"]])
    boot_args <- as.list(raw_call[nzchar(names(raw_call))])
    boot_args[["data"]] <- boot_data
    boot_args[["vcov"]] <- "iid"
    boot_args[["notes"]] <- FALSE

    boot_model <- tryCatch(
      suppressWarnings(do.call(feglm, boot_args)),
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

# ============================================================
# 10. Marginal effects for logit-style models
# ============================================================
ame_red_to_fragile <- avg_slopes(
  m_red_to_fragile,
  vcov = FALSE
)

ame_red_to_isolated <- avg_slopes(
  m_red_to_isolated,
  vcov = FALSE
)

ame_red_to_worse <- avg_slopes(
  m_red_to_worse,
  vcov = FALSE
)

ame_red_to_inundated <- avg_slopes(
  m_red_to_inundated,
  vcov = FALSE
)

ame_fragile_to_isolated <- avg_slopes(
  m_fragile_to_isolated,
  vcov = FALSE
)

ame_fragile_to_inundated <- avg_slopes(
  m_fragile_to_inundated,
  vcov = FALSE
)

ame_fragile_to_worse <- avg_slopes(
  m_fragile_to_worse,
  vcov = FALSE
)

ame_isolated_to_inundated <- avg_slopes(
  m_isolated_to_inundated,
  vcov = FALSE
)

ame_red_to_fragile_boot <- bootstrap_avg_slopes(
  m_red_to_fragile,
  redrisk_dat,
  label = "Redundant -> Fragile",
  seed = AME_BOOT_SEED + 1L
)

ame_red_to_isolated_boot <- bootstrap_avg_slopes(
  m_red_to_isolated,
  redrisk_dat,
  label = "Redundant -> Isolated",
  seed = AME_BOOT_SEED + 2L
)

ame_red_to_worse_boot <- bootstrap_avg_slopes(
  m_red_to_worse,
  redrisk_dat,
  label = "Redundant -> Worse",
  seed = AME_BOOT_SEED + 3L
)

ame_red_to_inundated_boot <- bootstrap_avg_slopes(
  m_red_to_inundated,
  redrisk_dat,
  label = "Redundant -> Inundated",
  seed = AME_BOOT_SEED + 4L
)

ame_fragile_to_isolated_boot <- bootstrap_avg_slopes(
  m_fragile_to_isolated,
  fragrisk_dat,
  label = "Fragile -> Isolated",
  seed = AME_BOOT_SEED + 5L
)

ame_fragile_to_inundated_boot <- bootstrap_avg_slopes(
  m_fragile_to_inundated,
  fragrisk_dat,
  label = "Fragile -> Inundated",
  seed = AME_BOOT_SEED + 6L
)

ame_fragile_to_worse_boot <- bootstrap_avg_slopes(
  m_fragile_to_worse,
  fragrisk_dat,
  label = "Fragile -> Worse",
  seed = AME_BOOT_SEED + 7L
)

ame_isolated_to_inundated_boot <- bootstrap_avg_slopes(
  m_isolated_to_inundated,
  isolrisk_dat,
  label = "Isolated -> Inundated",
  seed = AME_BOOT_SEED + 8L
)

ame_red_to_fragile
ame_red_to_isolated
ame_red_to_worse
ame_red_to_inundated
ame_fragile_to_isolated
ame_fragile_to_inundated
ame_fragile_to_worse
ame_isolated_to_inundated

ame_red_to_fragile_boot
ame_red_to_isolated_boot
ame_red_to_worse_boot
ame_red_to_inundated_boot
ame_fragile_to_isolated_boot
ame_fragile_to_inundated_boot
ame_fragile_to_worse_boot
ame_isolated_to_inundated_boot

# ============================================================
# 11. Export bootstrapped AMEs to Excel
# ============================================================
library(openxlsx)

ame_boot_combined <- bind_rows(
  ame_red_to_fragile_boot      %>% mutate(transition = "Redundant → Fragile"),
  ame_red_to_isolated_boot     %>% mutate(transition = "Redundant → Isolated"),
  ame_red_to_worse_boot        %>% mutate(transition = "Redundant → Worse"),
  ame_red_to_inundated_boot    %>% mutate(transition = "Redundant → Inundated"),
  ame_fragile_to_isolated_boot %>% mutate(transition = "Fragile → Isolated"),
  ame_fragile_to_inundated_boot %>% mutate(transition = "Fragile → Inundated"),
  ame_fragile_to_worse_boot    %>% mutate(transition = "Fragile → Worse"),
  ame_isolated_to_inundated_boot %>% mutate(transition = "Isolated → Inundated")
) %>%
  select(transition, term, estimate, std.error, statistic, p.value,
         conf.low, conf.high, conf.level, n_boot, n_boot_fail)

write.xlsx(
  ame_boot_combined,
  file = here::here("outputs", "tables", "ame_bootstrap_results.xlsx"),
  overwrite = TRUE
)

