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
# 3. Baseline models: who already has fragile access at 0 ft?
# ============================================================
base_dat <- analysis_dat %>%
  filter(slr_ft == 0) %>%
  mutate(
    prop_fragile = block_centroid_fragile / total_blocks
  )

# Paper-style weighted linear model on shares
m_base_share <- feols(
  share_fragile ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name,
  data = base_dat,
  weights = ~ total_blocks, # Weighted across total blocks
  vcov = ~ tract_geoid
) 
summary(m_base_share)

# Binomial grouped model: fragile blocks out of total blocks
m_base_binom <- feglm(
  prop_fragile ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name,
  data = base_dat,
  family = binomial(),
  weights = ~ total_blocks,
  vcov = ~ tract_geoid
) # Coefficents on the log-odds scale. 
summary(m_base_binom)

# Quick recap on the binomal model ---
#   weights = ~ baseline_redundant_n (or baseline_fragile_n, or
#   total_blocks) is the number of binomial trials. The outcome
#   proportion times this weight recovers the event count. The
#   likelihood naturally downweights block groups with few trials
#   because small samples carry less information about the true
#   transition probability.

# Binomial regression (feglm, family = binomial()):
#   Models a proportion as a count of events out of a known
#   number of trials (e.g., 12 blocks became fragile out of 30
#   that were redundant at baseline). Unlike OLS on a share,
#   the binomial likelihood (1) respects the 0-1 bounds of the
#   outcome, (2) naturally accounts for differences in precision
#   across block groups with different trial counts, and
#   (3) estimates coefficients on the log-odds scale, which can
#   be converted to probability-scale effects via avg_slopes().

# Broader baseline vulnerability
m_base_worse <- feols(
  share_fragile_or_worse ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name,
  data = base_dat,
  weights = ~ total_blocks,
  vcov = ~ tract_geoid
)
summary(m_base_worse)

# ============================================================
# 4. Pooled SLR models (slr_ft > 0), clustered by block group
#    Identifies off cross-sectional demographic variation within 
#    a county and within a given SLR scenario.
# ============================================================
trans_dat <- analysis_dat %>% filter(slr_ft > 0)

# Descriptive pooled model using saved share
m_loss_share <- feols(
  share_lost_redundancy ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name + slr_ft_f,
  data = trans_dat,
  weights = ~ total_blocks,
  vcov = ~ block_group_geoid
)

summary(m_loss_share)

# ============================================================
# 5. Transition models --
#    Among baseline-redundant blocks, who becomes fragile / isolated / inundated?
# ============================================================
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
  m_fragile_to_isolated,
  m_fragile_to_inundated,
  m_fragile_to_worse,
  m_fragile_to_isolated,
  m_fragile_to_inundated,
  m_fragile_to_worse,
  m_isolated_to_inundated,
  headers = c(
    "Fragile → Isolated",
    "Fragile → Inundated",
    "Fragile → Worse",
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

ame_red_to_fragile
ame_red_to_isolated
ame_red_to_worse
ame_red_to_inundated
ame_fragile_to_isolated
ame_fragile_to_inundated

# ============================================================
# 11. Simple model table
# ============================================================
modelsummary(
  list(
    "Baseline fragile share" = m_base_share,
    "Baseline fragile binomial" = m_base_binom,
    "Lost redundancy (share)" = m_loss_share,
    "Redundant -> fragile" = m_red_to_fragile,
    "Redundant -> isolated" = m_red_to_isolated,
    "Detour burden" = m_detour
  ),
  stars = TRUE
)

# ============================================================
# 12. Comparing frameworks: what does fragility expose
#     that isolation alone misses?
# ============================================================
# The core argument of my extension is that Best et al.'s
# binary isolated-vs-connected framework misses a large middle
# category of *fragile* access. These tables put isolation-only
# models next to redundancy/fragility models so the reader can
# see where demographic gradients appear, disappear, or change
# sign when you move from the binary lens to the richer one.

# ---- 12a. Baseline cross-section (slr_ft == 0) ----
# "Before any SLR, who already lives with degraded access?"
# Best-style: share isolated.  Extension: share fragile,
# share fragile-or-worse.

base_dat <- base_dat %>%
  mutate(
    prop_isolated = block_centroid_isolated / total_blocks
  )

# Best et al. lens: baseline isolation share (OLS)
m_base_isolated_ols <- feols(
  share_isolated ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name,
  data    = base_dat,
  weights = ~ total_blocks,
  vcov    = ~ tract_geoid
)

# Best et al. lens: baseline isolation (binomial)
m_base_isolated_binom <- feglm(
  prop_isolated ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name,
  data    = base_dat,
  family  = binomial(),
  weights = ~ total_blocks,
  vcov    = ~ tract_geoid
)


# (m_base_share and m_base_worse already exist from Section 3)

modelsummary(
  list(
    "Isolated (OLS)"          = m_base_isolated_ols,
    "Isolated (binomial)"     = m_base_isolated_binom,
    "Fragile (OLS)"           = m_base_share,
    "Fragile (binomial)"      = m_base_binom,
    "Fragile-or-worse (OLS)"  = m_base_worse
  ),
  stars    = TRUE,
  title    = "Table A — Baseline vulnerability: isolation vs. fragility (slr_ft = 0)",
  notes    = c(
    "Unit: block group. Weights: total blocks. Clustered SEs: tract.",
    "All covariates are z-scored. County fixed effects absorbed."
  )
)

# ---- 12b. SLR panel: unconditional new-outcome shares ----
# "Under SLR, who gains newly degraded access?"
# Best-style: share newly isolated.
# Extension: share newly fragile, share lost redundancy.

m_new_isolated_share <- feols(
  share_new_isolated ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name + slr_ft_f,
  data    = trans_dat,
  weights = ~ total_blocks,
  vcov    = ~ block_group_geoid
)

m_new_fragile_share <- feols(
  share_new_fragile ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name + slr_ft_f,
  data    = trans_dat,
  weights = ~ total_blocks,
  vcov    = ~ block_group_geoid
)

# (m_loss_share already exists from Section 4)

modelsummary(
  list(
    "New isolated (share)"       = m_new_isolated_share,
    "New fragile (share)"        = m_new_fragile_share,
    "Lost redundancy (share)"    = m_loss_share
  ),
  stars    = TRUE,
  title    = "Table B — SLR transitions: isolation-only vs. redundancy framework",
  notes    = c(
    "Unit: block group x SLR scenario (1-6 ft). Weights: total blocks.",
    "Clustered SEs: block group. County + scenario FEs absorbed."
  )
)

# ---- 12c. Conditional transitions from baseline redundant ----
# "Among blocks that HAD redundant access, who loses it —
#  and does the destination matter?"
# This is the sharpest test. The isolation framework would only
# flag redundant -> isolated. The extension also sees
# redundant -> fragile, which is a larger and earlier-onset group.

# (m_red_to_fragile, m_red_to_isolated, m_red_to_worse
#  already exist from Section 5)

modelsummary(
  list(
    "Redundant -> fragile"    = m_red_to_fragile,
    "Redundant -> isolated"   = m_red_to_isolated,
    "Redundant -> inundated"  = m_red_to_inundated,
    "Redundant -> any worse"  = m_red_to_worse
  ),
  stars    = TRUE,
  title    = "Table C — Conditional transitions from baseline redundancy (binomial)",
  notes    = c(
    "Unit: block group x SLR scenario, restricted to baseline_redundant_n > 0.",
    "Weights: baseline redundant block count. Clustered SEs: block group.",
    "County + scenario FEs absorbed. Coefficients are on the log-odds scale."
  )
)

# ---- 12d. Severity split: mild vs. severe degradation ----
# "Does the demographic profile of mild degradation differ
#  from severe degradation?"
# Mild  = fragile only (still has one dry path)
# Severe = isolated + inundated (no dry path or underwater)
#
# If coefficients differ between these two, that is direct
# evidence that the fragility concept captures a *different*
# population than isolation alone.

redrisk_dat <- redrisk_dat %>%
  mutate(
    # Severe: redundant blocks that became isolated OR inundated
    red_to_severe_n = baseline_redundant_to_isolated +
      baseline_redundant_to_inundated,
    prop_red_to_severe = if_else(
      baseline_redundant_n > 0,
      red_to_severe_n / baseline_redundant_n,
      NA_real_
    ),
    # Mild: redundant blocks that became fragile only
    # (prop_red_to_fragile already exists)
    
    # Composite: any degradation (already prop_red_to_worse)
    dummy = TRUE
  )

m_red_to_mild <- feglm(
  prop_red_to_fragile ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name + slr_ft_f,
  data    = redrisk_dat,
  family  = binomial(),
  weights = ~ baseline_redundant_n,
  vcov    = ~ block_group_geoid
)

m_red_to_severe <- feglm(
  prop_red_to_severe ~
    z_pct_black_nh + z_pct_hispanic + z_renter_share +
    z_log_median_income + z_median_age |
    county_name + slr_ft_f,
  data    = redrisk_dat,
  family  = binomial(),
  weights = ~ baseline_redundant_n,
  vcov    = ~ block_group_geoid
)

modelsummary(
  list(
    "Mild (redundant -> fragile)"             = m_red_to_mild,
    "Severe (redundant -> isolated/inundated)" = m_red_to_severe,
    "Any degradation (redundant -> worse)"     = m_red_to_worse
  ),
  stars    = TRUE,
  title    = "Table D — Mild vs. severe degradation from baseline redundancy",
  notes    = c(
    "Mild = transition to fragile only. Severe = transition to isolated or inundated.",
    "Binomial models, weights: baseline redundant blocks. Clustered SEs: block group.",
    "If coefficients differ between Mild and Severe, the fragility concept",
    "captures a demographically distinct population that isolation alone misses."
  )
)

# ---- 12e. Marginal effects for the comparison models ----
# Translate log-odds into probability-scale effects for
# interpretability.  Use clustered SEs to match the models.

ame_base_isolated <- avg_slopes(
  m_base_isolated_binom,
  vcov = FALSE
)

ame_red_to_mild <- avg_slopes(
  m_red_to_mild,
  vcov = FALSE
)

ame_red_to_severe <- avg_slopes(
  m_red_to_severe,
  vcov = FALSE
)

cat("\n=== AME: Baseline isolated (binomial) ===\n")
print(ame_base_isolated)

cat("\n=== AME: Redundant -> fragile (mild) ===\n")
print(ame_red_to_mild)

cat("\n=== AME: Redundant -> isolated/inundated (severe) ===\n")
print(ame_red_to_severe)

# ============================================================
# 13. Spatial autocorrelation diagnostic: Moran's I
# ============================================================
# Tests whether residuals from the main models are spatially
# clustered. If Moran's I is insignificant, no spatial issue.
# If significant, we note it and point to tract FE + clustering
# as partial protection.

library(sf)
library(spdep)
library(tigris)
options(tigris_use_cache = TRUE)

# ---- 13a. Get block group geometries ----
# Pull 2020 block group boundaries for the three counties.
# county FIPS: Broward = 011, Miami-Dade = 086, Palm Beach = 099

bg_broward   <- block_groups(state = "FL", county = "011", year = 2020)
bg_miamidade <- block_groups(state = "FL", county = "086", year = 2020)
bg_palmbeach <- block_groups(state = "FL", county = "099", year = 2020)

bg_sf <- bind_rows(bg_broward, bg_miamidade, bg_palmbeach) %>%
  rename(block_group_geoid = GEOID) %>%
  select(block_group_geoid, geometry)

# ---- 13b. Merge residuals onto geometries ----
# We test residuals from several key models.  Start with the
# baseline fragile share model (cross-sectional, one row per BG).

base_resids <- base_dat %>%
  mutate(
    resid_base_fragile = residuals(m_base_share),
    resid_base_isolated = residuals(m_base_isolated_ols),
    resid_base_worse = residuals(m_base_worse)
  ) %>%
  select(block_group_geoid, starts_with("resid_"))

bg_test <- bg_sf %>%
  inner_join(base_resids, by = "block_group_geoid") %>%
  filter(!st_is_empty(geometry))

# ---- 13c. Build spatial weights matrix ----
# Queen contiguity: two block groups are neighbors if they share
# any boundary point.  This is the most common default.
# zero.policy = TRUE allows block groups with no neighbors
# (e.g., islands) to remain in the dataset.

nb <- poly2nb(bg_test, queen = TRUE)
lw <- nb2listw(nb, style = "W", zero.policy = TRUE)

cat("Spatial weights summary:\n")
print(summary(nb))

# ---- 13d. Moran's I tests on baseline model residuals ----

cat("\n=== Moran's I: Baseline fragile share (OLS) ===\n")
mi_fragile <- moran.test(bg_test$resid_base_fragile, lw,
                         zero.policy = TRUE)
print(mi_fragile)

cat("\n=== Moran's I: Baseline isolated share (OLS) ===\n")
mi_isolated <- moran.test(bg_test$resid_base_isolated, lw,
                          zero.policy = TRUE)
print(mi_isolated)

cat("\n=== Moran's I: Baseline fragile-or-worse share (OLS) ===\n")
mi_worse <- moran.test(bg_test$resid_base_worse, lw,
                       zero.policy = TRUE)
print(mi_worse)

# ---- 13e. Panel model residuals (pick one scenario to test) ----
# Moran's I is a cross-sectional test, so we extract residuals
# from one SLR scenario at a time.  3 ft is a reasonable middle
# scenario to diagnose.

# Get the row indices in trans_dat that correspond to slr_ft == 3
scenario_3_idx <- which(trans_dat$slr_ft == 3)

# Extract residuals from the full trans_dat model at those rows
panel_resids_3ft <- trans_dat %>%
  filter(slr_ft == 3) %>%
  mutate(
    resid_loss_share = residuals(m_loss_share)[scenario_3_idx]
  ) %>%
  select(block_group_geoid, resid_loss_share)

bg_test_panel <- bg_sf %>%
  inner_join(panel_resids_3ft, by = "block_group_geoid") %>%
  filter(!st_is_empty(geometry))

# Rebuild weights for this subset (may differ slightly if some
# BGs drop out of the panel)
nb_panel <- poly2nb(bg_test_panel, queen = TRUE)
lw_panel <- nb2listw(nb_panel, style = "W", zero.policy = TRUE)

cat("\n=== Moran's I: Lost redundancy share at 3 ft SLR ===\n")
mi_loss_3ft <- moran.test(bg_test_panel$resid_loss_share, lw_panel,
                          zero.policy = TRUE)
print(mi_loss_3ft)

# ---- 13f. Summary for paper ----
cat("\n========================================\n")
cat("SUMMARY FOR FOOTNOTE / METHODS SECTION\n")
cat("========================================\n")
cat(sprintf(
  "Baseline fragile:      Moran's I = %.4f, p = %.2e\n",
  mi_fragile$estimate["Moran I statistic"],
  mi_fragile$p.value
))
cat(sprintf(
  "Baseline isolated:     Moran's I = %.4f, p = %.2e\n",
  mi_isolated$estimate["Moran I statistic"],
  mi_isolated$p.value
))
cat(sprintf(
  "Baseline frag/worse:   Moran's I = %.4f, p = %.2e\n",
  mi_worse$estimate["Moran I statistic"],
  mi_worse$p.value
))
cat(sprintf(
  "Lost redundancy (3ft): Moran's I = %.4f, p = %.2e\n",
  mi_loss_3ft$estimate["Moran I statistic"],
  mi_loss_3ft$p.value
))