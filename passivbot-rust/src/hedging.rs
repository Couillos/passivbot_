use crate::constants::{CLOSE, HIGH, LOW};
use crate::types::{BotParams, HedgeEntryMode, HedgeExitMode, VolatilityMethod};
use ndarray::ArrayView3;

// ========== PURE CALCULATION FUNCTIONS ==========

/// Calculates Average True Range (ATR)
/// Pure function: hlcvs + k + period → ATR
pub fn calculate_atr(
    hlcvs: &ArrayView3<f64>,
    k: usize,
    idx: usize,
    period: usize,
) -> Option<f64> {
    if k < period {
        return None;
    }
    
    let mut tr_sum = 0.0;
    let mut count = 0;
    
    for i in (k.saturating_sub(period - 1))..=k {
        let high = hlcvs[[i, idx, HIGH]];
        let low = hlcvs[[i, idx, LOW]];
        let prev_close = if i > 0 {
            hlcvs[[i - 1, idx, CLOSE]]
        } else {
            hlcvs[[i, idx, CLOSE]]
        };
        
        if !high.is_finite() || !low.is_finite() || !prev_close.is_finite() {
            continue;
        }
        
        // True Range = max(H-L, |H-PC|, |L-PC|)
        let tr1 = high - low;
        let tr2 = (high - prev_close).abs();
        let tr3 = (low - prev_close).abs();
        let tr = tr1.max(tr2).max(tr3);
        
        tr_sum += tr;
        count += 1;
    }
    
    if count == 0 {
        return None;
    }
    
    let atr = tr_sum / count as f64;
    if atr > 0.0 && atr.is_finite() {
        Some(atr)
    } else {
        None
    }
}

/// Calculates volatility using standard deviation of returns
pub fn calculate_volatility_std(
    hlcvs: &ArrayView3<f64>,
    k: usize,
    idx: usize,
    period: usize,
) -> f64 {
    if k < period {
        return 0.0;
    }
    
    let mut returns = Vec::with_capacity(period);
    
    for i in (k.saturating_sub(period - 1))..=k {
        if i == 0 {
            continue;
        }
        let current = hlcvs[[i, idx, CLOSE]];
        let prev = hlcvs[[i - 1, idx, CLOSE]];
        
        if prev > 0.0 && current.is_finite() && prev.is_finite() {
            returns.push((current - prev) / prev);
        }
    }
    
    if returns.is_empty() {
        return 0.0;
    }
    
    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance: f64 = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / returns.len() as f64;
    
    variance.sqrt()
}

/// Calculates Rate of Change (ROC) - percentage change
pub fn calculate_roc(
    hlcvs: &ArrayView3<f64>,
    k: usize,
    idx: usize,
    period: usize,
) -> f64 {
    if k < period {
        return 0.0;
    }
    
    let current_price = hlcvs[[k, idx, CLOSE]];
    let past_price = hlcvs[[k.saturating_sub(period), idx, CLOSE]];
    
    if past_price <= 0.0 || !current_price.is_finite() || !past_price.is_finite() {
        return 0.0;
    }
    
    ((current_price - past_price) / past_price).abs()
}

/// Gets current volatility based on configured method
pub fn get_current_volatility(
    hlcvs: &ArrayView3<f64>,
    k: usize,
    idx: usize,
    bp: &BotParams,
) -> f64 {
    match bp.hedge_volatility_method {
        VolatilityMethod::Roc => calculate_roc(hlcvs, k, idx, bp.hedge_roc_period),
        VolatilityMethod::Std => calculate_volatility_std(hlcvs, k, idx, bp.hedge_volatility_period),
    }
}

/// Checks if volatility is high
pub fn is_high_volatility(
    hlcvs: &ArrayView3<f64>,
    k: usize,
    idx: usize,
    bp: &BotParams,
) -> (bool, f64) {
    let volatility = get_current_volatility(hlcvs, k, idx, bp);
    (volatility >= bp.hedge_high_volatility_threshold, volatility)
}

/// Checks if volatility is normal
pub fn is_normal_volatility(
    hlcvs: &ArrayView3<f64>,
    k: usize,
    idx: usize,
    bp: &BotParams,
) -> (bool, f64) {
    let volatility = get_current_volatility(hlcvs, k, idx, bp);
    (volatility <= bp.hedge_normal_volatility_threshold, volatility)
}

// ========== EVALUATION FUNCTIONS ==========

/// Determines if hedge should be opened based on entry mode
/// Pure function: current state → bool
pub fn should_open_hedge(
    hlcvs: &ArrayView3<f64>,
    k: usize,
    idx: usize,
    long_price: f64,
    wallet_exposure: f64,
    bp: &BotParams,
) -> bool {
    let close = hlcvs[[k, idx, CLOSE]];
    
    if !close.is_finite() || close <= 0.0 {
        return false;
    }
    
    // 1. Check exposure >= 95%
    let exposure_threshold = bp.hedge_min_exposure_pct * bp.wallet_exposure_limit;
    if wallet_exposure < exposure_threshold {
        return false;
    }
    
    // 2. Evaluation based on entry mode
    match bp.hedge_entry_mode {
        HedgeEntryMode::AtrOnly => {
            // Only ATR + exposure
            let atr = match calculate_atr(hlcvs, k, idx, bp.hedge_atr_period) {
                Some(atr) if atr > 0.0 => atr,
                _ => return false,
            };
            
            let distance = (long_price - close).abs();
            let distance_in_atr = distance / atr;
            
            distance_in_atr >= bp.hedge_distance_atr_trigger
        }
        
        HedgeEntryMode::VolatilityOnly => {
            // Only volatility + exposure (IGNORES ATR)
            let (is_high, _) = is_high_volatility(hlcvs, k, idx, bp);
            is_high
        }
        
        HedgeEntryMode::AtrAndVolatility => {
            // Both requirements
            let atr = match calculate_atr(hlcvs, k, idx, bp.hedge_atr_period) {
                Some(atr) if atr > 0.0 => atr,
                _ => return false,
            };
            
            let distance = (long_price - close).abs();
            let distance_in_atr = distance / atr;
            
            if distance_in_atr < bp.hedge_distance_atr_trigger {
                return false;
            }
            
            let (is_high, _) = is_high_volatility(hlcvs, k, idx, bp);
            is_high
        }
    }
}

/// Determines if hedge should be closed
/// Pure function: current state → (bool, reason)
pub fn should_close_hedge(
    hlcvs: &ArrayView3<f64>,
    k: usize,
    idx: usize,
    long_exists: bool,
    wallet_exposure: f64,
    hedge_sl_price: f64,
    hedge_sl_moved_to_be: bool,
    bp: &BotParams,
) -> (bool, &'static str) {
    let close = hlcvs[[k, idx, CLOSE]];
    let high = hlcvs[[k, idx, HIGH]];
    
    // CASE 1: Long closed (orphan hedge)
    if !long_exists {
        return (true, "long_closed");
    }
    
    // CASE 2: Exposure with hysteresis
    let close_threshold = bp.hedge_min_exposure_pct_to_close * bp.wallet_exposure_limit;
    if wallet_exposure < close_threshold {
        return (true, "exposure_reduced");
    }
    
    // CASE 3: Volatility normalized (WithVolatility mode)
    if bp.hedge_exit_mode == HedgeExitMode::WithVolatility {
        let (is_normal, _) = is_normal_volatility(hlcvs, k, idx, bp);
        if is_normal {
            return (true, "volatility_normalized");
        }
    }
    
    // CASE 4: Stop loss hit
    if hedge_sl_moved_to_be && high >= hedge_sl_price {
        return (true, "stop_loss");
    }
    
    (false, "")
}

/// Calculates new SL for breakeven move
/// Pure function: current state → Option<new_sl>
pub fn calculate_new_sl_for_breakeven(
    hlcvs: &ArrayView3<f64>,
    k: usize,
    idx: usize,
    hedge_entry_price: f64,
    atr_at_entry: f64,
    hedge_sl_moved_to_be: bool,
    bp: &BotParams,
) -> Option<f64> {
    if hedge_sl_moved_to_be || atr_at_entry <= 0.0 {
        return None;
    }
    
    let close = hlcvs[[k, idx, CLOSE]];
    if !close.is_finite() || close <= 0.0 {
        return None;
    }
    
    // Check if price dropped >= 1 ATR from entry
    let price_drop = hedge_entry_price - close;
    let drop_in_atr = price_drop / atr_at_entry;
    
    if drop_in_atr >= bp.hedge_breakeven_atr {
        Some(hedge_entry_price) // Move SL to entry (breakeven)
    } else {
        None
    }
}

/// Checks if sizes are desynchronized
/// Pure function: current sizes → (bool, diff_pct)
pub fn are_sizes_desynchronized(
    long_size: f64,
    hedge_size: f64,
    tolerance_pct: f64,
) -> (bool, f64) {
    if long_size <= 0.0 {
        return (false, 0.0);
    }
    
    let difference = (long_size - hedge_size).abs();
    let diff_pct = difference / long_size;
    
    (diff_pct > tolerance_pct, diff_pct)
}

