/// Adaptive Trailing Grid Ratio Implementation
/// 
/// This module calculates a dynamic trailing_grid_ratio based on price distance from EMA.
/// During market crashes (price far below EMA), it switches from grid orders (DCA) to 
/// trailing orders (waiting for the bottom).

/// EMA state for adaptive distance calculation
#[derive(Debug, Clone)]
pub struct EMAState {
    pub span: f64,
    pub alpha: f64,
    pub value: f64,
    pub numerator: f64,
    pub denominator: f64,
}

impl EMAState {
    pub fn new(span: f64) -> Self {
        let alpha = 2.0 / (span + 1.0);
        EMAState {
            span,
            alpha,
            value: 0.0,
            numerator: 0.0,
            denominator: 0.0,
        }
    }

    /// Update EMA with new price value using adjusted EMA formula
    pub fn update(&mut self, price: f64) {
        if !price.is_finite() || price <= 0.0 {
            return;
        }

        if self.denominator == 0.0 {
            // First value, initialize
            self.numerator = self.alpha * price;
            self.denominator = self.alpha;
            self.value = price;
        } else {
            let one_minus_alpha = 1.0 - self.alpha;
            self.numerator = self.alpha * price + one_minus_alpha * self.numerator;
            self.denominator = self.alpha + one_minus_alpha * self.denominator;
            
            if self.denominator > 0.0 {
                self.value = self.numerator / self.denominator;
            }
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.denominator > 0.0
    }
}

/// State for adaptive trailing ratio calculation
#[derive(Debug, Clone)]
pub struct AdaptiveTrailingState {
    /// EMA state for adaptive distance calculation
    pub ema_state: EMAState,
    /// Smoothed trailing ratio value (if smoothing enabled)
    pub smoothed_ratio: Option<f64>,
}

impl AdaptiveTrailingState {
    pub fn new(span_minutes: f64) -> Self {
        AdaptiveTrailingState {
            ema_state: EMAState::new(span_minutes),
            smoothed_ratio: None,
        }
    }
}

/// Calculate sigmoid function for smooth transition
/// 
/// Formula: trailing_ratio = 1 / (1 + e^(k * (distance + offset)))
/// 
/// # Arguments
/// * `distance` - Price distance from EMA: (price - ema) / ema
/// * `steepness` - Controls transition sharpness (k parameter)
/// * `offset` - Distance at which trailing_ratio = 0.5
/// 
/// # Returns
/// Value between 0.0 and 1.0:
/// - 0.0 = 100% grid (normal DCA)
/// - 1.0 = 100% trailing (survival mode)
fn sigmoid(distance: f64, steepness: f64, offset: f64) -> f64 {
    let exponent = steepness * (distance + offset);
    
    // Handle numerical overflow for extreme distances
    if exponent > 20.0 {
        return 0.0;
    } else if exponent < -20.0 {
        return 1.0;
    }
    
    1.0 / (1.0 + exponent.exp())
}

/// Calculate adaptive trailing grid ratio
/// 
/// # Arguments
/// * `price` - Current price
/// * `state` - Adaptive trailing state (EMA and smoothing)
/// * `steepness` - Sigmoid steepness parameter
/// * `offset` - Sigmoid offset parameter
/// * `smoothing_enabled` - Whether to apply EMA smoothing
/// * `smoothing_span` - Smoothing window in minutes
/// 
/// # Returns
/// Trailing ratio between 0.0 and 1.0
pub fn calculate_adaptive_trailing_ratio(
    price: f64,
    state: &mut AdaptiveTrailingState,
    steepness: f64,
    offset: f64,
    smoothing_enabled: bool,
    smoothing_span: f64,
) -> f64 {
    // Update EMA with current price
    state.ema_state.update(price);
    
    let ema = state.ema_state.value;
    
    // Handle edge case: EMA not initialized or zero
    if ema == 0.0 || !ema.is_finite() {
        return 0.0; // Fallback to grid mode
    }
    
    // Calculate distance: (price - ema) / ema
    let mut distance = (price - ema) / ema;
    
    // Clamp extreme values
    distance = distance.max(-1.0).min(1.0);
    
    // Calculate raw trailing ratio using sigmoid
    // let ratio_raw = sigmoid(distance, steepness, offset);
    let ratio_raw = if distance.abs() > offset { 1.0 } else { 0.0 };
    
    // Apply smoothing if enabled
    let ratio = if smoothing_enabled {
        let alpha = 2.0 / (smoothing_span + 1.0);
        
        match state.smoothed_ratio {
            None => {
                // First calculation, initialize smoothed value
                state.smoothed_ratio = Some(ratio_raw);
                ratio_raw
            }
            Some(prev_smoothed) => {
                // Apply EMA smoothing
                let smoothed = prev_smoothed * (1.0 - alpha) + ratio_raw * alpha;
                state.smoothed_ratio = Some(smoothed);
                smoothed
            }
        }
    } else {
        ratio_raw
    };
    
    // Ensure result is in valid range
    ratio.max(0.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_normal_market() {
        // Price above EMA (+5%) -> should be close to 0 (grid mode)
        let distance = 0.05;
        let ratio = sigmoid(distance, 30.0, 0.12);
        assert!(ratio < 0.01, "Expected ratio < 0.01 for +5% distance, got {}", ratio);
    }

    #[test]
    fn test_sigmoid_at_ema() {
        // Price at EMA (0%) -> should be very small
        let distance = 0.0;
        let ratio = sigmoid(distance, 30.0, 0.12);
        assert!(ratio < 0.05, "Expected ratio < 0.05 at EMA, got {}", ratio);
    }

    #[test]
    fn test_sigmoid_inflection_point() {
        // Price at offset (-12%) -> should be ~0.5
        let distance = -0.12;
        let ratio = sigmoid(distance, 30.0, 0.12);
        assert!((ratio - 0.5).abs() < 0.01, "Expected ratio ~0.5 at inflection, got {}", ratio);
    }

    #[test]
    fn test_sigmoid_severe_crash() {
        // Price far below EMA (-30%) -> should be close to 1 (trailing mode)
        let distance = -0.30;
        let ratio = sigmoid(distance, 30.0, 0.12);
        assert!(ratio > 0.99, "Expected ratio > 0.99 for -30% distance, got {}", ratio);
    }

    #[test]
    fn test_sigmoid_overflow_protection() {
        // Test extreme positive distance
        let ratio_pos = sigmoid(10.0, 30.0, 0.12);
        assert_eq!(ratio_pos, 0.0, "Should return 0 for extreme positive distance");
        
        // Test extreme negative distance
        let ratio_neg = sigmoid(-10.0, 30.0, 0.12);
        assert_eq!(ratio_neg, 1.0, "Should return 1 for extreme negative distance");
    }

    #[test]
    fn test_adaptive_trailing_ratio_basic() {
        let mut state = AdaptiveTrailingState::new(60.0);
        
        // Feed some prices to initialize EMA around 100
        for _ in 0..100 {
            state.ema_state.update(100.0);
        }
        
        // Test price 10% below EMA
        let ratio = calculate_adaptive_trailing_ratio(
            90.0,
            &mut state,
            30.0,
            0.12,
            false,
            10.0,
        );
        
        // Should be around 0.45 (moderate trailing)
        assert!(ratio > 0.3 && ratio < 0.6, "Expected ratio 0.3-0.6 for -10% distance, got {}", ratio);
    }

    #[test]
    fn test_adaptive_trailing_with_smoothing() {
        let mut state = AdaptiveTrailingState::new(60.0);
        
        // Initialize EMA
        for _ in 0..100 {
            state.ema_state.update(100.0);
        }
        
        // Calculate ratio with smoothing
        let ratio1 = calculate_adaptive_trailing_ratio(
            85.0,
            &mut state,
            30.0,
            0.12,
            true,
            10.0,
        );
        
        // Second calculation should be smoothed
        let ratio2 = calculate_adaptive_trailing_ratio(
            85.0,
            &mut state,
            30.0,
            0.12,
            true,
            10.0,
        );
        
        // Both should be valid and similar (same price)
        assert!(ratio1 > 0.0 && ratio1 <= 1.0);
        assert!(ratio2 > 0.0 && ratio2 <= 1.0);
    }

    #[test]
    fn test_ema_zero_protection() {
        let mut state = AdaptiveTrailingState::new(60.0);
        
        // EMA not initialized (will be 0.0)
        let ratio = calculate_adaptive_trailing_ratio(
            100.0,
            &mut state,
            30.0,
            0.12,
            false,
            10.0,
        );
        
        // Should fallback to 0.0 (grid mode) when EMA is invalid
        assert_eq!(ratio, 0.0, "Should return 0 when EMA is not initialized");
    }
}
