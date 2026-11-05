/// Test statistic type for Model Confidence Set
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MCSStatistic {
    /// T_R statistic: `max_{i,j} |t_{ij}|` over the pairwise t-ratios of the
    /// surviving models (R `MCSprocedure(statistic = "TR")`).
    Range,
    /// T_max statistic: `max_i t_{i.}`, the largest t-ratio of a model's loss
    /// against the average loss of the surviving models
    /// (R `MCSprocedure(statistic = "Tmax")`).
    Max,
}

/// Information about a single elimination step
#[derive(Debug, Clone)]
pub struct MCSEliminationStep {
    /// The model index that was tested at this step (the worst surviving model)
    pub model_idx: usize,
    /// The p-value of the equal predictive ability test at this step
    pub p_value: f64,
    /// The MCS p-value of `model_idx`: the running maximum of the step
    /// p-values (Hansen, Lunde & Nason, 2011)
    pub mcs_p_value: f64,
    /// Whether the model was eliminated (`mcs_p_value < alpha`)
    pub eliminated: bool,
}

/// Result of Model Confidence Set procedure
#[derive(Debug, Clone)]
pub struct MCSResult {
    /// Indices of models in the Model Confidence Set (ascending)
    pub included_models: Vec<usize>,
    /// Indices of models eliminated from the MCS, in elimination order
    pub eliminated_models: Vec<usize>,
    /// The MCS p-value at which elimination stopped (the MCS p-value of the
    /// first retained step; the last step's if all but one model were
    /// eliminated)
    pub mcs_p_value: f64,
    /// Elimination steps up to and including the first step whose model is
    /// retained
    pub elimination_sequence: Vec<MCSEliminationStep>,
    /// The full elimination order (worst model first, the best model last),
    /// R's `elimination.order`
    pub elimination_order: Vec<usize>,
    /// MCS p-value of every model, indexed by model (the best model has 1),
    /// R's `MCS p-Value` column; model `i` is in the MCS iff
    /// `mcs_p_values[i] >= alpha`
    pub mcs_p_values: Vec<f64>,
    /// Number of bootstrap samples used
    pub n_bootstrap: usize,
    /// The statistic type used
    pub statistic_type: MCSStatistic,
}
