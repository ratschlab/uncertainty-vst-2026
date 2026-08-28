# src/models/registry.py
#
# Per-model classification, used to route Snakemake rules.
#
# This file has no imports, and no torch in particular, because the Snakefile imports it
# and the Snakefile runs in an environment without torch. The full ModelSpec, with the
# model class, the loss and the prediction extraction, lives in specs.py, which reads
# MODEL_KINDS from here.
#
# Kinds:
#   deterministic   mean only, no lambda, no predictive variance
#   gaussian_lam    Gaussian NLL, output (mean, log_var); lambda is the beta-NLL exponent
#   prob_no_lam     probabilistic, no lambda in the loss (Poisson, NB)
#
# A kind in LAMBDA_KINDS gets a /lambda_{lam}/ path segment and a lambda sweep.
# To add a model: one line here, and a ModelSpec entry in specs.py.

MODEL_KINDS = {
    "msemeanonly":             "deterministic",
    "homoscedastic":           "gaussian_lam",
    "heteroscedastic":         "gaussian_lam",
    "spot_homoscedastic":      "gaussian_lam",  # one variance per spot, shared across genes
    # Common-depth count variants, target cp_counts. The depth is one constant per dataset,
    # datasets.{ds}.count_common_depth in the config.
    "poisson_cp":                          "prob_no_lam",
    "negativebinomial_hetero_noanchor_cp": "prob_no_lam",
    # Restricted dispersion: one alpha per spot, or one per gene. 
    "negativebinomial_spotdisp_cp":        "prob_no_lam",
    "negativebinomial_genedisp_cp":        "prob_no_lam",
    # Depth-blind variants, fitted directly on the raw integer counts.
    "poisson_raw":                          "prob_no_lam",
    "negativebinomial_hetero_noanchor_raw": "prob_no_lam",
    # Unrounded cp variants: the same discrete predictive against a continuous target.
    "poisson_cp_nr":                          "prob_no_lam",
    "negativebinomial_hetero_noanchor_cp_nr": "prob_no_lam",
    "negativebinomial_spotdisp_cp_nr":        "prob_no_lam",
    "negativebinomial_genedisp_cp_nr":        "prob_no_lam",
    # STimage benchmark: a bare Linear head with no hidden layer, matching STimage's
    # per-gene Dense(2). See the STI_nb_* block in specs.py.
    "STI_nb_cp":          "prob_no_lam",   # cp_counts target
    "STI_nb_raw":         "prob_no_lam",   # raw_counts target, its own reporting axis
}

# Kinds whose output path carries a /lambda_{lam}/ segment.
LAMBDA_KINDS    = ("gaussian_lam",)
NO_LAMBDA_KINDS = ("deterministic", "prob_no_lam")

# Which config list supplies the lambda sweep for a model_type.
_LAMBDA_LIST_KEY = {
    "gaussian_lam": "lambda_values",
}

# Every model_type fitted with a count likelihood, Poisson or NB. Targets are raw_counts,
# cp_counts or cp_counts_nr; the first two are integers, cp_counts_nr is continuous.
COUNT_MODEL_TYPES = frozenset({
    "poisson_cp", "negativebinomial_hetero_noanchor_cp",
    "negativebinomial_spotdisp_cp", "negativebinomial_genedisp_cp",
    "poisson_cp_nr", "negativebinomial_hetero_noanchor_cp_nr",
    "negativebinomial_spotdisp_cp_nr", "negativebinomial_genedisp_cp_nr",
    "poisson_raw", "negativebinomial_hetero_noanchor_raw",
    "STI_nb_cp", "STI_nb_raw",
})


def is_count_model(model_type):
    """True if this model_type fits a count likelihood on a count target."""
    return model_type in COUNT_MODEL_TYPES


def kind_of(model_type):
    """Kind string for a model_type. Raises KeyError if it is unknown."""
    if model_type not in MODEL_KINDS:
        raise KeyError(
            f"Unknown model_type '{model_type}'. "
            f"Known: {sorted(MODEL_KINDS)}"
        )
    return MODEL_KINDS[model_type]


def types_of_kind(kind):
    """All model_types of one kind, in registration order."""
    return [mt for mt, k in MODEL_KINDS.items() if k == kind]


def uses_lambda(model_type):
    """True if this model_type's output path carries a /lambda_{lam}/ segment."""
    return kind_of(model_type) in LAMBDA_KINDS


def lambda_list_key(model_type):
    """Config key holding the lambda sweep for a lambda-bearing model."""
    return _LAMBDA_LIST_KEY[kind_of(model_type)]
