"""Validation shared by CLI and programmatic adapter construction.

Execution choices do not change checkpoint compatibility or model parameters.
"""


def validate_execution_backends(args):
    compute = getattr(args, "adapter_compute", "dense")
    attention = getattr(args, "attention_backend", "manual")
    if compute not in {"dense", "factorized"}:
        raise ValueError("adapter_compute must be 'dense' or 'factorized'")
    if attention not in {"manual", "sdpa"}:
        raise ValueError("attention_backend must be 'manual' or 'sdpa'")
    if compute != "dense" or attention != "manual":
        if args.adapter != "ohsinglora":
            raise ValueError("Nondefault execution backends require --adapter ohsinglora")
        if not set(getattr(args, "params", ["q", "k", "v"])).issubset({"q", "k", "v"}):
            raise ValueError("Nondefault execution backends support square q/k/v projections only (--params q k v)")

