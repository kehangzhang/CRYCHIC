"""Reviewed configuration symbols and producer-contract adapters."""

from typing import cast

from crychic.core.config import CrychicConfig
from crychic.core.enums import CommunicationMode
from crychic.data import ExpressionTransform, InputSchema


def input_schema_from_config(config: CrychicConfig) -> InputSchema:
    """Adapt the flat public configuration to the data-owned input contract."""

    if not isinstance(config, CrychicConfig):
        raise TypeError("config must be a CrychicConfig instance")
    return InputSchema(
        context_keys=tuple(config.context_keys),
        counts_layer=config.counts_layer,
        sample_key=config.sample_key,
        subject_key=config.subject_key,
        cell_type_key=config.cell_type_key,
        covariates=tuple(config.covariates),
        expression_layer=config.expression_layer,
        expression_source=config.expression_source,
        expression_transform=cast(
            ExpressionTransform | None, config.expression_transform
        ),
        normalized_zero_is_nondetection=(config.normalized_zero_is_nondetection),
        species=config.species,
        gene_namespace=config.gene_namespace,
        allow_duplicate_genes=config.allow_duplicate_genes,
    )


__all__ = [
    "CommunicationMode",
    "CrychicConfig",
    "ExpressionTransform",
    "InputSchema",
    "input_schema_from_config",
]
