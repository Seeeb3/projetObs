"""Load embeddings and encode INDUS inputs for the audit."""

from __future__ import annotations

import click
import numpy as np
import polars as pl

from tools.analysis.embedding_audit_models import (
    EMBEDDING_REQUIRED_COLUMNS,
    INDUS_METADATA_REQUIRED_COLUMNS,
    PRECOMPUTED_METADATA_REQUIRED_COLUMNS,
    EmbeddingSource,
    HeliophysicsEmbeddingAuditConfig,
    load_metadata_dataframe,
    validate_required_columns,
    validate_unique_bibcodes,
)


def build_precomputed_audit_dataframe(
    config: HeliophysicsEmbeddingAuditConfig,
) -> pl.DataFrame:
    """Build the joined audit dataframe from precomputed embeddings.

    Args:
        config: Audit configuration.

    Returns:
        Joined dataframe combining precomputed embeddings and metadata.
    """

    if config.embeddings_parquet is None:
        raise ValueError(
            "embeddings_parquet is required for precomputed embedding mode"
        )

    embeddings_dataframe = pl.read_parquet(config.embeddings_parquet)

    metadata_dataframe = load_metadata_dataframe(
        config.metadata_csv,
        PRECOMPUTED_METADATA_REQUIRED_COLUMNS,
    )

    validate_required_columns(
        embeddings_dataframe,
        EMBEDDING_REQUIRED_COLUMNS,
        "embeddings parquet",
    )

    validate_unique_bibcodes(embeddings_dataframe, "embeddings parquet")

    joined_dataframe = (
        embeddings_dataframe.rename({"keyword_label": "embedding_keyword_label"})
        .join(metadata_dataframe, on="bibcode", how="inner")
        .select(
            [
                "bibcode",
                "embedding_keyword_label",
                "keyword_label",
                "embedding_backend",
                "embedding_dim",
                "embedding",
                "title",
                "keywords",
                "matched_positive_rules",
            ]
        )
    )

    if joined_dataframe.height != config.expected_row_count:
        raise ValueError(
            "Joined audit dataframe row count mismatch: "
            f"expected {config.expected_row_count}, found {joined_dataframe.height}"
        )

    mismatched_labels = joined_dataframe.filter(
        pl.col("embedding_keyword_label") != pl.col("keyword_label")
    )

    if mismatched_labels.height > 0:
        mismatch_preview = ", ".join(
            mismatched_labels.select("bibcode").get_column("bibcode").head(10).to_list()
        )
        raise ValueError(
            "Embedding labels do not match metadata labels for bibcodes: "
            f"{mismatch_preview}"
        )

    validate_unique_bibcodes(joined_dataframe, "joined audit dataframe")

    return joined_dataframe.drop("embedding_keyword_label")


def prepare_indus_source_dataframe(metadata_dataframe: pl.DataFrame) -> pl.DataFrame:
    """Prepare the metadata rows used as INDUS text inputs.

    Args:
        metadata_dataframe: Metadata dataframe containing abstracts.

    Returns:
        Filtered dataframe with one non-empty abstract text per bibcode.
    """

    return (
        metadata_dataframe.with_columns(
            pl.col("abstract").fill_null("").str.strip_chars().alias("indus_text")
        )
        .filter(pl.col("indus_text") != "")
        .select(
            [
                "bibcode",
                "keyword_label",
                "title",
                "keywords",
                "matched_positive_rules",
                "indus_text",
            ]
        )
    )


def mean_pool_last_hidden_state(
    last_hidden_state: "torch.Tensor",
    attention_mask: "torch.Tensor",
) -> "torch.Tensor":
    """Mean-pool token embeddings with an attention mask.

    Args:
        last_hidden_state: Model token embeddings.
        attention_mask: Attention mask with ones on valid tokens.

    Returns:
        Mean-pooled sentence embeddings.
    """

    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)

    masked_hidden_state = last_hidden_state * mask

    token_count = mask.sum(dim=1).clamp(min=1.0)

    return masked_hidden_state.sum(dim=1) / token_count


def encode_texts_with_indus(
    config: HeliophysicsEmbeddingAuditConfig, texts: list[str]
) -> np.ndarray:
    """Encode texts with the cached INDUS transformer model.

    Args:
        config: Audit configuration.
        texts: Input texts to encode.

    Returns:
        Dense float32 embedding matrix.
    """

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.indus_model_id,
        local_files_only=True,
    )

    model = AutoModel.from_pretrained(
        config.indus_model_id,
        local_files_only=True,
        torch_dtype=torch.float32,
    )

    model.eval()

    model.to("cpu")

    embedding_batches: list[np.ndarray] = []
    total_batches = (
        len(texts) + config.indus_batch_size - 1
    ) // config.indus_batch_size

    with torch.inference_mode():
        for batch_start in range(0, len(texts), config.indus_batch_size):
            batch_texts = texts[batch_start : batch_start + config.indus_batch_size]
            batch_index = (batch_start // config.indus_batch_size) + 1

            if (
                batch_index == 1
                or batch_index % 10 == 0
                or batch_index == total_batches
            ):
                click.echo(
                    f"[*] INDUS encoding batch {batch_index}/{total_batches} "
                    f"({len(batch_texts)} texts)..."
                )

            encoded_inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.indus_max_length,
                return_tensors="pt",
            )

            model_outputs = model(**encoded_inputs)

            pooled_embeddings = mean_pool_last_hidden_state(
                last_hidden_state=model_outputs.last_hidden_state,
                attention_mask=encoded_inputs["attention_mask"],
            )

            normalized_embeddings = F.normalize(
                pooled_embeddings.to(dtype=torch.float32),
                p=2,
                dim=1,
            )

            embedding_batches.append(normalized_embeddings.cpu().numpy())

    if not embedding_batches:
        raise ValueError("INDUS encoding produced no embedding batches")

    embedding_matrix = np.vstack(embedding_batches).astype(np.float32, copy=False)

    if not np.isfinite(embedding_matrix).all():
        raise ValueError("INDUS encoding produced non-finite values")

    return embedding_matrix


def build_indus_audit_dataframe(
    config: HeliophysicsEmbeddingAuditConfig,
) -> pl.DataFrame:
    """Build the audit dataframe by encoding abstracts with INDUS.

    Args:
        config: Audit configuration.

    Returns:
        Audit dataframe with freshly encoded INDUS embeddings.
    """

    metadata_dataframe = load_metadata_dataframe(
        config.metadata_csv,
        INDUS_METADATA_REQUIRED_COLUMNS,
    )

    indus_source_dataframe = prepare_indus_source_dataframe(metadata_dataframe)

    if indus_source_dataframe.height != config.expected_row_count:
        raise ValueError(
            "INDUS source dataframe row count mismatch: "
            f"expected {config.expected_row_count}, found {indus_source_dataframe.height}"
        )

    click.echo(
        f"[*] Encoding {indus_source_dataframe.height} abstracts with cached INDUS model "
        f"{config.indus_model_id}..."
    )

    embedding_matrix = encode_texts_with_indus(
        config,
        indus_source_dataframe.get_column("indus_text").to_list(),
    )

    embedding_backend = f"local_transformers_mean_pool:{config.indus_model_id}"

    return indus_source_dataframe.with_columns(
        [
            pl.lit(embedding_backend).alias("embedding_backend"),
            pl.lit(int(embedding_matrix.shape[1])).alias("embedding_dim"),
            pl.Series("embedding", embedding_matrix.tolist()),
        ]
    ).drop("indus_text")


def build_audit_dataframe(config: HeliophysicsEmbeddingAuditConfig) -> pl.DataFrame:
    """Build the source dataframe for the selected embedding source.

    Args:
        config: Audit configuration.

    Returns:
        Audit dataframe ready for projection and metric computation.
    """

    if config.embedding_source == EmbeddingSource.PRECOMPUTED:
        return build_precomputed_audit_dataframe(config)
    if config.embedding_source == EmbeddingSource.INDUS_CACHED:
        return build_indus_audit_dataframe(config)
    raise ValueError(f"Unsupported embedding source: {config.embedding_source}")
