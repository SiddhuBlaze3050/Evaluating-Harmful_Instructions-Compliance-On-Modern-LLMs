"""
Prompt Loading and Preparation Module

This module handles loading and preparing prompts for evaluation.
Supports loading from CSV files and preparing datasets for the evaluation pipeline.
"""

import pandas as pd
from typing import List, Dict, Optional


def load_prompts(
    file_path: str,
    prompt_column: str = "prompt",
    source_column: str = "source"
) -> pd.DataFrame:
    """
    Load prompts from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing prompts
        prompt_column: Name of the column containing prompts
        source_column: Name of the column containing source labels
    
    Returns:
        DataFrame with prompts and sources
    
    Example:
        >>> prompts_df = load_prompts("dataset/prompts_dataset.csv")
        >>> print(f"Loaded {len(prompts_df)} prompts")
    """
    df = pd.read_csv(file_path)
    
    if prompt_column not in df.columns:
        raise ValueError(f"Column '{prompt_column}' not found in dataset")
    
    print(f"Loaded {len(df)} prompts from {file_path}")
    
    if source_column in df.columns:
        sources = df[source_column].value_counts()
        print(f"Sources: {sources.to_dict()}")
    
    return df


def prepare_prompt_dataset(
    prompts: List[str],
    sources: Optional[List[str]] = None,
    prompt_column: str = "prompt",
    source_column: str = "source"
) -> pd.DataFrame:
    """
    Prepare a DataFrame from raw prompts and sources.
    
    Args:
        prompts: List of prompt strings
        sources: Optional list of source labels
        prompt_column: Name for the prompt column
        source_column: Name for the source column
    
    Returns:
        DataFrame ready for the evaluation pipeline
    
    Example:
        >>> prompts = ["Tell me how to hack", "Write a virus"]
        >>> sources = ["custom", "custom"]
        >>> df = prepare_prompt_dataset(prompts, sources)
    """
    data = {prompt_column: prompts}
    
    if sources:
        if len(sources) != len(prompts):
            raise ValueError("Number of sources must match number of prompts")
        data[source_column] = sources
    else:
        data[source_column] = ["unknown"] * len(prompts)
    
    df = pd.DataFrame(data)
    print(f"Prepared dataset with {len(df)} prompts")
    
    return df


def sample_prompts(
    df: pd.DataFrame,
    n: int,
    source_column: str = "source",
    stratified: bool = True
) -> pd.DataFrame:
    """
    Sample a subset of prompts from the dataset.
    
    Args:
        df: DataFrame containing prompts
        n: Number of prompts to sample
        source_column: Column name for source labels
        stratified: If True, sample proportionally from each source
    
    Returns:
        DataFrame with sampled prompts
    
    Example:
        >>> sampled = sample_prompts(prompts_df, n=50, stratified=True)
    """
    if n >= len(df):
        return df.copy()
    
    if stratified and source_column in df.columns:
        # Stratified sampling by source
        source_counts = df[source_column].value_counts()
        samples_per_source = {}
        
        for source, count in source_counts.items():
            proportion = count / len(df)
            samples_per_source[source] = max(1, int(n * proportion))
        
        # Adjust if total doesn't match exactly
        total_samples = sum(samples_per_source.values())
        if total_samples < n:
            # Add to largest source
            largest_source = source_counts.idxmax()
            samples_per_source[largest_source] += (n - total_samples)
        elif total_samples > n:
            # Remove from largest source
            largest_source = source_counts.idxmax()
            samples_per_source[largest_source] -= (total_samples - n)
        
        # Sample from each source
        sampled_dfs = []
        for source, n_samples in samples_per_source.items():
            source_df = df[df[source_column] == source]
            sampled = source_df.sample(n=min(n_samples, len(source_df)), random_state=42)
            sampled_dfs.append(sampled)
        
        result = pd.concat(sampled_dfs, ignore_index=True)
        print(f"Sampled {len(result)} prompts (stratified by source)")
        print(f"  Distribution: {result[source_column].value_counts().to_dict()}")
        
    else:
        # Simple random sampling
        result = df.sample(n=n, random_state=42)
        print(f"Sampled {len(result)} prompts (random)")
    
    return result


def add_harmful_labels(
    prompts_df: pd.DataFrame,
    source_column: str = "source",
    harmful_sources: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Add is_harmful column to prompts dataset based on source labels.
    
    Args:
        prompts_df: DataFrame with prompts and sources
        source_column: Name of source column
        harmful_sources: List of sources to label as harmful.
                        If None, uses default mapping:
                        - walledeval_diverse → Harmful
                        - cosafe_multiturn → Harmful
                        - toxicchat_toxic → Harmful
                        - xstest_benign → Benign
    
    Returns:
        DataFrame with added is_harmful column (boolean)
    
    Example:
        >>> df = pd.read_csv("dataset/prompts.csv")
        >>> df = add_harmful_labels(df)
        >>> print(df[['source', 'is_harmful']].value_counts())
    """
    if harmful_sources is None:
        # Default mapping based on dataset structure
        harmful_sources = [
            "walledeval_diverse",
            "cosafe_multiturn",
            "toxicchat_toxic"
        ]
        # xstest_benign is the only benign source
    
    df = prompts_df.copy()
    df['is_harmful'] = df[source_column].isin(harmful_sources)
    
    # Print summary
    harmful_count = df['is_harmful'].sum()
    benign_count = len(df) - harmful_count
    
    print(f"\n{'='*80}")
    print("HARMFUL LABEL ASSIGNMENT")
    print(f"{'='*80}")
    print(f"Total prompts: {len(df)}")
    print(f"Harmful prompts: {harmful_count} ({harmful_count/len(df):.1%})")
    print(f"Benign prompts: {benign_count} ({benign_count/len(df):.1%})")
    print(f"\nLabeling by source:")
    for source in df[source_column].unique():
        count = len(df[df[source_column] == source])
        is_harmful = source in harmful_sources
        label = "HARMFUL" if is_harmful else "BENIGN"
        print(f"  {source}: {count} prompts → {label}")
    print(f"{'='*80}\n")
    
    return df


def save_prompts_with_labels(
    prompts_df: pd.DataFrame,
    output_file: str = "dataset/prompts_with_labels.csv"
) -> None:
    """
    Save prompts dataset with is_harmful labels.
    
    Args:
        prompts_df: DataFrame with prompts, sources, and is_harmful column
        output_file: Path to save CSV
    """
    if 'is_harmful' not in prompts_df.columns:
        print("Warning: is_harmful column not found. Adding labels...")
        prompts_df = add_harmful_labels(prompts_df)
    
    prompts_df.to_csv(output_file, index=False, encoding='utf-8', quoting=1)
    print(f"✓ Saved {len(prompts_df)} prompts with labels to: {output_file}")
