import pandas as pd


def build_merged_dataset(
    spam_csv_path="data/spam.csv",
    emails_csv_path="data/emails.csv",
    output_path="data/dataset.csv",
):
    """
    Loads spam.csv and emails.csv, standardizes both to a common
    ['label', 'text'] format (label = 'ham'/'spam'), merges them,
    and SAVES the result to output_path as the single source-of-truth
    dataset for the rest of the pipeline.
    """
    # --- Load spam.csv (columns: v1=label, v2=text) ---
    df1 = pd.read_csv(spam_csv_path, encoding="latin-1")
    df1 = df1[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    df1["label"] = df1["label"].str.strip().str.lower()

    # --- Load emails.csv (columns: text, spam=1/0) ---
    df2 = pd.read_csv(emails_csv_path)
    df2 = df2[["text", "spam"]].rename(columns={"spam": "label"})
    df2["label"] = df2["label"].map({1: "spam", 0: "ham"})
    # strip the leading "Subject:" that appears on every row, so the text
    # format matches spam.csv (which has no such prefix)
    df2["text"] = df2["text"].str.replace(r"(?i)^\s*subject\s*:\s*", "", regex=True)

    # --- Merge ---
    merged = pd.concat([df1[["label", "text"]], df2[["label", "text"]]], ignore_index=True)

    # basic cleanup
    merged["text"] = merged["text"].astype(str).str.strip()
    merged = merged.dropna(subset=["label", "text"])
    merged = merged[merged["text"] != ""]

    # drop exact duplicate rows (same label + text)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["label", "text"]).reset_index(drop=True)
    removed = before - len(merged)

    print(f"spam.csv:   {len(df1)} rows")
    print(f"emails.csv: {len(df2)} rows")
    print(f"Merged (pre-dedup): {before} rows")
    print(f"Removed {removed} duplicate rows")
    print(f"Final merged dataset: {len(merged)} rows")
    print(merged["label"].value_counts())

    merged.to_csv(output_path, index=False)
    print(f"Saved merged dataset to {output_path}")

    return merged


def load_dataset(path="data/dataset.csv"):
    """Loads the final merged dataset (run build_merged_dataset() once first)."""
    return pd.read_csv(path)


if __name__ == "__main__":
    build_merged_dataset()