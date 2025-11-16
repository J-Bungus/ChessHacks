import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(
    input_csv="pairs.csv",
    train_csv="train.csv",
    val_csv="validation.csv",
    val_ratio=0.1,
    seed=42
):
    print("Loading dataset...")
    df = pd.read_csv(input_csv)

    print(f"Total rows: {len(df):,}")

    # Split the dataset
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )

    print(f"Training rows:   {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")

    # Save
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(f"\nSaved:")
    print(f" - {train_csv}")
    print(f" - {val_csv}")
    print("Done.")

if __name__ == "__main__":
    split_dataset(
        input_csv="pairs.csv",     # your original dataset
        train_csv="train.csv",
        val_csv="validation.csv",
        val_ratio=0.1,             # 10% validation
        seed=1337                  # reproducible split
    )
