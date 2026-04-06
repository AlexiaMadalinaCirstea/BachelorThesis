import pandas as pd
import os
import argparse


def load_data(train_path, test_path):
    print("Loading training data...")
    train_df = pd.read_csv(train_path)

    print("Loading testing data...")
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def inspect_and_save(df, name, out_dir):
    output_file = os.path.join(out_dir, f"{name}_inspection.txt")

    with open(output_file, "w") as f:
        f.write(f"===== {name.upper()} INFO =====\n")
        f.write(f"Shape: {df.shape}\n\n")

        f.write("Dtypes:\n")
        f.write(str(df.dtypes))
        f.write("\n\n")

        f.write("Missing values:\n")
        f.write(str(df.isnull().sum().sort_values(ascending=False)))
        f.write("\n\n")

        f.write("Label distribution:\n")
        f.write(str(df['label'].value_counts()))
        f.write("\n")

    print(f"Saved inspection to {output_file}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_df, test_df = load_data(args.train_path, args.test_path)

    inspect_and_save(train_df, "train", args.out_dir)
    inspect_and_save(test_df, "test", args.out_dir)


if __name__ == "__main__":
    main()