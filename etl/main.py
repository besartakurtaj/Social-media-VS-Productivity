from extract import extract_data
from transform import transform_data
from load import load_data

if __name__ == "__main__":
    input_file = "../data/social_media_vs_productivity.csv"
    output_file = "../data/processed_dataset.csv"

    df = extract_data(input_file)

    df_transformed = transform_data(df)

    load_data(df_transformed, output_file)
