# Get experiment source texts in paper
python -m src.generate_source_text \
    --raw_source_text_dir raw_data/source_text/ \
    --output_source_text_dir local_data/source_text/ \
    --config paper_experiment
