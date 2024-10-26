# Get experiment source texts in paper
python -m src.generate_dataset \
    --config paper_experiment \
    --exp_dir local_data/ \
    --query_path local_data/query/query_orig_rewrite.json \
    --source_text_path local_data/source_text/source_texts_paper_experiment.json \
    --notice_dir raw_data/notice

