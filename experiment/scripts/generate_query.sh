# # Get original queries
# python -m src.generate_query \
#     --raw_query_dir raw_data/query/ \
#     --output_query_dir local_data/query/ \
#     --config orig \

# Get rewritten queries
python -m src.generate_query \
    --raw_query_dir raw_data/query/ \
    --output_query_dir local_data/query/ \
    --config orig_rewrite \
    --rewrite_cred_filename openai_cred.txt \
    --n_rewrite 10