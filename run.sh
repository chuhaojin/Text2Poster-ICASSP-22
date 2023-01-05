example_id=2
python main.py \
  --input_text_file example/input_text_elements_$example_id.json \
  --output_folder example/outputs_$example_id \
  --background_folder bk_image_folder \
  --top_n 5 \
  --save_process
