export model_path=/home/sankuai/dolphinfs_xiangjingyang/huggingface.co/meta-llama/Llama-2-7b-hf
export tokenizer_path=${model_path}
export save_path=./results
python3 examples/quant_model.py \
--model_path ${model_path} \
--tokenizer_path ${tokenizer_path} \
--batch_size 8 \
--dtype float16 \
--quant_config quant_config/llama/w4a8.yaml \
--save_path ${save_path}

export quantized_model_path=./results/Llama-2-7b-hf
python3 examples/eval_model.py \
--model_path ${quantized_model_path} \
--tokenizer_path ${tokenizer_path} \
--tasks piqa,winogrande,hellaswag,arc_challenge,arc_easy \
--batch_size 8 \
--max_length 2048 \
 --eval_ppl