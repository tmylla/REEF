
models=(
    'llama-2-7b'
    'llama-2-finance-7b' 'vicuna-7b-v1.5' 'wizardmath-7b' 'chinese-llama-2-7b' 'codellama-7b' 'llemma-7b'
    'Sheared-LLaMA-1.3B-Pruned' 'Sheared-LLaMA-1.3B' 'Sheared-LLaMA-1.3B-ShareGPT'
    'Sheared-LLaMA-2.7B-Pruned' 'Sheared-LLaMA-2.7B' 'Sheared-LLaMA-2.7B-ShareGPT'
    'wandallama-2-7b' 'gblmllama-2-7b' 'sparsellama-2-7b'
    'evollm-jp-7b' 'shisa-gamma-7b' 'wizardmath-7b-1.1' 'abel-7b-002'
    'fusellm-7b' "llama-2-7b" 'openllama-2-7b' 'mpt-7b'
    'xwinlm-7b'
    'llama-3-8b' 'amber' 'internlm-7b'
    'llama-2-13b' 'llama-2-13b-chat' 'vicuna-13b' 'xwinlm-13b' 'chinesellama-2-13b'
    'plamo-13b' 'baichuan-2-13b' 'qwen-14b-v1.5'
    'internlm2-20b-chat' 'mistral-8_7b-it-v0.1' 'qwen-72b-chat-v1.5'
)

datasets=(
    "truthfulqa"
    "toxigen"
    "stereoset"
    "pku-rlhf-10k"
    "confaide"
)

device='cuda'

idx=0
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "$dataset $model"

        CUDA_VISIBLE_DEVICES=$idx python generate_activations.py --model "$model" --layers -1 --datasets "$dataset" --downsample 200 --device "$device"
        
        # CUDA_VISIBLE_DEVICES=$idx python generate_activations.py --model "$model" --layers -1 --datasets "$dataset" --device "$device"

        # CUDA_VISIBLE_DEVICES=$idx python generate_activations.py --model "$model" --layers -1 --datasets "$dataset" --device "$device" --load_in_4bit

    done
done
