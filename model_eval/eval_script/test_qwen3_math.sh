source activate opencompass
sleep 2


if [ $# -eq 0 ]; then
  echo "Please provide the model path in the command line, e.g., llama3.1-8b/full/sft-gsm8k, and the dataset name, e.g., gsm8k."
  exit 1
fi
START_TIME=`date +%Y%m%d-%H:%M:%S`
MODEL_PATH=$1
DATANAME=$2
EXP_NAME=qwen3_test

GPUS_PER_NODE=1
N_NODE=1
output_file="opencompass/configs/models/qwen2_5/vllm_qwen3_8b_instruct_${START_TIME}.py"
LOG_FILE=logs/${START_TIME}_${EXP_NAME}.log

cat << EOF > $output_file
from opencompass.models import VLLMwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content
models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen3-8b-instruct-vllm',
        path='${MODEL_PATH}',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=32768,
        batch_size=8,
        generation_kwargs=dict(temperature=0.6,top_p=0.95,top_k=20),
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
EOF
echo "YAML 配置文件已成功创建：$output_file"

#python replace_chat_template_to_default.py --model_path ${MODEL_PATH} 

opencompass \
    -w outputs_qwen3_test/${DATANAME}_${START_TIME}_${EXP_NAME} \
    --max-out-len 32768 \
    --datasets \
    aime2025_repeat8_cver_gen \
    omni_math_gen OlympiadBenchMath_0shot_cver_gen \
    gsm8k_0shot_cver_gen  math_prm800k_500_0shot_cot_cver_gen aime2024_repeat8_cver_gen \
    cmimc2025_repeat8_cver_gen hmmt2025_repeat8_cver_gen brumo2025_repeat8_cver_gen \
   --hf-type chat --models vllm_qwen3_8b_instruct_${START_TIME} --max-num-worker 1 -a vllm -r
