prompts_file="/data/sanchez_trlx/llama2/llama.cpp/prompts/chat-with-sanchez.txt"
results_folder="/data/sanchez_trlx/llama2/dataset/results"
llama_exec="/data/sanchez_trlx/llama2/llama.cpp/main"
model_path="/data/sanchez_trlx/llama2/llama.cpp/models/7B/ggml-model-q4_0.bin"
log_file="/data/sanchez_trlx/llama2/logs.txt"
dataset_folder="/data/sanchez_trlx/llama2/dataset/mathqa"

# Iterate through each file in the dataset folder
for file in "${dataset_folder}"/*
do

    result_filename=$(basename "${file}")
    if [ -f "${results_folder}/${result_filename}" ]; then
        echo "Skipping existing result file for ""${file}"
        continue
    fi

    # Run the llama.cpp/main command using the concatenated file
    combined_input="$(cat "${prompts_file}")$(cat "${file}")""\n###\nConversation\n\nStudent:"
    echo "${combined_input}" > tmp_combined_input.txt

    echo "Processing ""${file}" >> log_file
    echo "Processing ""${file}"

    #echo "Prompt ""${combined_input}"

    ${llama_exec} -m ${model_path} -n 1024 --repeat_penalty 1.0 --n-gpu-layers 25000 --log-disable -e -p "${combined_input}" > tmp_result.txt

    echo "Processed"

    # Add the final result to the corresponding result file
    result_filename=$(basename "${file}")
    cat tmp_result.txt >> "${results_folder}/${result_filename}"

    # Clean up temporary files
    #rm tmp_result.txt
done