# Main script for running reasoning tasks on Modal
from importlib.resources import path
import modal

# Manually create `models` volume and upload from local via cli

# Create: modal volume create models

# List: modal volume list
models_volume = modal.Volume.from_name("models")

# Create a modal image and install libraries
# Copy the current folder/repo, except main.py file

flash_attn_release = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)

# 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl


provocation_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
).run_commands(
    "apt-get update && apt-get install -y git build-essential libnuma-dev",
    "pip install numpy huggingface-hub pandas pyyaml scikit-learn scipy polars matplotlib accelerate",
    "pip install torch tqdm",
    "pip install 'sglang[all]==0.5.9'",
    # "pip install --force-reinstall --no-deps git+https://github.com/huggingface/transformers.git",
    "pip install --upgrade sgl_kernel",
).add_local_dir(".", "/root/", copy=True, ignore=["./main.py",  # This will be copied when running modal on main.py
                                       "./data/instances/", "./models",  # these will come from Volumes
                                       "./.gitignore", "./CHANGELOG.txt", "./LICENSE", "./MANIFEST.in",
                                       "./pyproject.toml", "./README.md", "./.git", "./.idea", "./__pycache__",
                                       "./data/__pycache__",]).run_commands(
    "python3 /root/patch_sglang.py && python3 /root/patch_sglang_prefill.py"
)

# provocation_image = modal.Image.from_registry(
#     "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
# ).run_commands(
#     "apt-get update && apt-get install -y git build-essential",
#     "pip install numpy huggingface-hub pandas pyyaml scikit-learn scipy polars matplotlib accelerate",
#     "pip install torch tqdm transformers",
#     "pip install packaging ninja wheel setuptools",
# ).env({"CUDA_HOME": "/usr/local/cuda"}).run_commands(
#     "pip install flash-attn ",
# ).add_local_dir(".", "/root/", ignore=["./main.py",  # This will be copied when running modal on main.py
#                                        "./data/instances/", "./models",  # these will come from Volumes
#                                        "./.gitignore", "./CHANGELOG.txt", "./LICENSE", "./MANIFEST.in",
#                                        "./pyproject.toml", "./README.md", "./.git", "./.idea", "./__pycache__",
#                                        "./data/__pycache__",])

# Create Modal app
app = modal.App("provocation", image=provocation_image)

# Nvidia B200 # $6.25 / h
# Nvidia H200 # $4.54 / h
# Nvidia H100 # $3.95 / h
# Nvidia A100, 80 GB # $2.50 / h
# Nvidia A100, 40 GB # $2.10 / h
# Nvidia L40S # $1.95 / h
# Nvidia A10 # $1.10 / h
# Nvidia L4 # $0.80 / h
# Nvidia T4 # $0.59 / h

TP_DEGREE = 8

@app.function(volumes={"/root/models/": models_volume},
              timeout=86000,
              gpu=f"H200:{TP_DEGREE}",  # Adjust GPU type and count as needed
              memory=16384,
              cpu=8)
def run():

    import os, subprocess
    current_dir = os.getcwd()
    import sglang
    print ("################## SGLANG VERSION ##################")
    print(sglang.__version__)

    # parent_dir = os.path.dirname(current_dir)
    print("Current directory:", current_dir)
    
    os.makedirs("/root/models/new_results", exist_ok=True)
    os.makedirs("/root/models/hub", exist_ok=True)
    subprocess.run(["ls", "-al", current_dir])

    # Route TP_DEGREE to all child scripts via environment.
    os.environ["TP_DEGREE"] = str(TP_DEGREE)

    # input_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    input_list = [2, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    seed_1_list = [907921355002, 848224508876, 72374813649, 56483805493, 408562453599, 481568476781, 121455843125, 568493433001, 10203010204, 1234567890, 1234534690, 1234567896]
    seed_2_list = [380553961179, 52459502640, 965615506304, 332904421914, 612694720271, 48438200968, 164859935551, 622717036501, 12512512310, 9876543210, 9876534610, 9876543216]


    input_list = [250, 300, 350, 400, 450, 500]
    seed_1_list = [121455843125, 568493433001, 10203010204, 1234567890, 1234534690, 1234567896]
    seed_2_list = [164859935551, 622717036501, 12512512310, 9876543210, 9876534610, 9876543216]



    # input_list = [2, 10, 50, 100, 150]
    # seed_1_list = [907921355002, 848224508876, 72374813649, 56483805493, 408562453599, 481568476781]
    # seed_2_list = [380553961179, 52459502640, 965615506304, 332904421914, 612694720271, 48438200968]



    model_list = [
        # "google/gemma-3-270m-it",
        # "Qwen/Qwen3-0.6B", 
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        # "deepseek-ai/DeepSeek-V4-Flash",
    ]

    # Override for smaller experiment
    # model_list = [
    #    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # ]

    for model_name in model_list:
        # cmd = [
        #     "python",
        #     "-u",
        #     "./text_pass.py",
        #     "0",
        #     "0",
        #     "0",
        #     "0",
        #     "0",
        #     model_name,
        # ]
        # print(f"Launching text_pass.py for model: {model_name}")
        # print("Command:", " ".join(cmd))

        # env = dict(os.environ)
        # env["PYTHONUNBUFFERED"] = "1"
        # lower_model_name = model_name.lower()
        # if any(tag in lower_model_name for tag in ["7b", "8b", "14b", "32b", "70b"]):
        #     mem_fraction_default = "0.45"
        # else:
        #     mem_fraction_default = "0.20"
        # env.setdefault("MEM_FRACTION_STATIC", mem_fraction_default)
        # env.setdefault("SAVE_EVERY_PROMPTS", "20")
        # env.setdefault("MAX_PROMPT_TOKENS", "4000")
        # env.setdefault("MAX_PREFILL_TOKENS_PER_CALL", "6000")
        # env.setdefault("DISABLE_CUDA_GRAPH", "1")
        # print(
        #     "Runtime env overrides: "
        #     f"MEM_FRACTION_STATIC={env['MEM_FRACTION_STATIC']} "
        #     f"SAVE_EVERY_PROMPTS={env['SAVE_EVERY_PROMPTS']} "
        #     f"MAX_PROMPT_TOKENS={env['MAX_PROMPT_TOKENS']} "
        #     f"MAX_PREFILL_TOKENS_PER_CALL={env['MAX_PREFILL_TOKENS_PER_CALL']} "
        #     f"DISABLE_CUDA_GRAPH={env['DISABLE_CUDA_GRAPH']}"
        # )
        # result = subprocess.run(cmd, env=env)
        # print(f"text_pass.py exit code: {result.returncode}")
        # if result.returncode != 0:
        #     raise RuntimeError(f"text_pass.py failed for model {model_name} with exit code {result.returncode}")

        for i in range(len(input_list)):

            input_size = input_list[i]
            seed_one = seed_1_list[i]
            seed_two = seed_2_list[i]
            run_idx = 0
            temp = 0

            # if int(input_size) >= 300 and model_name ==  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
            #     subprocess.call(f'python ./reasoning_test_modular_addition.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)
            #     subprocess.call(f'python ./reasoning_test_brackets.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)
            #     subprocess.call(f'python ./reasoning_test_index.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)
                                
            # if int(input_size) >= 450 and model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B":
            #     subprocess.call(f'python ./reasoning_test_modular_addition.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)
            #     subprocess.call(f'python ./reasoning_test_brackets.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)
            #     subprocess.call(f'python ./reasoning_test_parity.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)
            #     subprocess.call(f'python ./reasoning_test_index.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)
                
            # elif int(input_size) == 500 and model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":
            #     subprocess.call(f'python ./reasoning_test_parity.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)
            #     subprocess.call(f'python ./reasoning_test_index.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)

            # if int(input_size) >= 450 and model_name == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B":
            #     subprocess.call(f'python ./reasoning_test_modular_addition.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)
            #     subprocess.call(f'python ./reasoning_test_brackets.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)
            #     subprocess.call(f'python ./reasoning_test_parity.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)
            #     subprocess.call(f'python ./reasoning_test_index.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)
                
            # else:
            #     print (f"############# Skipping model: {model_name} for input size: {input_size} - already computed. #############")


            print(f"Running Modular Addition with Model = {model_name}\nInput Size = {input_size}\nSeed One = {seed_one}\nSeed Two = {seed_two}\nRun Index = {run_idx}\nTemperature = {temp}")
            subprocess.call(f'python ./reasoning_test_modular_addition.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)
            subprocess.call(f'python ./reasoning_test_brackets.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)
            subprocess.call(f'python ./reasoning_test_parity.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)
            subprocess.call(f'python ./reasoning_test_index.py {input_size} {seed_one} {seed_two} {run_idx} {temp} {model_name}', shell=True)
    
    print("All tasks completed.")


# > modal run main.py
# --detach flag to run in background, continue even terminal is closed
@app.local_entrypoint()
def main():
    # print(run.local())
    run.remote()

