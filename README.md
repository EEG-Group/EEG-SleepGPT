# EEG-SleepGPT


![EEG-SleepGpt](image\EEG-SleepGpt.png)



## Environment Set Up

Install required packages:
```bash
conda create -n EEG-SleepGPT python=3.12
conda activate EEG-SleepGPT 
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install transformers datasets==2.21.0 tiktoken wandb h5py einops pandas scikit-learn
```
## Run Experiments
### Prepare pre-training data
You should transfer raw EEG files (such as .cnt, .edf, .bdf, and so on) into pikle files using the example code in dataset_maker (one file represents one sample). Notably, you can also write your own codes for preprocessing EEG data. 
```bash
python python prepare_TUH_pretrain.py
```
Also, you should prepare the text dataset by runing the script in text_dataset_maker.
```bash
python python prepare.py
```
### Train the text-aligned neural tokenizer
The neural tokenizer is trained by vector-quantized neural spectrum prediction. We train it on platforms with 8 * NVIDIA A100-80G GPUs.
```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 train_vq.py \
    --dataset_dir /path/to/your/dataset \
    --out_dir /path/to/save \
    --wandb_log \
    --wandb_project your_project_name \
    --wandb_runname your_runname \
    --wandb_api_key your_api_key \
```
### EEG-SleepGPT pre-train
We pre-train EEG-SleepGPT by multi-channel autoregressive pre-training.
```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 train_pretrain.py \
    --dataset_dir /path/to/your/dataset \
    --out_dir /path/to/save \
    --tokenizer_path checkpoints/VQ.pt \
    --wandb_log \
    --wandb_project your_project_name \
    --wandb_runname your_runname \
    --wandb_api_key your_api_key \
```
### Multi-task instruction tuning on downstream tasks
Before fine-tuning, use the code in dataset_maker/(make_TUAB.py, make_TUEV.py, etc.) to preprocess the downstream datasets as well as split data into training, validation, and test set. Notably you are encouraged to try different hyperparameters, such as the learning rate and warmup_epochs which can largely influence the final performance, to get better results.
```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 train_instruction.py \
    --dataset_dir /path/to/your/dataset \
    --out_dir /path/to/save \
    --NeuroLM_path checkpoints/NeuroLM-B.pt \
    --wandb_log \
    --wandb_project your_project_name \
    --wandb_runname your_runname \
    --wandb_api_key your_api_key \
```

