# Fine-Tuning and Inference with TI LoRAs

## STRICT DATASET GATHERING AND ANNOTATION GUIDELINES
- Collect 16-20 images of the character in diverse angles (close-up, mid-range and wide/long range shots)
- In case the subject in question is a prop or a set/background, relatively fewer images will do
- Gather images that match your final goal (Ex: If you want to show Allu Arjun with full mustache and beard, collect similar images for model fine-tuning)
- Make sure to include several images of the character , among many others, where the face is crystal clear 
- Assign a prompt for EVERY image in the dataset and use a unique `special token` for character identification (Ex: A photo of (([AA] man)) dancing in rain)
- During dataset annotation, `**STRICTLY**` do not describe the facial and body features of the charcater, that will be taken care by the initializer concept

  | Do ✅ |  Don't ❌ |
  | ----------- | ------------ |
  |  A photo of (([AA] man)) dancing in rain  | A photo of (([AA] man)) in his early 30s, muscular build, dusky skin tone, tall, dancing in rain | 


- The special token and the class should appear `**ONLY ONCE**` in the dataset annotations

  | Do ✅ |  Don't ❌ |
  | ----------- | ------------ |
  | A photo of (([AA] man)), wearing a red shirt, dancing in rain | A photo of (([AA] man)), dancing in rain, where (([AA] man)) is wearing a red shirt | 


- Keep the annotation prompts simple and crisp
- For the initializer concept, only describe the following aspects of a character: age, ethinicity, skin colour, height, specific facial and hair features, body features
- `**DO NOT**` provide the character's special token, clothing, expression and posture, etc in the initializer concept

  | Do ✅ |  Don't ❌ |
  | ------- | ----- |
  | A 30 year old Indian man, 6 feet tall, muscular build, dusky skin tone, sharp nose and jawline | (([AA] man)) tall, curly long hair, wearing pink shirt, confident expression |



## CLI Running

```bash
AITK_JOB_ID=<any_random_number> python3 run.py <path/to/your/config.yaml>
```

## H200s Dev Set-Up

In config:

```bash
Host shivampc
  HostName shivampc.pratikn.com
  User gpuaccess
  IdentityFile /Users/shivanvitha/id_rsa
  ProxyCommand /opt/homebrew/bin/cloudflared access ssh --hostname %h

Host h200vm
  HostName localhost
  Port 32777
  User root
  ProxyJump shivampc
  IdentityFile /Users/shivanvitha/id_rsa
```

Enable key-based authentication:

```bash
cat id_rsa.pub | ssh -o ProxyCommand="cloudflared access ssh --hostname shivampc.pratikn.com" gpuaccess@localhost "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"


cat id_rsa.pub | ssh -J gpuaccess@shivampc -p 32777 root@localhost "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

## Tips for Dataset Images
- Use high quality .png images
- Gather images keeping the target generations in mind (Eg,: If the output scenes of Alia Bhatt are to be set in Pakistan, use her images from Raazi instead of Rocky aur Rani)
- ALWAYS include the special token in EVERY prompt in the dataset
- Keep your prompts concise and straight to the point
- Make sure that the face is clear in every image and include long/wide range shots as well if the inference outputs should have the same

## Fine-Tuning

### On AMD MI300X GPUs - dheyo_amd Branch

- Login to the jump server from your VSCode on `193.143.78.200`

- SSH to gpu-60 with Tunneling

```bash
ssh -L 7777:localhost:7777 ubuntu@gpu-60
```

- Navigate to ai-toolkit directory 

```bash
cd /shareddata/dheyo/shivanvitha/ai-toolkit
```

- Activate the virtual environment

```bash 
source aitool/bin/activate
```

- Launch UI 

```bash
cd ui
```

```bash
npm run build_and_start
```

- Open the application on port 7777 on the browser 

```bash 
http://localhost:7777/
```

- Upload dataset(s): Navigate to the `Datasets` link on the left menu and add your images there. Write the captions for each image in the textbox provided beneath each image. [DO NOT FORGET TO PRESS ENTER ONCE YOU FINISH WRITING EVERY SINGLE CAPTION]

- **IMPORTANT: Assign a unique special token to each character and follow it up by the class of the subject. (eg.,: [A] man, [AB] woman)**

- Navigate to the `+ New Job` section on the left menu and choose the below settings:

| Nodel Configuration     | Value           |
|-----------------|--------------------------|
| Training Name   | Give some name based on the character(s) that you are about to train |
| GPU ID      | Choose any idle GPU (among the 8 [`GPU #0`, `GPU #1`,..., `GPU #7` ]) |
| Model Architecture | Qwen/Qwen-Image |
| Save Every      | 300 or 400 |
| Batch Size      | 2 |
| Steps      | At least 4000 |

- Leave the `Embedding Training` option as it is in the training configuration

| Dataset Configuration     | Value           |
|-----------------|--------------------------|
| Dataset 1 - Trigger/Special Token | The special token to be used for the character in the dataset (eg,: [A], [AB], etc..) |
| Dataset 1 - Initializer Concept | A short description of the character (eg,: A 35 year old Indian man with a strong build, brown skin, beard and mustache, A young Indian woman with fair skin, dimples on cheeks, fair skin, slim build)|

Do the same if there are more than one datasets to be trained after clicking on `Add Dataset`

- Sample Prompts: At the bottom of the page you can provide certain validation prompts. Make sure that these have the special trigger followed by the class of the object just like in the training dataset prompts

- Click on `Create Job` after adding all the settings

- Click on the play button at the top right corner

- **Note: If the training fails, ping Shivanvitha Ambati well before 10pm**



----


### On NVIDIA H100 GPUs - dheyo_nvidia Branch

Command to run from the terminal:

```bash
CUDA_VISIBLE_DEVICES=0 AITK_JOB_ID=<any_random_number> python3 run.py <path_to_your_config_file>
```

- Login to dheyo01 VSCode on `lh100.dheyo.ai` with password `Gailen804!`

- Navigate to the working directory 

```bash
cd /data/shivanvitha/dheyo_ai_toolkit/ai-toolkit
```

- Activate the virtual environment 

```bash 
source ../../../ai-toolkit/aitool/bin/activate
```

- Launch UI 

```bash
cd ui
```

```bash
npm run build_and_start
```

- Open the application on port 7777 on the browser 

```bash 
lh100.dheyo.ai:7777
```

- The rest of the instructions are same as described in the AMD section

----

## Inference 

- Stay in the same directory (irrespective of which machine) and find the file with the name -- `inference_qwen_image_lora.py`

- The checkpoints are stored in the `output` directory. Find the sub-directory under this with the training name you provided during fine-tuning

- Copy the absolute paths of the below

| Category     | Pattern           |
|-----------------|--------------------------|
| Transformer LoRA Path  | <your_training_name>_<ckpt_number>.safetensors |
| Text Encoder Path      | text_encoder_<your_training_name>_<ckpt_number> |
| Tokenizer Path | tokenizer_<your_training_name>_<ckpt_number> |
| Token Abstraction JSON apth      | There will be a tokens.json file |


- Create a command and launch it

Always look out for idle GPUs and give the device ID accordingly in the environment variable (`HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES`) in front of the actual script command

**For multi prompt bulk generation, each prompt should start on a newline**

### On AMD:

Single Prompt Inference:

```bash
HIP_VISIBLE_DEVICES=7 python3 inference_qwen_image_lora.py --model_path "Qwen/Qwen-Image" \
--transformer_lora_path <your_transformer_lora_path> \
--tokenizer_path <your_tokenizer_path> \
--text_encoder_path <your_text_encoder_path> \
--token_abstraction_json_path tokens.json \
--num_inference_steps 50 \
--output_image_path inferenced_images/<some_file_name>.png \
--instruction "A photo of [A] man in prison, crying, wearing prison outfit with 420 written on his shirt" \
--aspect_ratio "16:9"
```

Multiple Prompt Inference:
```bash
HIP_VISIBLE_DEVICES=7 python3 inference_qwen_image_lora.py --model_path "Qwen/Qwen-Image" \
--transformer_lora_path <your_transformer_lora_path> \
--tokenizer_path <your_tokenizer_path> \
--text_encoder_path <your_text_encoder_path> \
--token_abstraction_json_path tokens.json \
--num_inference_steps 50 \
--output_image_path inferenced_images/<some_file_name>.png \
--prompts_path <your_prompts_txt_file>.txt \ \
--aspect_ratio "16:9"
```


### On NVIDIA:

```bash
CUDA_VISIBLE_DEVICES=1 python3 inference_qwen_image_lora.py --model_path "Qwen/Qwen-Image" \
--transformer_lora_path <your_transformer_lora_path> \
--tokenizer_path <your_tokenizer_path> \
--text_encoder_path <your_text_encoder_path> \
--token_abstraction_json_path tokens.json \
--num_inference_steps 50 \
--output_image_path inferenced_images/<some_file_name>.png \
--instruction "A photo of [A] man in prison, crying, wearing prison outfit with 420 written on his shirt" \
--aspect_ratio "16:9"
```

Multiple Prompt Inference:
```bash
CUDA_VISIBLE_DEVICES=1 python3 inference_qwen_image_lora.py --model_path "Qwen/Qwen-Image" \
--transformer_lora_path <your_transformer_lora_path> \
--tokenizer_path <your_tokenizer_path> \
--text_encoder_path <your_text_encoder_path> \
--token_abstraction_json_path tokens.json \
--num_inference_steps 50 \
--output_image_path inferenced_images/<some_file_name>.png \
--prompts_path <your_prompts_txt_file>.txt \ \
--aspect_ratio "16:9"
```
