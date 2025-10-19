
# From Easy to Hard: The MIR Benchmark for Progressive Interleaved Multi-Image Reasoning 


## 🚀 Project Overview  
The main contributions of this paper are as follows:

- **Benchmark dataset (MIR)**: A new dataset focusing on *interleaved multi-image reasoning*, i.e., questions that require jointly reasoning across **multiple images** *and* interleaved textual context. ([arxiv.org](https://arxiv.org/pdf/2509.17040))  
- **Structured reasoning steps**: Each sample in the MIR benchmark is annotated with five reasoning stages: *Summary*, *Caption*, *Text→Region*, *Region→Region*, *Conclusion*.  
- **Curriculum (easy → hard) training strategy**: A stage-wise learning scheme that first fine-tunes on “easy” samples then progressively introduces “harder” ones, guiding a multimodal large language model (MLLM) from simpler to more complex reasoning.  
- **Extensive experiments**: Fine-tuning various MLLMs on MIR + curriculum learning significantly improves both in-domain and out-of-domain performance.

The goal is to facilitate research into multimodal models (especially large vision-language models) that can handle *multiple images + interleaved text* reasoning beyond standard single-image VQA or simple multi-image tasks.


## 📥 Environment Setup  
### Prerequisites  
- Python ≥ 3.10  
- CUDA ≥ 11.8
- Torch ≥ 2.5.1+cu118


### Installation  
Qwen-VL series models are fine-tuned using **ms-swift 3.2.0.dev0**. You can install **ms-swift** from source following the [official documentation](https://swift.readthedocs.io/en/latest/GetStarted/SWIFT-installation.html) 

Other models are fine-tuned following the official implementations provided in their respective GitHub repositories or papers.

### Downloading the Dataset  
The MIRBench dataset is organized in JSON format, where each sample corresponds to a multi-image reasoning instance. Each entry typically contains a list of image paths (images), a natural language question (question), several answer options (options), and the correct answer (conclusion). In addition, every instance is annotated with multi-step reasoning traces, including stages such as Summary, Caption, Text-to-Region, Region-to-Region. The dataset has been released on Hugging Face at [MIRBench](https://huggingface.co/datasets/ShellyCoder/MIRBench), but the data is currently under review and being updated. We will complete the maintenance as soon as possible. We hope that this structured reasoning dataset will contribute to advancing the capabilities of multimodal models. The overall structure of the dataset is as follows:
```bash
MIRBench
  - all_data_rel.json  
  - images.zip
```


### Pre-processing  
We use the Qwen-VL series to perform multiple sampling passes for difficulty segmentation and provide a reasoning example for reference.
```bash
python vlm_infer.py
```

You can flexibly combine data from different reasoning stages as needed, including adding special tokens such as `<conclusion>...</conclusion>` to separate different parts of the content. We also provide an example of how to process the data in this way.

```bash
python preprocess.py
```

### Training / Fine-tuning  
We first divide the data into different difficulty levels (while shuffling the order within each level) and then concatenate them into a new JSON file.  
In the training script, we set `shuffle=False` to ensure that the model is trained in the predefined easy-to-hard order. We provide a training example script as follows:

```bash
sh sft.sh
```

### Evaluation  
After SFT training based on the dataset format, the model often generates extensive reasoning content, making it difficult to perform exact matching for the correct option.  
To address this, you can extract the **conclusion** part (if it is separated by special tokens), or use a large language model API to parse and extract the predicted option, and then compute the accuracy using exact match evaluation.


## 📄 License & Citation  
This project is licensed under **Apache-2.0** .  

If you use the MIR benchmark or its methods, please cite:
```bibtex
@article{DuEtAl2025_MIRBenchmark,
  title={From Easy to Hard: The MIR Benchmark for Progressive Interleaved Multi-Image Reasoning},
  author={Hang Du and Jiayang Zhang and Guoshun Nan and Wendi Deng and Zhenyan Chen and Chenyang Zhang and Wang Xiao and Shan Huang and Yuqi Pan and Tao Qi and Sicong Leng},
  journal={arXiv preprint arXiv:2509.17040},
  year={2025}
}
```

