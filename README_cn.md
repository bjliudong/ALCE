# Enabling Large Language Models to Generate Text with Citations

<p align="center"><img src="https://github.com/princeton-nlp/ALCE/blob/main/assets/moose.png?raw=true" alt="ALCE" width="15%"><br>*: ALCE 的发音为 /elk/，因为 ALCE 是麋鹿（欧洲）或驼鹿（北美）的拉丁语单词。
</p>



这个代码库包含了这篇论文的代码和数据集。 [Enabling Large Language Models to Generate Text with Citations](https://arxiv.org/abs/2305.14627). 
在这篇论文中我们提出了 ALCE，一个, 一个基准， **A**utomatic **L**LMs' **C**itation **E**valuation. 
ALCE 包含三个数据集: ASQA, QAMPARI, 和 ELI5.
我们针对大语言模型生成的三项指标：流畅度、正确性和引文质量，提供了自动评估代码。
这个代码库也包含代码
这个代码库还包括用于复制我们论文中的基线的代码。



<img src="https://github.com/princeton-nlp/ALCE/blob/main/assets/ALCE.png?raw=true" alt="ALCE" width="100%">




## Quick Links

- [Enabling Large Language Models to Generate Text with Citations](#enabling-large-language-models-to-generate-text-with-citations)
  - [Quick Links](#quick-links)
  - [部署环境要求](#部署环境要求)
  - [数据](#数据)
    - [检索](#检索)
  - [代码结构](#代码结构)
  - [再现基线](#再现基线)
    - [Post-hoc 引文](#post-hoc-引文)
  - [评估](#评估)
  - [人类评估](#人类评估)
  - [Bug和问题?](#bug和问题)
  - [引用](#引用)


## 部署环境要求

请安装最新版本的Pytorch (`torch`), HuggingFace Transformers (`transformers`), HuggingFace Accelerate (`accelerate`), and the OpenAI API package (`openai`). 这个代码库曾在以下环境进行测试
`torch==2.1.0.dev20230514+cu118`, `transformers==4.28.1`, `accelerate==0.17.1`, and `openai==0.27.4` with Python 3.9.7.

## 数据

你可以通过以下命令下载数据集 （以及检索结果）

```bash
bash download_data.sh
```

所有数据都将被存储在 `data/` 目录下。我们的数据包含从 ASQA 和 QAMPARI 以 DPR/GTR 检索的 top-100 结果，以及从 QAMPARI 以 BM25 检索的 top-100 结果。我们也提供了经过重拍的 oracle 检索结果，其中 top-5 段落可以实现与原始 top-100 段落相同的召回情况。

### 检索

您可以使用以下命令重现文章检索步骤：

```bash
python retrieval.py --data {path/to/data} --retriever {bm25/gtr} --output_file {path/to/output}
```

检索步骤还需要其他包。明确的是，你需要安装 `pyserini==0.21.0`(他们在 github 的代码库 [repo](https://github.com/castorini/pyserini/tree/master) 是很有帮助的) 和 `sentence-transformers==2.2.2`.

对于使用Sphere通过通用爬虫进行的BM25检索，必须首先从Github下载Sphere索引 [repo](https://github.com/facebookresearch/Sphere)，还需要设置环境变量  `BM25_SPHERE_PATH` 为下载索引的路径。你可以使用下面的命令：

```bash
wget -P faiss_index https://dl.fbaipublicfiles.com/sphere/sphere_sparse_index.tar.gz
tar -xzvf faiss_index/sphere_sparse_index.tar.gz -C faiss_index
export BM25_SPHERE_PATH=$PWD/faiss_index
```

需要注意的是，考虑到语料库的巨大规模，这一步骤极其昂贵和耗时。我们发现，较大的CPU内存往往有助于提高速度。

对于GTR，我们首先使用DPR-wikipedia快照构建索引，您可以使用DPR的下载脚本获得该快照 [repo](https://github.com/facebookresearch/DPR)，然后将环境变量 `DPR_WIKI_TSV` 设置为TSV文件的路径。 你可以使用下面的命令操作：

```bash
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gzip -xzvf psgs_w100.tsv.gz
export DPR_WIKI_TSV=$PWD/psgs_w100.tsv
```

然后，您想将“GTR_EMB”设置为维基百科语料库的GTR嵌入的路径，第一次运行检索脚本将自动构建和保存索引。
构建密集索引对于GPU内存来说可能是昂贵的（我们使用80GB GPU）并且耗时；整个索引将占用大约31GB。
如果您觉得此步骤太贵，也可以使用以下方式下载：

```bash
wget https://huggingface.co/datasets/princeton-nlp/gtr-t5-xxl-wikipedia-psgs_w100-index/resolve/main/gtr_wikipedia_index.pkl
export GTR_EMB=$PWD/gtr_wikipedia_index.pkl
```

为了重现DPR检索，我们参考 DPR [repo](https://github.com/facebookresearch/DPR) ，我们使用在NQ上训练的原始DPR检查点。

## 代码结构

* `run.py`: 运行文件以重现我们的基线生成。
* `eval.py`:用于计算生成数的 eval 文件。
* `prompts`: 包含所有提示文件的文件夹。
* `configs/`: 文件夹，其中包含用于重新生成基线的所有配置文件。
* `tools/`: 杂项代码（生成摘要/片段、重新排序等）


## 再现基线


你可以依据我们的论文再现基线。

```bash
python run.py --config configs/{config_name}
```

您也可以覆盖配置文件中的任何参数，或者只需通过命令行添加新参数：

```
python run.py --config configs/{config_name} --seed 43 --model vicuna-13b
```

配置文件以下面的规则进行命名 `{LLM}_{#demos and #passages}_{retriever}_{method}.yaml`。方法名包括：
* `default` 论文中对应的 **Vanilla** 模型。
* `summary` 对应 **Summary** 模型。
* `extraction` 对应 **Snippet** 模型。
* `interact_doc_id` 对应 **Interact** 模型。
* `interact_search` 对应 **InlineSearch** 模型。
* `closedbook` 对应 **ClosedBook** 模型。

我们的代码同时支持 OpenAI API 和 离线的 HuggingFace 模型：

* 对于 OpenAI 模型 (例如， ChatGPT)， 你需要设置环境变量 `OPENAI_API_KEY` 和 `OPENAI_ORG_ID`。如果你使用 Azure OpenAI API, 你需要设置环境变量 `OPENAI_API_KEY` 和 `OPENAI_API_BASE`。你还需要添加参数 `--azure`。
    * 注意在 Azure OpenAI API, ChatGPT 的名字不同，你需要设置为 `--model gpt-35-turbo`。
* 对于开源模型，您应该将模型名称设置为等于HuggingFace模型方法的输入 `.from_pretrained` 。 这可能是本地目录 (比如老的 LLaMA 模型) 或通往 HuggingFace hub 的路径。

有关参数的详细用法，请参阅 `run.py`。

模型输出以及黄金答案和运行配置将存储在中的json文件中 `result/`。


### Post-hoc 引文

对于 closed-book 模型, 可以使用 `post_hoc_cite.py` 来添加引文到一个 post-hoc 方式 (使用 GTR-large)。 运行 post-hoc 引文，执行：

```bash
python post_hoc_cite.py --f result/{RESULT JSON FILE NAME} --external_docs data/{CORRESPONDING DATA}
```

post-hoc 引文的输出结果将存储在 `result/`，并且带有后缀 `post_hoc_cite.gtr-t5-large-external`。

## 评估

ACLE 评估的实现在 `eval.py`。

对于 ASQA，使用以下命令

```bash
python eval.py --f {path/to/result/file} --citations --qa --mauve
```

对于 QAMPARI，使用以下命令

```bash
python eval.py --f {path/to/result/file} --citations
```

对于 ELI5，使用以下命令

```bash
python eval.py --f {path/to/result/file} --citations --claims_nli --mauve
```

评估结果也将保存在 `result/`，具有与输入相同的名称和后缀 `.score`。

## 人类评估

我们的人类评估结果（第6节）位于目录下 [`human_eval`](human_eval). 
数据和分析都可用，请参阅论文了解详细信息。 

## Bug和问题?

如果您对代码或论文有任何疑问，请随时发送电子邮件给高天宇 (`tianyug@cs.princeton.edu`). 如果您在使用代码时遇到任何问题，或者想报告错误，可以打开 issue。请尝试详细说明问题，以便我们可以更好更快地帮助您！



## 引用

如果在你的工作中使用 ALCE 请在论文中采用如下引用：

```bibtex
@inproceedings{gao2023enabling,
   title={Enabling Large Language Models to Generate Text with Citations},
   author={Gao, Tianyu and Yen, Howard and Yu, Jiatong and Chen, Danqi},
   year={2023},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
}
```
