import argparse
import collections
import json
import re
import string
import torch
import copy

from nltk import sent_tokenize
import numpy as np
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm
import sys
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)

from utils import normalize_answer, get_max_memory, remove_citations

QA_MODEL="gaotianyu1350/roberta-large-squad"
AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"

global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None


def compute_f1(a_gold, a_pred):
    """Compute F1 score between two strings."""

    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False


def compute_rouge(data):
    """Main function for rouge scoring.
    If two references are provided,
    the best score is chosen for each instance.
    Args:
        data: requires field `output` and `answer` (or `annotations` for ASQA)
        metrics: list of evaluation metrics
    Returns:
        dictionary representation of rouge scores
    """
    def _rouge_calculation(hypotheses,
                        references1,
                        references2=[],
                        metrics=['rougeLsum']):

        if references2 == []:
            references2 = references1

        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for i in range(len(hypotheses)):
            scores1 = scorer.score(references1[i], hypotheses[i])
            scores2 = scorer.score(references2[i], hypotheses[i])
            if scores1['rougeLsum'].fmeasure > scores2['rougeLsum'].fmeasure:
                aggregator.add_scores(scores1)
            else:
                aggregator.add_scores(scores2)

        scores = {m: [] for m in metrics}

        for m in metrics:
            fmeasure = aggregator.aggregate()[m].mid.fmeasure
            scores[m].append(fmeasure)

        for m in scores:
            scores[m] = 100 * sum(scores[m]) / len(scores[m])

        return scores

    hypotheses = {}
    references1 = {}
    references2 = {}

    for idx, item in enumerate(data):
        hypotheses[idx] = item["output"]
        if "annotations" in item and item['annotations'] is not None: # For ASQA
            references1[idx] = item["annotations"][0]["long_answer"]
            references2[idx] = item["annotations"][1]["long_answer"]
        else:
            references1[idx] = item["answer"]
            references2[idx] = item["answer"]

    h, r1, r2 = [], [], []

    for key in references1:
        h.append(hypotheses[key])
        r1.append(references1[key])

        if references2 is not None:
            r2.append(references2[key])

    h = ['\n'.join(sent_tokenize(text.lower())) for text in h]
    r1 = ['\n'.join(sent_tokenize(text.lower())) for text in r1]
    r2 = ['\n'.join(sent_tokenize(text.lower())) for text in r2]
    scores = _rouge_calculation(h, r1, r2)

    return scores['rougeLsum']


def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        return 0, 0

    acc = []
    hit = []

    for item in data:
        loc_acc = []
        for qa_pair in item['qa_pairs']:
            loc_acc.append(exact_presence(qa_pair['short_answers'], item["output"]))
        acc.append(np.mean(loc_acc))
        hit.append( int(np.mean(loc_acc) == 1) )

    return 100 * np.mean(acc), 100 * np.mean(hit)


def compute_len(data):
    """Compute average length of predictions."""

    res, cntr = 0, 0
    for item in data:
        res += len(item["output"].split())
        cntr += 1
    return res / cntr


def compute_qa(data):
    """
    Compute QA-based accuracy.
    用于计算基于问答对（QA pairs）的准确度指标，它使用了一个预训练的问答模型来评估生成的文本回答是否准确

    Args:
        data: requires filed `qa_pairs/short_answers` and `output`
            qa_pairs：包含问题和简短答案对的列表。
            output：模型生成的文本输出。
    Returns:
        QA metrics (QA-EM, QA-F1, QA-Hit)
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        logger.warn("Warning: no QA pairs found in data")
        return {
            'QA-EM': 0,
            'QA-F1': 0,
            'QA-Hit': 0,
        }

    # Load model
    # 加载模型：函数首先检查是否需要加载用于问答的RoBERTa-large SQuAD模型。如果需要，它将加载该模型。
    logger.info("Loading the RoBERTa-large SQuAD model for QA-based accuracy...")
    qa_pipeline = pipeline("question-answering", model=QA_MODEL, device=0)
    logger.info("Done")

    # Get prediction
    logger.info("Computing the QA-based accuracy...")
    # 初始化变量：函数初始化了几个列表来存储精确度(EM)、F1分数和命中次数（Hit）的局部计数。
    em, f1, bins = [], [], []
    for item in tqdm(data):
        # 从项目中获取问题列表和上下文文本。如果输出文本长度大于0，则使用输出文本作为上下文，否则使用一个空格作为上下文。
        question = [qa_pair['question'] for qa_pair in item['qa_pairs']]
        context = item['output'] if len(item['output']) > 0 else " "
        # 获取预测：使用加载的问答模型的管道（pipeline）获取问题的答案预测。管道函数允许处理不可能的答案（handle_impossible_answer=True）。
        results = qa_pipeline(question=question, context=context, handle_impossible_answer=True)
        loc_counter, loc_em, loc_f1 = 0, 0, 0

        for idx, res in enumerate(results):
            # 对于每个问题，遍历其对应的答案列表（short_answers）。
            answers = item["qa_pairs"][idx]["short_answers"]
            prediction = res["answer"]

            # 计算预测答案与每个答案之间的精确度（Exact Match）和F1分数。
            loc_em += max([compute_exact(a, prediction) for a in answers])
            loc_f1 += max([compute_f1(a, prediction) for a in answers])
            # 累加每个问题的最大精确度和F1分数，并增加计数器。
            loc_counter += 1

        em.append(loc_em / loc_counter)
        f1.append(loc_f1 / loc_counter)
        # 计算平均值：计算所有问题的最大精确度和F1分数的平均值，并确定有多少个问题的答案完全匹配（即精确度等于计数器的值）。
        bins.append(loc_em == loc_counter)

    # 返回结果：函数返回一个字典，包含以下键：
    return {
        # 'QA-EM'：所有问题的最大精确度平均值的百分比形式。
        'QA-EM': 100 * np.mean(em),
        # 'QA-F1'：所有问题的最大F1分数平均值的百分比形式。
        'QA-F1': 100 * np.mean(f1),
        # 'QA-Hit'：所有问题答案完全匹配的比例的百分比形式。
        'QA-Hit': 100 * np.mean(bins)
    }


def compute_mauve(data):
    """Compute Mauve score."""

    logger.info("Computing MAUVE...")
    human_data = []
    model_data = []
    for item in data:
        # Remove ending punctuations
        # Remove any new lines
        # Truncate by 100 words
        human_data.append(' '.join((item['question'] + " " + item['answer'].strip()).split()[:100]).rstrip(string.punctuation))
        model_data.append(' '.join((item['question'] + " " + item['output'].strip()).split()[:100]).rstrip(string.punctuation))

    import mauve
    out = mauve.compute_mauve(
        p_text=human_data,
        q_text=model_data,
        device_id=0,
        max_text_length=512,
        verbose=True,
        batch_size=8,
        featurize_model_name="gpt2-large"
    )
    return out.mauve * 100


def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference


def compute_claims(data):
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info("Computing claims...")
    scores = []
    for item in tqdm(data):
        normalized_output = remove_citations(item['output'])
        entail = 0
        claims = item["claims"]
        for claim in claims:
            entail += _run_nli_autoais(normalized_output, claim)
        scores.append(entail / len(claims))
    return 100 * np.mean(scores)


def compute_autoais(data,
                    decontext=False,
                    concat=False,
                    qampari=False,
                    at_most_citations=None,):
    """
    Compute AutoAIS score.
    用于计算 AutoAIS 分数的函数，它涉及到自然语言推断（NLI）任务，用于评估模型生成的文本是否与给定的参考文献集相符合。

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
            data 是需要评估的数据，其中应包含 output 和 docs 字段。
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output。是否去除输出中的上下文信息。
    """

    # 1. 初始化模型：如果全局变量 autoais_model 和 autoais_tokenizer 还没有被初始化，函数会加载 AutoAIS 模型和分词器。
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    # 2. 开始计算：函数打印日志信息，表明开始运行 AutoAIS。
    logger.info(f"Running AutoAIS...")

    # 3. 格式化文档：定义了一个内部函数 _format_document，用于将文档格式化为 AutoAIS 模型所需的格式。
    def _format_document(doc):
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc['title'], doc['sent'])
        else:
            return "Title: %s\n%s" % (doc['title'], doc['text'])

    # 4. 初始化变量：初始化了一些用于存储中间结果和统计信息的变量，如 ais_scores 和 ais_scores_prec。
    ais_scores = []
    ais_scores_prec = []

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []
    for item in tqdm(data):
        # Get sentences by using NLTK
        # 使用 NLTK 库对输出文本进行分句处理。
        if qampari:
            sents = [item['question'] + " " + x.strip() for x in item['output'].rstrip().rstrip(".").rstrip(",").split(",")]
        else:
            sents = sent_tokenize(item['output'])
        if len(sents) == 0:
            continue

        # 移除句子中的引用标记，并存储结果。
        target_sents = [remove_citations(sent).strip() for sent in sents]

        entail = 0
        entail_prec = 0
        total_citations = 0
        # 对每个句子，计算其与参考文献集的自然语言推断分数。
        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id] # Citation removed and (if opted for) decontextualized
            joint_entail = -1 # Undecided

            # Find references
            # 寻找句子中的引用标记，并确定引用的参考文献 ID。
            ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] # In text citation id starts from 1
            logger.info(f"For `{sent}`, find citations {ref}")
            # 如果没有引用或引用超出范围，则该句子的推断分数为0。
            if len(ref) == 0:
                # No citations
                joint_entail = 0
            elif any([ref_id >= len(item['docs']) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:  # 如果有引用且在范围内，根据引用的文献构建一个联合段落。
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = '\n'.join([_format_document(item['docs'][psgs_id]) for psgs_id in ref])

            # If not directly rejected by citation format error, calculate the recall score
            # 如果引用格式错误没有直接拒绝，则计算召回分数
            if joint_entail == -1: 
                # 7. 计算推断分数：使用 _run_nli_autoais 函数计算联合段落和目标句子之间的 NLI 分数。
                joint_entail = _run_nli_autoais(joint_passage, target_sent)
                autoais_log.append({
                    "question": item['question'],
                    "output": item['output'],
                    "claim": sent,
                    "passage": [joint_passage],
                    "model_type": "NLI",
                    "model_output": joint_entail,
                })

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            # calculate the precision score if applicable
            # 计算精度分数（如果适用）
            # 8. 计算精确度和召回率：如果句子有多个引用，函数还会计算精确度分数，即模型是否引用了不必要的文档。
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                # 计算精确度分数，即模型是否引用了不必要的文档。
                for psgs_id in ref:
                    # condition A
                    passage = _format_document(item['docs'][psgs_id]) 
                    nli_result = _run_nli_autoais(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = '\n'.join([_format_document(item['docs'][pid]) for pid in subset_exclude])
                        nli_result = _run_nli_autoais(passage, target_sent)
                        if nli_result: # psgs_id is not necessary
                            flag = 0
                            sent_mcite_overcite += 1 
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail 

        # 9. 统计信息：函数记录了多引用句子的数量、由联合引用支持的句子数量，以及过度引用（overcite）的情况。
        sent_total += len(sents)
        ais_scores.append(entail / len(sents))
        ais_scores_prec.append(entail_prec / total_citations if total_citations > 0 else 0) # len(sents))

    if sent_mcite > 0 and sent_mcite_support > 0:
        print("Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite." % (
            100 * sent_mcite / sent_total, 
            100 * sent_mcite_support / sent_mcite, 
            100 * sent_mcite_overcite / sent_mcite_support
        ))

    # 10. 返回结果：函数返回一个字典，包含两个键：
    #   - "citation_rec"：所有句子的平均召回率，乘以100得到百分比。
    #   - "citation_prec"：如果有引用，所有句子的平均精确度，乘以100得到百分比。
    return {
        "citation_rec": 100 * np.mean(ais_scores),
        "citation_prec": 100 * np.mean(ais_scores_prec),
    }


def compute_qampari_f1(data, cot=False):
    prec = []
    rec = []
    rec_top5 = []
    f1 = []
    f1_top5 = []

    num_preds = []
    for item in data:
        if cot:
            if ":" in item['output']:
                o = ':'.join(item['output'].split(":")[1:]) # try to separate the COT part and the answer list part.
            else:
                o = ""
        else:
            o = item['output']
        preds = [normalize_answer(x.strip()) for x in o.rstrip().rstrip(".").rstrip(",").split(",")]
        preds = [p for p in preds if len(p) > 0] # delete empty answers
        num_preds.append(len(preds))
        answers = [[normalize_answer(x) for x in ans] for ans in item['answers']]
        flat_answers = [item for sublist in answers for item in sublist]
        
        prec.append(sum([p in flat_answers for p in preds]) / len(preds) if len(preds) > 0 else 0)
        rec.append(sum([any([x in preds for x in a]) for a in answers]) / len(answers))
        rec_top5.append(min(5, sum([any([x in preds for x in a]) for a in answers])) / min(5, len(answers)))
        if (prec[-1] + rec[-1]) == 0:
            f1.append(0)
        else:
            f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]))
        if (prec[-1] + rec_top5[-1]) == 0:
            f1_top5.append(0) 
        else:
            f1_top5.append(2 * prec[-1] * rec_top5[-1] / (prec[-1] + rec_top5[-1]))

    return {
        "num_preds": np.mean(num_preds),
        "qampari_prec": 100 * np.mean(prec),
        "qampari_rec": 100 * np.mean(rec),
        "qampari_rec_top5": 100 * np.mean(rec_top5),
        "qampari_f1": 100 * np.mean(f1),
        "qampari_f1_top5": 100 * np.mean(f1_top5),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, required=True, help="Output file. Should have field `question`, `output`, (ROUGE) `answer`, \
                        (accuracy) `qa_pairs`, (AIS) `docs`")
    parser.add_argument("--no_rouge", action="store_true", help="Do not evaluate ROUGE score")
    parser.add_argument("--qa", action="store_true", help="Use the QA model")
    parser.add_argument("--mauve", action="store_true", help="Use the mauve score model")
    parser.add_argument("--citations", action="store_true", help="Evaluation with citation")
    parser.add_argument("--at_most_citations", type=int, default=3, help="At most take this many documents (mostly for precision)")
    parser.add_argument("--claims_nli", action="store_true", help="Use claims for ELI5")

    # QAMPARI
    parser.add_argument("--cot", action="store_true", help="For QAMPARI, try to find colon and separate the COT and answer listing")

    args = parser.parse_args()

    with open(args.f) as f:
        data_with_config = json.load(f)
    data = data_with_config['data'] 

    if "qampari" in args.f:
        args.no_rouge = True
        args.qa = False
        args.mauve = False
        args.decontext = False
        qampari = True
    else:
        qampari = False

    # Truncate by newline and remove on the fly search result
    logger.warning("We remove all the pre/appended space/newlines and we truncate the answer by the first newline.")
    logger.warning("We replace any on the fly search result to standard bracket citation format.")
    for i in range(len(data)):
        data[i]['output'] = data[i]['output'].strip().split("\n")[0]
        data[i]['output'] = data[i]['output'].replace("<|im_end|>", "")


    # Remove all citations for all non-AutoAIS evaluation
    normalized_data = copy.deepcopy(data)
    for i in range(len(normalized_data)):
        normalized_data[i]['output'] = remove_citations(normalized_data[i]['output'])

    result = {}
    result['length'] = compute_len(normalized_data)
    result['str_em'], result['str_hit'] = compute_str_em(normalized_data)
    if qampari:
        result.update(compute_qampari_f1(normalized_data, cot=args.cot))
    if not args.no_rouge:
        result['rougeLsum'] = compute_rouge(normalized_data)
    if args.qa:
        result.update(compute_qa(normalized_data))
    if args.mauve:
        result['mauve'] = compute_mauve(normalized_data)
    if args.citations: 
        result.update(compute_autoais(data, qampari=qampari, at_most_citations=args.at_most_citations))
    if args.claims_nli:
        result["claims_nli"] = compute_claims(normalized_data)

    print(result)
    with open(args.f + ".score", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
