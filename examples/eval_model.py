import argparse
import collections
import os

import lm_eval
import torch
import torch.nn as nn
from lm_eval import tasks, simple_evaluate
from lm_eval.models.huggingface import HFLM
from tqdm import tqdm
from transformers import AutoTokenizer

from QQQ.gptq.models import get_quantized_model_class
from QQQ.utils import (
    get_model_architecture,
    get_model_config,
    parse_quant_config,
    get_loaders,
    pattern_match,
    update_results,
    setup_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="path contains model weight and quant config",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="path contains tokenizer",
    )
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def eval_model(model, tokenizer, args):
    # 最大的程度
    max_length = args.max_length
    print(max_length)
    results = {}
    # eval ppl
    # 测试ppl
    if args.eval_ppl:
        for task in ["wikitext2"]:
            _, testloader = get_loaders(
                task,
                seed=0,
                model=args.tokenizer_path,
                seqlen=max_length,
            )
            if "c4" in task:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            # 获取到训练样本
            nsamples = testenc.numel() // max_length

            nlls = []
            # 计算ppl的代码
            for i in tqdm(range(nsamples)):
                batched_inps = testenc[:, (i * max_length): ((i + 1) * max_length)].to(
                    model.device
                )
                outputs = model.model(batched_inps)
                hidden_states = outputs[0]
                logits = model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * max_length): ((i + 1) * max_length)][
                               :, 1:
                               ].to(model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * max_length
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * max_length))

            result = collections.defaultdict(dict)
            versions = collections.defaultdict(dict)
            n_shot = collections.defaultdict(dict)
            result[task]["ppl"] = ppl.item()
            versions[task] = 0
            n_shot[task] = 0
            t_results = {
                "results": dict(result),
                "versions": dict(versions),
                "n-shot": dict(n_shot),
            }
            print(t_results)
            update_results(results, t_results)
    # eval other datasets
    if args.tasks != "":
        task_names = pattern_match(args.tasks.split(","), tasks.TaskManager().all_tasks)
        lm = HFLM(
            pretrained=model,
            backend="causal",
            device="cuda",
            batch_size=args.batch_size,
            tokenizer=tokenizer,
            max_lengt=max_length,
        )
        t_results = simple_evaluate(
            lm,
            tasks=task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
        )
        update_results(results, t_results)

    print(lm_eval.utils.make_table(results))


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    # 模型中自带的配置文件+解析文件参数, 文件是量化模型的时候能存储的
    config_path = os.path.join(args.model_path, "quant_config.json")
    quant_config = parse_quant_config(config_path)

    # 获取模型配置
    config = get_model_config(args.model_path)
    # 获取模型
    model_type = get_model_architecture(config)
    # 模型到量化的模型, 涉及到量化的层等信息
    quant_model_class = get_quantized_model_class(model_type)

    # 得到model, 得到tokenizer
    model = quant_model_class.from_pretrained(
        args.model_path,
        quant_config=quant_config,
        device_map="sequential",
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
    )
    # 测试模型
    eval_model(model, tokenizer, args)
