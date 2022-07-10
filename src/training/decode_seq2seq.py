"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import os
import logging
import glob
import argparse
import math
from tqdm import tqdm, trange
import numpy as np
import torch
import random
import pickle

import training.seq2seq_loader as seq2seq_loader
from model.tokenization_bart import BartTokenizer
from model.modeling_bart import BartForConditionalGeneration

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def remove_duplicate(list, return_idx=False, is_set=False):
    new_list, new_list_idx = set(), []
    for i, elem in enumerate(list):
        if is_set:
            elem = frozenset(elem.split())
        if elem not in new_list:
            new_list.add(elem)
            new_list_idx.append(i)
    if return_idx:
        return [list[i] for i in new_list_idx], new_list_idx
    else:
        return [list[i] for i in new_list_idx]


def create_input_wo_repetition(src_file):
    # copy2(src_file, src_file + '.tmp')
    with open(src_file) as f:
        srcs = [line.strip() for line in f.readlines() if len(line.strip())]
        new_srcs, idx_list = remove_duplicate(srcs, return_idx=True, is_set=False)
    new_src_file = src_file + '.tmp'
    with open(new_src_file, 'w') as fw:
        fw.write('\n'.join(new_srcs) + '\n')


def recover_output_from_unique(src_file, tgt_file, is_set=False):
    with open(src_file + '.tmp', 'r') as fr1, open(tgt_file, 'r') as fr2:
        if is_set:
            mapping = {frozenset(line_src.strip().split()): line_tgt.strip()
                 for line_src, line_tgt in zip(fr1.readlines(), fr2.readlines())
                 if len(line_src.strip())}
        else:
            mapping = {line_src.strip(): line_tgt.strip()
                 for line_src, line_tgt in zip(fr1.readlines(), fr2.readlines())
                 if len(line_src.strip())}
    # os.rename(src_file + '.tmp', src_file)
    with open(src_file) as f, open(tgt_file, 'w') as fw:
        for line in f:
            if len(line.strip()):
                if is_set:
                    fw.write(mapping[frozenset(line.strip().split())] + '\n')
                else:
                    fw.write(mapping[line.strip()]+'\n')


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bart_model", default="facebook/bart-large", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--model_recover_path", default="../../../output/train_unilm_newinital/model.15.bin", type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--ffn_type', default=0, type=int,
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")
    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")


    parser.add_argument("--input_file", type=str,
                        default="../../../dataset/final_data/commongen/commongen.test.src_alpha.txt",
                        help="Input file")  # "../../../../dataset/final_data/commongen/commongen.dev.single_alpha.txt"
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_dir", type=str, default="../../../output/train_unilm_newinital/Gen/test",
                        help="output dir")  # "../../../../output/unilm/Gen/model_base.5.bin.test"
    parser.add_argument("--output_file", type=str, default="model.15",
                        help="output file")  # "../../../../output/unilm/Gen/model_base.5.bin.test"
    parser.add_argument("--split", type=str, default="test",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", default=True, type=bool,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', default=True, type=bool,
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--new_pos_ids', action='store_true',
                        help="Use new position ids for LMs.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for decoding.")
    parser.add_argument("--do_sample", default=False, type=bool,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--t', type=float, default=None,
                        help='Temperature for sampling')
    parser.add_argument('--p', type=float, default=None,
                        help='p for Nucleus (top-p) sampling')
    parser.add_argument('--beam_size', type=int, default=5,
                        help="Beam size for searching")
    parser.add_argument('--need_score_traces', default=False, type=bool)
    parser.add_argument('--forbid_duplicate_ngrams', default=True, type=bool, )
    parser.add_argument('--forbid_ignore_word', type=str, default=".",
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=64,
                        help="maximum length of target sequence")
    parser.add_argument('--max_src_length', type=int, default=32,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument('--length_penalty', type=int, default=2.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument("--do_no_pos",
                        default=False,
                        type=bool,
                        help="Whether to use position embedding")


    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = args.output_dir + "/" + args.output_file

    if args.need_score_traces and args.beam_size <= 1:
        raise ValueError(
            "Score trace is only available for beam search with beam size > 1.")
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BartTokenizer.from_pretrained(
        args.bart_model, do_lower_case=args.do_lower_case)

    # tokenizer.max_len = args.max_seq_length

    pair_num_relation = 0
    bi_uni_pipeline = []
    bi_uni_pipeline.append(
        seq2seq_loader.Preprocess4Seq2seqDecoder(list(tokenizer.encoder.keys()), tokenizer.convert_tokens_to_ids,
                                                 args.max_src_length, max_tgt_length=args.max_tgt_length,
                                                 new_segment_ids=args.new_segment_ids,
                                                 mode="s2s", num_qkv=args.num_qkv,
                                                 s2s_special_token=args.s2s_special_token,
                                                 s2s_add_segment=args.s2s_add_segment,
                                                 s2s_share_segment=args.s2s_share_segment, pos_shift=args.pos_shift))

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 + \
                      (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 2
    sep_token = "</s>"
    cls_token = "<s>"
    pad_token = "<pad>"
    mask_token = "<mask>"

    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        [mask_token, sep_token, "[S2S_SOS]"])
    forbid_ignore_set = None
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))
    print(args.model_recover_path)

    model_recover_path = args.model_recover_path.strip()
    logger.info("***** Recover model: %s *****", model_recover_path)
    model_recover = torch.load(model_recover_path)

    create_input_wo_repetition(args.input_file)

    model = BartForConditionalGeneration.from_pretrained(args.bart_model,
                                                         state_dict=model_recover)

    del model_recover

    if args.fp16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    torch.cuda.empty_cache()
    model.eval()
    next_i = 0
    max_src_length = args.max_seq_length - 2 - args.max_tgt_length

    with open(args.input_file + '.tmp', encoding="utf-8") as fin:
        input_lines = []
        input_word_subword = []
        for src in fin.readlines():
            src_entity_id = []
            src_tk = []
            word_subword = []
            for src1_i, src1 in enumerate(src.split()):
                src_tk1 = tokenizer.tokenize(' ' + src1)
                word_subword.append(len(src_tk1))
                for tk in src_tk1:
                    src_tk.append(tk)

            input_lines.append(src_tk[:max_src_length])
            input_word_subword.append(word_subword[:max_src_length])

        if args.subset > 0:
            logger.info("Decoding subset: %d", args.subset)
            input_lines = input_lines[:args.subset]

    input_lines = sorted(list(enumerate(input_lines)),
                         key=lambda x: -len(x[1]))
    input_word_subword = [input_word_subword[i] for i, lit in input_lines]

    output_lines = [""] * len(input_lines)
    score_trace_list = [None] * len(input_lines)
    beam_search_text = [""] * len(input_lines)
    total_batch = math.ceil(len(input_lines) / args.batch_size)

    with tqdm(total=total_batch) as pbar:
        while next_i < len(input_lines):
            _chunk = input_lines[next_i:next_i + args.batch_size]
            _chunk_word_subword = input_word_subword[next_i: next_i + args.batch_size]

            buf_id = [x[0] for x in _chunk]
            buf = [x[1] for x in _chunk]
            buf_word_subword = [x for x in _chunk_word_subword]

            next_i += args.batch_size
            max_a_len = max([len(x) for x in buf])
            instances = []
            for instance in [(x, word_subword) for x, word_subword, in zip(buf, buf_word_subword)]:
                for proc in bi_uni_pipeline:
                    instances.append(proc(instance))
            with torch.no_grad():
                batch = seq2seq_loader.batch_list_to_batch_tensors(
                    instances)
                batch = [t.to(device) if t is not None else None for t in batch]
                input_ids, subword_mask, word_subword = batch
                traces = model.generate(
                    input_ids=input_ids,
                    attention_mask=subword_mask,
                    word_subword=word_subword,
                    max_length=args.max_tgt_length + len(input_ids[0]),
                    temperature=args.t,
                    top_p=args.p,
                    do_sample=args.do_sample,
                    num_return_sequences=args.num_return_sequences,
                    num_beams=args.beam_size,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    do_no_pos=args.do_no_pos,
                )

                # if args.beam_size > 1:
                #     traces = {k: v.tolist() for k, v in traces.items()}
                #     output_ids = traces['pred_seq']
                # else:
                output_ids = traces.tolist()
                output_ids = list(grouper(args.num_return_sequences, output_ids))
                for i in range(len(buf)):
                    for j in range(args.num_return_sequences):  # iterate over beam
                        w_ids = output_ids[i][j][2:]
                        output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []
                        for t in output_buf:
                            if t in (sep_token, pad_token):
                                break
                            output_tokens.append(t)
                        output_sequence = tokenizer.convert_tokens_to_string(output_tokens)
                        if args.num_return_sequences == 1:
                            output_lines[buf_id[i]] = output_sequence
                        else:
                            output_lines[buf_id[i]] += (output_sequence + '\t||\t')
                        if args.need_score_traces:
                            score_trace_list[buf_id[i]] = {
                                'scores': traces['scores'][i], 'wids': traces['wids'][i], 'ptrs': traces['ptrs'][i]}

                            for b_index in range(len(buf)):
                                beam_search_trace = traces['wids'][b_index]
                                beam_search_trace = list(np.transpose(np.array(beam_search_trace)))
                                each_sentence = []
                                for ind in range(len(beam_search_trace)):
                                    output_buf = tokenizer.convert_ids_to_tokens(list(beam_search_trace[ind]))
                                    output_tokens = []
                                    for t in output_buf:
                                        if t in (sep_token, pad_token):
                                            break
                                        output_tokens.append(t)
                                    output_sequence = ' '.join(detokenize(output_tokens))
                                    each_sentence.append(output_sequence)
                                beam_search_text[buf_id[b_index]] = each_sentence

            pbar.update(1)
    if args.output_file:
        fn_out = args.output_file
    else:
        fn_out = model_recover_path + '.' + args.split
    with open(fn_out, "w", encoding="utf-8") as fout:
        for l in output_lines:
            fout.write(l)
            fout.write("\n")

    recover_output_from_unique(args.input_file, fn_out, is_set=False)

    if args.need_score_traces:
        with open(fn_out + ".beam.pickle", "wb") as fout_trace:
            pickle.dump(beam_search_text, fout_trace)
        with open(fn_out + ".trace.pickle", "wb") as fout_trace:
            pickle.dump(
                {"version": 0.0, "num_samples": len(input_lines)}, fout_trace)
            for x in score_trace_list:
                pickle.dump(x, fout_trace)


if __name__ == "__main__":
    main()
