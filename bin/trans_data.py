import re
import json
import random


def trans_to_sequence_label_format(in_file, out_file):
    with open(in_file, 'r') as f, open(out_file, 'w') as out_f:
        idx = 0
        for line in f:
            idx += 1
            spans = line.strip().split('  ')
            tags = [span[-1] for span in spans]
            sent = re.sub(r'/[abco]\s\s', '=', line)
            # print(sent[:-3])
            sent_toks, sent_ners = [], []
            for tag, mention in zip(tags, sent[:-3].split('=')):
                toks = mention.split('_')
                sent_toks.extend(toks)
                if tag == 'o':
                    sent_ners.extend(['O' for _ in range(len(toks))])
                else:
                    if len(toks) == 1:  # Single-token mention
                        sent_ners.append('S-' + tag)
                    elif len(toks) == 2:  # Two-token mention
                        sent_ners.append('B-' + tag)
                        sent_ners.append('E-' + tag)
                    else:  # >=3
                        sent_ners.append('B-' + tag)  # Begin
                        for _ in range(len(toks)-2):
                            sent_ners.append('I-' + tag)  # Inside
                        sent_ners.append('E-' + tag)  # End
            obj = {
                'text': sent_toks,
                'ner': sent_ners
            }
            out_f.write(json.dumps(obj) + '\n')


def construct_train_dev(original_train_file, train_file, dev_file):
    with open(original_train_file, 'r') as f:
        lines = []
        for line in f:
            lines.append(line)
        random.shuffle(lines)
        with open(dev_file, 'w') as dev_f:
            for line in lines[:1500]:
                dev_f.write(line)
        with open(train_file, 'w') as train_f:
            for line in lines[1500:]:
                train_f.write(line)


def construct_test(in_file):
    with open(in_file, 'r') as f, open('data/test.json', 'w') as out_f:
        for line in f:
            toks = line.strip().split('_')
            sent_len = len(toks)
            ner = ' '.join(['O']*sent_len)
            obj = {
                'text': toks,
                'ner': ner  # add dummy ner tags
            }
            out_f.write(json.dumps(obj) + '\n')


trans_to_sequence_label_format('data/train.txt', 'data/train_full.json')
construct_train_dev('data/train_full.json', 'data/train.json', 'data/dev.json')
construct_test('data/test.txt')


