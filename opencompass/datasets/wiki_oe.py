# flake8: noqa
# yapf: disable

import json
import os
import re

from datasets import Dataset, DatasetDict, load_dataset
# from configs._multilingual_24July.datasets.WIKI_OE.prompts import \
#     judge_prompt_template
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import detect_language, first_option_postprocess
from opencompass.utils.prompt import PromptList
from opencompass.utils import detect_language, translate_text
from opencompass.models import OpenAI
from .base import BaseDataset

class WIKI_OE_Dataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        f_path = kwargs.get('path', None)
        src_lan = kwargs.get('src_lan', None)
        ques_lan = kwargs.get('ques_lan', None)
        dataset = DatasetDict()
        f = open(f_path, 'r', encoding='utf-8')
        lines = f.readlines()
        objs = []
        for line in lines:
            obj = json.loads(line)
            objs.append(obj)
        out_dict_list = []
        for obj in objs:
            obj['ques_lan'] = ques_lan
            obj['is_reverse'] = False
            if f"question_{ques_lan}" in obj:
                new_obj = {
                    'out': obj,
                    'question': obj[f'question_{ques_lan}'].strip()
                }
                out_dict_list.append(new_obj)
            reverse_obj = obj.copy()
            if f"question_reverse_{ques_lan}" in reverse_obj:
                reverse_obj['is_reverse'] = True
                new_reverse_obj = {
                    'out': reverse_obj,
                    'question': obj[f'question_reverse_{ques_lan}'].strip()
                }
                out_dict_list.append(new_reverse_obj)
        dataset = Dataset.from_list(out_dict_list)
        return dataset



class WIKI_OE_Evaluator(BaseEvaluator):
    """
    ref: opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator
    """
    def __init__(self, judge_model_path='gpt-4o-2024-11-20', judge_prompt_template={}, **kwargs):
        super().__init__(**kwargs)
        self.judge_model_path = judge_model_path
        self.judge_prompt_template = judge_prompt_template

    def score(self, predictions, references, origin_prompt) -> dict:

        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length.'}

        details = {}
        total, n_lang_consistent, n_lang_inconsistent, n_lang_faildetect, correct, wrong, unclear = 0, 0, 0, 0, 0, 0, 0  # noqa
        
        judge_model = OpenAI(path=self.judge_model_path,
                       max_seq_len=16384,
                       query_per_second=1,
                       retry=2,
                       temperature=0.0)
        
        for index, (raw_pred, meta_info, prompt) in enumerate(zip(predictions, references, origin_prompt)):
            total += 1
            lang = meta_info['lan']
            qid = meta_info['qid']
            answer = meta_info[f'answer_{lang}']
            question = meta_info[f'question_{lang}']
            detail = {
                'meta_info': meta_info,
                'qid': qid,
                'prompt':prompt,
                'question': question,
                'llm_answer': raw_pred,
                'en_llm_answer': '',
                'zh_llm_answer': '',
                'ref_answer': answer,
                'lang': lang,
                'is_correct': False,
                'is_lang_inconsistent': False,
                'is_lang_faildetect': False
            }
            # ------------------------------------------- check pred lang
            pred_lang = detect_language(raw_pred)
            if pred_lang == 'other':
                n_lang_faildetect += 1
                detail['is_lang_faildetect'] = True
            elif pred_lang == lang:
                n_lang_consistent += 1
            else:
                n_lang_inconsistent += 1
                detail['is_lang_inconsistent'] = True

            # ------------------------------------------- translate to en
            if pred_lang != 'en':
                en_pred = translate_text(
                    raw_pred,
                    target_lang='EN-US',
                    source_lang=pred_lang if pred_lang != 'other' else None)
            else:
                en_pred = raw_pred
            if pred_lang != 'zh':
                zh_pred = translate_text(
                    raw_pred,
                    target_lang='ZH',
                    source_lang=pred_lang if pred_lang != 'other' else None)
            else:
                zh_pred = raw_pred
            detail['en_llm_answer'] = en_pred
            detail['zh_llm_answer'] = zh_pred

            # ------------------------------------------- openai judge
            if 'en_user' not in self.judge_prompt_template:
                return {'error': 'please add \' en_user \' template in your judge_prompt_template.'}
            user_prompt = self.judge_prompt_template['en_user'].format(
                question=question,
                answer=answer,
                raw_pred=raw_pred)
            system_prompt = self.judge_prompt_template['en_system'] if 'en_system' in self.judge_prompt_template else ''
            detail['judge_user_prompt'] = user_prompt

            messages = PromptList([{
                'role': 'SYSTEM',
                'prompt': system_prompt,
            }, {
                'role': 'HUMAN',
                'prompt': user_prompt,
            }])
            response = judge_model._generate(input=messages,
                                       max_out_len=2048,
                                       temperature=0.0)
            detail['judge_resp'] = response
            if 'yes' in response.lower() and 'no' not in response.lower():
                correct += 1
                detail['is_correct'] = True
            elif 'no' in response.lower() and 'yes' not in response.lower():
                wrong += 1
            else:
                unclear += 1
                detail['ans_fail_parse'] = True
            details[str(index)] = detail

        assert total == correct + wrong + unclear
        results = {
            'correct_ratio': correct / total * 100,
            'incorrect_ratio': wrong / total * 100,
            'ans_fail_parse_ratio': unclear / total * 100,
            'lang_consistent_ratio': n_lang_consistent / total * 100,
            'lang_inconsistent_ratio': n_lang_inconsistent / total * 100,
            'lang_faildetect_ratio': n_lang_faildetect / total * 100,
            'details': details
        }
        return results

class WIKI_OE_Evaluator_V2(BaseEvaluator):
    """
    ref: opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator
    """
    def __init__(self, judge_model_path='gpt-4o-2024-11-20', judge_prompt_template={}, **kwargs):
        super().__init__(**kwargs)
        self.judge_model_path = judge_model_path
        self.judge_prompt_template = judge_prompt_template

    def score(self, predictions, references, origin_prompt) -> dict:

        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length.'}

        # total, n_lang_consistent, n_lang_inconsistent, n_lang_faildetect, correct, wrong, not_attempted, unclear = 0, 0, 0, 0, 0, 0, 0, 0  # noqa
        
        judge_model = OpenAI(path=self.judge_model_path,
                             key='ENV',
                             openai_proxy_url='ENV',
                             max_seq_len=32768,
                             query_per_second=1,
                             retry=2,
                             temperature=0.0)

        details = {}
        result_dict = {}
        for index, (raw_pred, meta_info, prompt) in enumerate(zip(predictions, references, origin_prompt)):
            raw_pred_json = {"answer":"C", "confidence_score":0}
            match = re.search(r'\{.*?\}', raw_pred, re.DOTALL)
            if match:
                json_part = match.group()  
                try:
                    raw_pred_json = json.loads(json_part)  # 尝试解析 JSON
                except json.JSONDecodeError as e:
                    raw_pred_json = {"answer":"C", "confidence_score":0}

            lang = meta_info['ques_lan']
            qid = meta_info['qid']
            knowledge_track_id = meta_info['knowledge_track_id']
            is_reverse = meta_info['is_reverse']
            confidence = raw_pred_json['confidence_score']
            llm_answer = raw_pred_json['answer']

            if knowledge_track_id not in result_dict: 
                result_dict[knowledge_track_id] = {}

            question_key = f'question_{lang}' if not is_reverse else f'question_reverse_{lang}'
            answer_key = f'answer_{lang}' if not is_reverse else f'answer_reverse_{lang}'


            answer = meta_info[answer_key]
            question = meta_info[question_key]
            result_dict[knowledge_track_id][question_key] = {}
            result_dict[knowledge_track_id][question_key]['confidence'] = confidence
            
            detail = {
                'meta_info': meta_info,
                'qid': qid,
                'knowledge_track_id': knowledge_track_id,
                'prompt':prompt,
                'question': question,
                'llm_answer': llm_answer,
                'llm_answer_confidence': confidence,
                'en_llm_answer': '',
                'zh_llm_answer': '',
                'ref_answer': answer,
                'lang': lang,
                'is_correct': False,
                'is_not_attempted': False,
            }
            
            # ------------------------------------------- translate to en
            pred_lang = detect_language(raw_pred)
            raw_pred = raw_pred if raw_pred != '' else "I don't know the answer."
            if pred_lang != 'en':
                en_pred = translate_text(
                    raw_pred,
                    target_lang='EN-US',
                    source_lang=pred_lang if pred_lang != 'other' else None)
            else:
                en_pred = raw_pred
            if pred_lang != 'zh':
                zh_pred = translate_text(
                    raw_pred,
                    target_lang='ZH',
                    source_lang=pred_lang if pred_lang != 'other' else None)
            else:
                zh_pred = raw_pred
            detail['en_llm_answer'] = en_pred
            detail['zh_llm_answer'] = zh_pred

            # ------------------------------------------- openai judge
            user_prompt = self.judge_prompt_template.format(
                question=question,
                target=answer,
                predicted_answer=raw_pred)
            detail['judge_prompt'] = user_prompt

            messages = PromptList([{
                'role': 'HUMAN',
                'prompt': user_prompt,
            }])
            response = judge_model._generate(input=messages,
                                       max_out_len=2048,
                                       temperature=0.0)
            detail['judge_resp'] = response
            if response.lower().startswith('a'):
                result_dict[knowledge_track_id][question_key]['result'] = 'CORRECT'
                detail['is_correct'] = True
            elif response.lower().startswith('b'):
                result_dict[knowledge_track_id][question_key]['result'] = 'INCORRECT'
            elif response.lower().startswith('c'):
                result_dict[knowledge_track_id][question_key]['result'] = 'NOT_ATTEMPTED'
                detail['is_not_attepmted'] = True
            else:
                result_dict[knowledge_track_id][question_key]['result'] = 'NOT_ATTEMPTED'
                detail['is_not_attepmted'] = True
            details[str(index)] = detail

        # assert total == correct + wrong + unclear + not_attempted
        # correct_ratio = correct / total * 100
        # incorrect_ratio = wrong / total * 100
        # not_attempted_ratio = not_attempted / total * 100
        # correct_given_attempted_ratio = correct / (total - not_attempted) * 100
        # F_score = 2 / (1 / correct_ratio + 1 / correct_given_attempted_ratio)
        results = {
            'correct_ratio(all_ques)': 0,
            'incorrect_ratio(all_ques)': 0,
            'not_attempted_ratio(all_ques)': 0,
            'correct_given_attempted_ratio(all_ques)': 0,
            'F-score(all_ques)': 0,
            'confidence(all_ques)': 0,
            'correct_ratio(ques)': 0,
            'incorrect_ratio(ques)': 0,
            'not_attempted_ratio(ques)': 0,
            'correct_given_attempted_ratio(ques)': 0,
            'F-score(ques)': 0,
            'confidence(ques)': 0,
            'correct_ratio(reverse_ques)': 0,
            'incorrect_ratio(reverse_ques)': 0,
            'not_attempted_ratio(reverse_ques)': 0,
            'correct_given_attempted_ratio(reverse_ques)': 0,
            'F-score(reverse_ques)': 0,
            'confidence(reverse_ques)': 0,
            'ques_and_reverse_ques_num': 0,
            'ques_true_reverse_ques_true_num': 0,
            'ques_true_reverse_ques_false_num': 0,
            'ques_false_reverse_ques_true_num': 0,
            'ques_false_reverse_ques_false_num': 0,
            'details': details
        }
        all_ques_num = 0
        correct_all_ques_num = 0
        incorrect_all_ques_num = 0
        not_attempted_all_ques_num = 0
        all_ques_confidence_sum = 0

        ques_num = 0
        correct_ques_num = 0
        incorrect_ques_num = 0
        not_attempted_ques_num = 0
        ques_confidence_sum = 0

        reverse_ques_num = 0
        correct_reverse_ques_num = 0
        incorrect_reverse_ques_num = 0
        not_attempted_reverse_ques_num = 0
        reverse_ques_confidence_sum = 0

        ques_and_reverse_ques_num = 0
        ques_true_reverse_ques_true_num = 0
        ques_true_reverse_ques_false_num = 0
        ques_false_reverse_ques_true_num = 0
        ques_false_reverse_ques_false_num = 0

        for knowledge_track_id, ques_dict in result_dict.items():
            if f"question_{lang}" in ques_dict and f"question_reverse_{lang}" in ques_dict:
                ques_and_reverse_ques_num += 1
                all_ques_num += 2
                all_ques_confidence_sum += int(ques_dict[f"question_{lang}"]['confidence'])
                all_ques_confidence_sum += int(ques_dict[f"question_reverse_{lang}"]['confidence'])

                ques_is_true = False
                if ques_dict[f"question_{lang}"]['result'] == 'CORRECT':
                    correct_all_ques_num += 1
                    correct_ques_num += 1
                    ques_is_true = True
                elif ques_dict[f"question_{lang}"]['result'] == 'INCORRECT':
                    incorrect_all_ques_num += 1
                    incorrect_ques_num += 1
                elif ques_dict[f"question_{lang}"]['result'] == 'NOT_ATTEMPTED':
                    not_attempted_all_ques_num += 1
                    not_attempted_ques_num += 1
                ques_num += 1
                ques_confidence_sum += int(ques_dict[f"question_{lang}"]['confidence'])

                reverse_ques_is_true = False
                if ques_dict[f"question_reverse_{lang}"]['result'] == 'CORRECT':
                    correct_all_ques_num += 1
                    correct_reverse_ques_num += 1
                    reverse_ques_is_true = True
                elif ques_dict[f"question_reverse_{lang}"]['result'] == 'INCORRECT':
                    incorrect_all_ques_num += 1
                    incorrect_reverse_ques_num += 1
                elif ques_dict[f"question_reverse_{lang}"]['result'] == 'NOT_ATTEMPTED':
                    not_attempted_all_ques_num += 1
                    not_attempted_reverse_ques_num += 1
                reverse_ques_num += 1
                reverse_ques_confidence_sum += int(ques_dict[f"question_reverse_{lang}"]['confidence'])

                if ques_is_true and reverse_ques_is_true:
                    ques_true_reverse_ques_true_num += 1
                elif ques_is_true and not reverse_ques_is_true:
                    ques_true_reverse_ques_false_num += 1
                elif not ques_is_true and reverse_ques_is_true:
                    ques_false_reverse_ques_true_num += 1
                else:
                    ques_false_reverse_ques_false_num += 1
            elif f"question_{lang}" in ques_dict:
                all_ques_num += 1
                # confidence = 0
                # try:
                #     confidence = 
                # except ValueError:
                #     confidence = 0   
                all_ques_confidence_sum += int(ques_dict[f"question_{lang}"]['confidence'])

                if ques_dict[f"question_{lang}"]['result'] == 'CORRECT':
                    correct_all_ques_num += 1
                    correct_ques_num += 1
                elif ques_dict[f"question_{lang}"]['result'] == 'INCORRECT':
                    incorrect_all_ques_num += 1
                    incorrect_ques_num += 1
                elif ques_dict[f"question_{lang}"]['result'] == 'NOT_ATTEMPTED':
                    not_attempted_all_ques_num += 1
                    not_attempted_ques_num += 1
                ques_num += 1
                ques_confidence_sum += int(ques_dict[f"question_{lang}"]['confidence'])
            elif f"question_reverse_{lang}" in ques_dict:
                all_ques_num += 1
                all_ques_confidence_sum += int(ques_dict[f"question_reverse_{lang}"]['confidence'])

                if ques_dict[f"question_reverse_{lang}"]['result'] == 'CORRECT':
                    correct_all_ques_num += 1
                    correct_reverse_ques_num += 1
                elif ques_dict[f"question_reverse_{lang}"]['result'] == 'INCORRECT':
                    incorrect_all_ques_num += 1
                    incorrect_reverse_ques_num += 1
                elif ques_dict[f"question_reverse_{lang}"]['result'] == 'NOT_ATTEMPTED':
                    not_attempted_all_ques_num += 1
                    not_attempted_reverse_ques_num += 1
                reverse_ques_num += 1
                reverse_ques_confidence_sum += int(ques_dict[f"question_reverse_{lang}"]['confidence'])
        
        results['correct_ratio(all_ques)'] = correct_all_ques_num / all_ques_num * 100 if all_ques_num != 0 else 0
        results['incorrect_ratio(all_ques)'] = incorrect_all_ques_num / all_ques_num * 100 if all_ques_num != 0 else 0
        results['not_attempted_ratio(all_ques)'] = not_attempted_all_ques_num / all_ques_num * 100 if all_ques_num != 0 else 0
        results['correct_given_attempted_ratio(all_ques)'] = correct_all_ques_num / (all_ques_num - not_attempted_all_ques_num) * 100 if (all_ques_num - not_attempted_all_ques_num) != 0 else 0
        results['F-score(all_ques)'] = 2 / (1 / results['correct_ratio(all_ques)'] + 1 / results['correct_given_attempted_ratio(all_ques)']) if results['correct_ratio(all_ques)'] != 0 and results['correct_given_attempted_ratio(all_ques)'] != 0 else 0
        results['confidence(all_ques)'] = all_ques_confidence_sum / all_ques_num if all_ques_num != 0 else 0
        results['correct_ratio(ques)'] = correct_ques_num / ques_num * 100 if ques_num != 0 else 0
        results['incorrect_ratio(ques)'] = incorrect_ques_num / ques_num * 100 if ques_num != 0 else 0
        results['not_attempted_ratio(ques)'] = not_attempted_ques_num / ques_num * 100 if ques_num != 0 else 0
        results['correct_given_attempted_ratio(ques)'] = correct_ques_num / (ques_num - not_attempted_ques_num) * 100 if (ques_num - not_attempted_ques_num) != 0 else 0
        results['F-score(ques)'] = 2 / (1 / results['correct_ratio(ques)'] + 1 / results['correct_given_attempted_ratio(ques)']) if results['correct_ratio(ques)'] != 0 and results['correct_given_attempted_ratio(ques)'] != 0 else 0
        results['confidence(ques)'] = ques_confidence_sum / ques_num if ques_num != 0 else 0
        results['correct_ratio(reverse_ques)'] = correct_reverse_ques_num / reverse_ques_num * 100 if reverse_ques_num != 0 else 0
        results['incorrect_ratio(reverse_ques)'] = incorrect_reverse_ques_num / reverse_ques_num * 100 if reverse_ques_num != 0 else 0
        results['not_attempted_ratio(reverse_ques)'] = not_attempted_reverse_ques_num / reverse_ques_num * 100 if reverse_ques_num != 0 else 0
        results['correct_given_attempted_ratio(reverse_ques)'] = correct_reverse_ques_num / (reverse_ques_num - not_attempted_reverse_ques_num) * 100 if (reverse_ques_num - not_attempted_reverse_ques_num) != 0 else 0
        results['F-score(reverse_ques)'] = 2 / (1 / results['correct_ratio(reverse_ques)'] + 1 / results['correct_given_attempted_ratio(reverse_ques)']) if results['correct_ratio(reverse_ques)'] != 0 and results['correct_given_attempted_ratio(reverse_ques)'] != 0 else 0
        results['confidence(reverse_ques)'] = reverse_ques_confidence_sum / reverse_ques_num if reverse_ques_num != 0 else 0
        results['ques_and_reverse_ques_num'] = ques_and_reverse_ques_num
        results['ques_true_reverse_ques_true_num'] = ques_true_reverse_ques_true_num
        results['ques_true_reverse_ques_false_num'] = ques_true_reverse_ques_false_num
        results['ques_false_reverse_ques_true_num'] = ques_false_reverse_ques_true_num
        results['ques_false_reverse_ques_false_num'] = ques_false_reverse_ques_false_num

        return results
