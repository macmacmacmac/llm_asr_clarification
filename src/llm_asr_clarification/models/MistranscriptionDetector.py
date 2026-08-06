import os
import ipdb
from llm_asr_clarification import get_logger
from transformers import AutoTokenizer
import re
import json
import pandas as pd
import torch
import pickle
import numpy as np
from jiwer import wer
import jiwer
from rouge_score import rouge_scorer
import string
from abc import ABC, abstractmethod

# ┌───────────────────────────────────────────────┐
# │               GLOBAL VARIABLES                │
# └───────────────────────────────────────────────┘
# Define a robust transformation pipeline
# This applies data cleaning steps in order, from top to bottom
JIWER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),                # Convert all text to lowercase
    jiwer.RemovePunctuation(),          # Strip characters like commas, periods, question marks
    jiwer.RemoveMultipleSpaces(),       # Turn multi-spaces into a single space
    jiwer.Strip(),                      # Clean up leading/trailing whitespaces
    jiwer.ReduceToListOfListOfWords()   # Format text tokens perfectly for jiwer's internal engine
])

ROUGE_SCORER = rouge_scorer.RougeScorer(
    ["rougeL"],
    # use_stemmer=True,
    use_stemmer=False
)

LLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"   # or another causal LM
LLM_TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)


# ┌───────────────────────────────────────────────┐
# │                 HELPER METHODS                │
# └───────────────────────────────────────────────┘
def rouge_l(pred, ref):
    return ROUGE_SCORER.score(
        ref,   # reference first
        pred   # prediction second
    )["rougeL"]


def normalize_text(text: str) -> str:
    """Mirrors the jiwer text normalization pipeline, except for last stage"""
    if not text:
        return ""
    
    # 1. jiwer.ToLowerCase()
    text = text.lower()
    
    # 2. jiwer.RemovePunctuation()
    # Removes standard punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. jiwer.RemoveMultipleSpaces()
    text = re.sub(r'\s+', ' ', text)
    
    # 4. jiwer.Strip()
    text = text.strip()
    
    return text


class MistranscriptionDetector(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def pred_mistranscribed(self, line_numbers: list[int]) -> list[bool]:
        pass


class RandomBernoulliDetector(MistranscriptionDetector):
    """
    Trivial baseline detector, returns positive predictions with probability p.
    """
    def __init__(self, p=0.592):
        """
        The default value of p here is taken from data analysis phase, which is the percent of sentences 
        across the training meetings which are mistranscribed (according to ROUGE-L < 0.5) by whisper_tiny.
        """
        self.p = p

        
    def pred_mistranscribed(self, line_numbers: list[int]) -> list[bool]:
        return (np.random.rand(len(line_numbers)) < self.p).tolist()


class GTDetector(MistranscriptionDetector):
    """
    Teacher detector based on GT used to train other models.    
    """
    def __init__(self, **kwargs):

        defaults = {
            'meeting_path' : "./shared/datasets/amicorpus/train/ES2005d/",
            'rougeL_threshold' : 0.5
        }

        params  = {**defaults, **kwargs}

        meeting_path = params['meeting_path']
        self.rougeL_threshold = params['rougeL_threshold']

        # Read the transcript using the provided path
        beam_results = os.path.join(meeting_path, "artifacts", "beam_results.json")
        with open(beam_results, "r", encoding="utf-8") as f:
            lines = f.read()
        lines = json.loads(lines)

        # Process lines
        data = []
        for line in lines:
            for i in range(1,6):
                beam_no = f'beam_{i}'
                beam = line.pop(beam_no)
    
                line[f"{beam_no}_text"] = beam["text"]
                line[f"{beam_no}_asrlogprob"] = beam["asr_avg_log_prob"]
                line[f"{beam_no}_llmlogprob"] = beam["llm_avg_log_prob"]
            line["meeting_name"] = meeting_path.split("/")[-1]

            data.append(line)

        df = pd.DataFrame(data)

        df['text'] = df['beam_1_text']

        df["rougeL"] = [
            rouge_l(normalize_text(pred), normalize_text(ref)).fmeasure
            for pred, ref in zip(df["text"], df["gt"])
        ]
        df["wer"] = [
            min(1.0, wer(reference = ref, hypothesis = pred, reference_transform = JIWER_TRANSFORM, hypothesis_transform = JIWER_TRANSFORM))
            for pred, ref in zip(df["text"], df["gt"])
        ]
        
        df['num_tokens_text'] = df['beam_1_text'].apply(
            lambda x: len(LLM_TOKENIZER.encode(x))
        )
        df['num_tokens_gt'] = df['gt'].apply(
            lambda x: len(LLM_TOKENIZER.encode(x))
        )

        self.df = df
        
    def pred_mistranscribed(self, line_numbers: list[int]):
        X_sample = self.df.iloc[line_numbers]['rougeL']
        return (X_sample < self.rougeL_threshold).tolist()


class RFDetector(MistranscriptionDetector):
    """
    Random forest detector. Trained in log_prob_beams_elbowplot.ipynb
    """
    def __init__(self, **kwargs):

        defaults = {
            'meeting_path' : "./shared/datasets/amicorpus/train/ES2005d/",
            'model_path' : './model_weights/rf_model.pkl',
            'prob_threshold' : 0.3583 # Random Forest output prob Threshold determined using data analysis
        }

        params  = {**defaults, **kwargs}

        meeting_path = params['meeting_path']
        model_path = params['model_path']
        self.prob_threshold = params['prob_threshold']

        # Read the transcript using the provided path
        beam_results = os.path.join(meeting_path, "artifacts", "beam_results.json")
        with open(beam_results, "r", encoding="utf-8") as f:
            lines = f.read()
        lines = json.loads(lines)

        # Process lines
        data = []
        for line in lines:
            for i in range(1,6):
                beam_no = f'beam_{i}'
                beam = line.pop(beam_no)
    
                line[f"{beam_no}_text"] = beam["text"]
                line[f"{beam_no}_asrlogprob"] = beam["asr_avg_log_prob"]
                line[f"{beam_no}_llmlogprob"] = beam["llm_avg_log_prob"]
            line["meeting_name"] = meeting_path.split("/")[-1]

            data.append(line)

        df = pd.DataFrame(data)
        
        llm_logprob_columns = [f'beam_{i}_llmlogprob' for i in range(1,6)]
        asr_logprob_columns = [f'beam_{i}_asrlogprob' for i in range(1,6)]
        
        llm_logprobs = torch.tensor(df[llm_logprob_columns].values)
        asr_logprobs = torch.tensor(df[asr_logprob_columns].values)
        
        # # ALL BEAMS
        # scores = F.softmax(ALPHA*llm_logprobs + 0.0asr_logprobs, dim=1)
        # highest_score_idxs = torch.argmax(scores, dim=1, keepdim=True)
        # highest_scores = torch.gather(scores, dim=1, index=highest_score_idxs)

        # JUST BEAM 1
        # scores = llm_logprobs
        # highest_scores = scores[torch.arange(scores.size(0)), torch.zeros(scores.size(0), dtype=torch.long)]


        # AGGREGATED STATS
        df['max_llmlogprobs'] = torch.max(llm_logprobs, dim=1).values.numpy()
        df['min_llmlogprobs'] = torch.min(llm_logprobs, dim=1).values.numpy()
        df['spread_llmlogprobs'] = df['max_llmlogprobs'] - df['min_llmlogprobs']

        df['max_asrlogprobs'] = torch.max(asr_logprobs, dim=1).values.numpy()
        df['min_asrlogprobs'] = torch.min(asr_logprobs, dim=1).values.numpy()
        df['spread_asrlogprobs'] = df['max_asrlogprobs'] - df['min_asrlogprobs']
        
        df['text'] = df['beam_1_text']
        df['num_tokens_text'] = df['beam_1_text'].apply(
            lambda x: len(LLM_TOKENIZER.encode(x))
        )
        # df['num_tokens_gt'] = df['gt'].apply(
        #     lambda x: len(tokenizer.encode(x))
        # )

        self.df = df
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
            self.features = self.model.feature_names_in_

        
    def pred_mistranscribed(self, line_numbers: list[int]):
        X_sample = self.df.iloc[line_numbers][self.features]
        pred_probas = self.model.predict_proba(X_sample)[:,1]
        return (pred_probas >= self.prob_threshold).tolist()



# Driver code to test implementation
if __name__ == '__main__':
    random = RandomBernoulliDetector()
    gt = GTDetector()
    rf = RFDetector()
    ipdb.set_trace()