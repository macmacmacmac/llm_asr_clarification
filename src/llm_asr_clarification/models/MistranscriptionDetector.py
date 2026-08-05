import os
import ipdb
from llm_asr_clarification import get_logger
from transformers import AutoTokenizer
from pathlib import Path
import re
import json
import pandas as pd
import torch
import pickle
import ipdb

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"   # or another causal LM
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class MistranscriptionDetector:
    def __init__(
            self, 
            meeting_path="./datasets/amicorpus/train/ES2005d/",
            model_path='./model_weights/rf_model.pkl'
        ):

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
            lambda x: len(tokenizer.encode(x))
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
        return pred_probas

# Driver code to test implementation
if __name__ == '__main__':
    detector = MistranscriptionDetector()
    # ipdb.set_trace()
    result = detector.pred_mistranscribed(
        [0]
    )
    print(result)
    ipdb.set_trace()