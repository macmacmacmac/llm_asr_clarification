import os
import json
import torch
import numpy as np
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from llm_asr_clarification.models.ImportanceLSTM import ImportanceLSTM

# Load the SentenceTransformer model once globally for efficiency
SIM_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device="cuda" if torch.cuda.is_available() else "cpu")

class ImportanceDetector(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_important_lines(self, line_numbers: list[int]) -> list[bool]:
        pass

class RandomImportanceDetector(ImportanceDetector):
    def __init__(self, **kwargs):
        defaults = {
            'num_lines': 50
        }
        params = {**defaults, **kwargs}
        self.num_lines = params['num_lines']
        
    def get_important_lines(self, line_numbers: list[int]) -> list[bool]:
        # Return True for random `num_lines`
        mask = np.zeros(len(line_numbers), dtype=bool)
        idx = np.random.choice(len(line_numbers), min(len(line_numbers), self.num_lines), replace=False)
        mask[idx] = True
        return mask.tolist()

class GTImportanceDetector(ImportanceDetector):
    def __init__(self, **kwargs):
        defaults = {
            'meeting_path': "./shared/datasets/amicorpus/train/ES2005d/",
            'k': 5
        }
        params = {**defaults, **kwargs}
        self.meeting_path = params['meeting_path']
        self.k = params['k']
        
        # Load gold answers
        quiz_path = os.path.join(self.meeting_path, "quiz", "quiz_from_parsed_diarized_gt.json")
        with open(quiz_path, "r") as f:
            quiz = json.load(f)
        self.gold_answers = [q["correct_answer"] for q in quiz]
        
        # Load GT lines
        gt_transcript_path = os.path.join(self.meeting_path, "transcripts", "parsed_diarized_gt.txt")
        with open(gt_transcript_path, "r") as f:
            gt_content = f.read().strip()
        self.gt_lines = gt_content.split("\n")
        
    def get_important_lines(self, line_numbers: list[int]) -> list[bool]:
        # Clean segments
        clean_segments = [s.split(":")[-1] for s in self.gt_lines]
        
        segments_embeds = SIM_MODEL.encode(clean_segments, convert_to_tensor=True)
        gold_answers_embeds = SIM_MODEL.encode(self.gold_answers, convert_to_tensor=True)
        
        cos_matrix = cos_sim(gold_answers_embeds, segments_embeds).detach().cpu()
        
        _, gold_idxs = torch.topk(cos_matrix, k=self.k, dim=1)
        gold_idxs = gold_idxs.view(-1).tolist()
        
        # Assuming the total number of lines corresponds to the length of gt_lines
        total_lines = len(self.gt_lines)
        mask = [False] * total_lines
        for idx in gold_idxs:
            if idx < total_lines:
                mask[idx] = True
                
        # Return the boolean values for the requested line_numbers
        return [mask[i] for i in line_numbers]


class LSTMImportanceDetector(ImportanceDetector):
    def __init__(self, **kwargs):
        defaults = {
            'meeting_path': "./shared/datasets/amicorpus/train/ES2005d/",
            'model_path': './shared/model_weights/importance_lstm/importance_detector_student_datasets/importance_lstm.pt',
            'window_size': 10,
            'prob_threshold': 0.6771,
            'transcript_file': 'custom_transcript_gt_segments.txt'
        }
        params = {**defaults, **kwargs}
        self.meeting_path = params['meeting_path']
        self.window_size = params['window_size']
        self.prob_threshold = params['prob_threshold']
        
        transcript_path = os.path.join(self.meeting_path, "transcripts", params['transcript_file'])
        with open(transcript_path, "r") as f:
            transcript_content = f.read().strip()
        self.transcript_lines = transcript_content.split("\n")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ImportanceLSTM(num_layers=1).to(self.device)
        self.model.load_state_dict(torch.load(params['model_path'], map_location=self.device))
        self.model.eval()

    def get_important_lines(self, line_numbers: list[int]) -> list[bool]:
        clean_segments = [s.split(":")[-1] for s in self.transcript_lines]
        embeds = SIM_MODEL.encode(clean_segments, convert_to_tensor=True) # shape (N, 384)
        
        preds = []
        with torch.no_grad():
            for i in line_numbers:
                target_emb = embeds[i].unsqueeze(0).to(self.device) # (1, 384)
                
                # context is previous `window_size` lines
                start_idx = max(0, i - self.window_size)
                context_emb = embeds[start_idx:i]
                
                # pad with zeros if context is less than window_size
                if context_emb.size(0) < self.window_size:
                    pad_len = self.window_size - context_emb.size(0)
                    pad = torch.zeros((pad_len, 384), device=context_emb.device)
                    context_emb = torch.cat([pad, context_emb], dim=0)
                
                context_emb = context_emb.unsqueeze(0).to(self.device) # (1, window_size, 384)
                
                logit = self.model(context_emb, target_emb)
                prob = torch.sigmoid(logit).item()
                preds.append(prob >= self.prob_threshold)
                
        return preds
