import os
import ipdb
from llm_asr_clarification import get_logger
from pathlib import Path
import re

class OracleTranscript:
    def __init__(
            self, 
            transcript_path="./datasets/amicorpus/ES2005d/transcripts/parsed_diarized_gt.txt",
            logger=None
        ):

        # Setup logger
        if logger is None:
            self.logger = get_logger(Path(__file__).name) 

        # Read the transcript using the provided path
        try:
            with open(transcript_path, "r") as f:
                self.lines = f.readlines()
        except Exception as err:
            self.logger.error(f"Error while reading {transcript_path}", exc_info=True)

        
    def get_oracle_transcription(self, start_time: int, end_time: int):
        oracle_lines = []

        # For each line
        for line in self.lines:
            # Get start time, end time, and text for the line
            oracle_start, oracle_end, oracle_text = self._extract_content(line)

            # Calculate the intersection of the two time windows
            overlap = max(0, min(oracle_end, end_time) - max(oracle_start, start_time))

            if overlap:
                oracle_lines.append(oracle_text)

        return oracle_lines
    

    def get_lines(self):
        return self.lines

            
    def _extract_content(self, line):
        # Extract all sequences of digits from the string
        numbers = re.findall(r'\d+', line)

        # Get the first two numbers and convert them to integers
        start_time = int(numbers[0])
        end_time = int(numbers[1])

        # Get the line text
        text = line.split(')')[-1]

        return start_time, end_time, text
    

# Driver code to test implementation
if __name__ == '__main__':
    oracle_transcript = OracleTranscript()

    result = oracle_transcript.get_oracle_transcription(
        start_time=135,
        end_time=138
    )

    print(result)