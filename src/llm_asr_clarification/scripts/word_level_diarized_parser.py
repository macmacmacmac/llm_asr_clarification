import os
import argparse
from llm_asr_clarification import get_logger #<------ trying to experiment with using loggers for saving results
                                             #        more readable, and we dont need to use json files everywhere
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
from collections import deque

# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing=================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--msg", type=str, default="example")
    # parser.add_argument("--ami_path", type=str, default="./datasets/amicorpus")
    parser.add_argument("--ami_path", type=str, default="/group/jrwhitehill/amicorpus")

    args, _ = parser.parse_known_args(args_list)

    # Build the logger here
    # first arg is
    logger = get_logger(exp_name)    
    logger.info(
        f"{"="*100}\n\t\t\t\tRunning script: {exp_name}\n{"="*100}"
    )

    # log received args
    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(
        f"Received the following arguments:\n{received_args_log}"
    )

    #==============================================================================================
    word_level_paths = {} #A dict of {meeting name: [file paths]}
    #{
    # <meeting_name>: [file path of word level transcripts for each speaker]
    #}
    words_path = os.path.join(args.ami_path, "ami_public_manual_1.6.2/words/")
    for entry in os.scandir(words_path):
        if ".xml" in entry.name:
            meeting_name = entry.name.split(".")[0]
            meeting_path = entry.path
            if meeting_name in word_level_paths:
                word_level_paths[meeting_name].append(meeting_path)
            else:
                word_level_paths[meeting_name] = [meeting_path]

    PAUSE_THRESHOLD = 0.15

    for meeting_name in tqdm(word_level_paths):

        paths = word_level_paths[meeting_name]

        speaker_queues = {}

        for path in paths:
            try:
                speaker_id = os.path.basename(path).split(".")[1]

                words = []

                tree = ET.parse(path)
                root = tree.getroot()

                for elem in root:
                    if elem.tag != "w":
                        continue

                    if "starttime" not in elem.attrib:
                        continue

                    words.append(
                        {
                            "start": float(elem.attrib["starttime"]),
                            "end": float(elem.attrib["endtime"]),
                            "text": elem.text or "",
                            "is_punc": elem.attrib.get("punc") == "true",
                        }
                    )

                words.sort(key=lambda x: x["start"])
                speaker_queues[speaker_id] = deque(words)

            except Exception as e:
                logger.info(f"I was reading this file: {path} and got an error :(")
                logger.info(f"Error: {e}")

        segments = []

        while True:

            active_speakers = [
                (speaker, queue[0]["start"])
                for speaker, queue in speaker_queues.items()
                if queue
            ]

            if not active_speakers:
                break

            speaker = min(active_speakers, key=lambda x: x[1])[0]

            queue = speaker_queues[speaker]

            segment_tokens = []

            word = queue.popleft()
            segment_tokens.append(word)

            last_end = word["end"]

            SENTENCE_ENDINGS = {".", "?", "!"}

            while queue:

                word = queue.popleft()
                segment_tokens.append(word)

                # End segment after sentence-ending punctuation
                if word["is_punc"] and word["text"] in SENTENCE_ENDINGS:
                    break
            text = ""

            for token in segment_tokens:

                if not token["text"]:
                    continue

                if token["is_punc"]:
                    text += token["text"]
                elif not text:
                    text = token["text"]
                else:
                    text += " " + token["text"]

            if text:
                segments.append((speaker, text))

        transcript = ""

        for speaker, text in segments:
            transcript += f"[Speaker {speaker}]: {text}\n\n"

        output_path = os.path.join(
            args.ami_path,
            meeting_name,
            "transcripts",
            "parsed_diarized_gt.txt",
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript)
