import os
import argparse
from llm_asr_clarification import get_logger #<------ trying to experiment with using loggers for saving results
                                             #        more readable, and we dont need to use json files everywhere
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm

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

    for meeting_name in tqdm(word_level_paths):
        # this is a path to n different XML files
        paths = word_level_paths[meeting_name]

        # TODO: open all the xml files and combine them into one long string that just has the words
        tokens = []

        for path in paths:
            try:
                tree = ET.parse(path)
                root = tree.getroot()
                for elem in root:
                    if elem.tag != "w":
                        continue
                    tokens.append(
                        (
                            float(elem.attrib["starttime"]),
                            elem.text or "",
                            elem.attrib.get("punc") == "true",
                        )
                    )
            except Exception as e:
                logger.info(f"I was reading this file: {path} and got an error :(")
                logger.info(f"Error: {e}")

            tokens.sort(key=lambda x: x[0])
            transcript = ""
            for _, text, is_punc in tokens:
                if not text:
                    continue

                if is_punc:
                    transcript += text
                elif not transcript:
                    transcript = text
                else:
                    transcript += " " + text

        # output path
        output_path = os.path.join(args.ami_path, meeting_name, "transcripts", "parsed_gt.txt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        # try:
        #     output_path = os.path.join(args.ami_path, meeting_name, "transcripts", "parsed_gt.txt")
        #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #     with open(output_path, "w", encoding="utf-8") as f:
        #         f.write(transcript)
        # except Exception as e:
        #     logger.info(f"I was trying to write this file: {output_path} and got an error :(")
        #     logger.info(f"Error: {e}")




    
