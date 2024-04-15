import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")


def remove_dash_from_lines(text):
    # Splitting the text into lines, removing "- " if it's at the start, and then joining back into a single string
    return "\n".join(
        line[2:] if line.startswith("- ") else line for line in text.split("\n")
    )


def clean_xsum_human_line(line):
    # Removing trailing whitespace
    line = line.rstrip()
    # Removing trailing period if it exists
    if line.endswith("."):
        line = line[:-1]
    return line


def split_text_into_lines(text):
    """Returns a new string where the sentences are separated by newlines"""
    sentences = sent_tokenize(text)
    return "\n".join(sentences)


import re


def get_clean_claude_summary(highlights):
    lines = highlights.split("\n")

    # Applying all the cleaning rules:
    # - Remove blank lines
    # - Remove lines with specific phrases or ending with a colon
    # - Remove initial numbers like "1. ", trailing whitespace, periods, and leading "- "
    cleaned_lines = []
    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespace
        if (
            line
            and not any(
                string in line.lower()
                for string in [
                    "points from the summary",
                    "highlights from the summary",
                    "summary of the article",
                    "highlights from the article",
                    "highlight summaries of the",
                ]
            )
            and not line.endswith(":")
        ):
            line = re.sub(r"^\d+\.\s+", "", line)  # Remove initial numbers like "1. "
            line = line.lstrip("* ")  # Remove leading "* "
            line = line.rstrip(".")  # Remove trailing period
            line = line.replace(
                '."\n', '"\n'
            )  # Remove trailing period before end-of-line quotation mark
            if line.endswith(
                '."'
            ):  # Remove trailing period before final quotation mark
                line = line[:-2] + line[-1]
            line = line.lstrip(
                "- "
            ).lstrip()  # Remove leading "- " and any extra whitespace
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def get_clean_cnn_human_summary(summary):
    def clean_line(line):
        line = line.rstrip(".").rstrip()  # Remove trailing period and whitespace
        if line.startswith("NEW: "):  # Remove leading "NEW: "
            line = line[5:]
        return line

    return "\n".join(clean_line(l) for l in summary.split("\n"))


def get_clean_xsum_llama_summary(summary):
    if summary[0] == '"' and summary[-1] == '"':
        summary = summary[1:-1]
    return summary.split("\n")[-1]


from data import load_from_json, save_to_json

results = load_from_json("summaries/cnn_train_llama_responses.json")
results = {k: get_clean_claude_summary(v) for k, v in results.items()}
save_to_json(results, "summaries/cnn_train_llama_responses_.json")
