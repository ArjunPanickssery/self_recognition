from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from pprint import pprint
import json

from prompts import (
    DATASET_SYSTEM_PROMPTS,
    COMPARISON_PROMPT_TEMPLATE,
    COMPARISON_SYSTEM_PROMPT,
    DETECTION_PROMPT_TEMPLATE,
    DETECTION_PROMPT_TEMPLATE_VS_HUMAN,
    DETECTION_PROMPT_TEMPLATE_VS_MODEL,
    DETECTION_SYSTEM_PROMPT,
    COMPARISON_PROMPT_TEMPLATE_WITH_SOURCES,
    COMPARISON_PROMPT_TEMPLATE_WITH_WORSE,
    SCORING_SYSTEM_PROMPT,
    RECOGNITION_SYSTEM_PROMPT,
    RECOGNITION_PROMPT_TEMPLATE,
)

GPT_MODEL_ID = {
    "gpt4": "gpt-4-1106-preview",
    "gpt35": "gpt-3.5-turbo-1106",
    "xsum_2_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8nc8TgDp",
    "xsum_10_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8nYmytb4",
    "xsum_500_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8kP7i66k",
    "xsum_always_1_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8nZloDpW",
    "xsum_random_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8nZloDpW",
    "xsum_readability_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8oLO7FOF",
    "xsum_length_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8ooNDQYs",
    "xsum_vowelcount_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8ooNNbtT",
    "cnn_2_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8rX9zfcC",
    "cnn_10_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8rXDPMYM",
    "cnn_500_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8rYivqW8",
    "cnn_always_1_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8rYwud4k",
    "cnn_random_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8rYvYVKD",
    "cnn_readability_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8rbOOAw9",
    "cnn_length_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8rbPCDli",
    "cnn_vowelcount_ft_gpt35": "ft:gpt-3.5-turbo-1106:nyu-arg::8raOJ2nT",
}

load_dotenv()
openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()


def get_gpt_summary(article, dataset, model) -> str:
    history = [
        {"role": "system", "content": DATASET_SYSTEM_PROMPTS[dataset]},
        {
            "role": "user",
            "content": f"Article:\n{article}\n\nProvide only the summary with no other text.",
        },
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=history,
        max_tokens=100,
        temperature=0,
    )
    return response.choices[0].message.content


def get_summary(article, dataset, model):
    if model == "claude":
        return get_claude_summary(
            article,
            dataset,
        )
    if model == "gpt4":
        return get_gpt_summary(article, dataset, model="gpt-4-1106-preview")
    if model.endswith("gpt35"):
        return (
            get_gpt_summary(
                article,
                dataset,
                model=GPT_MODEL_ID[model],
            ),
        )


def get_claude_summary(article, dataset="xsum"):
    response_type = "highlights" if dataset in ["cnn", "dailymail"] else "summary"
    message = anthropic_client.beta.messages.create(
        model="claude-2.1",
        max_tokens=100,
        system=DATASET_SYSTEM_PROMPTS[dataset],
        messages=[
            {
                "role": "user",
                "content": f"Article:\n{article}\n\nProvide only the {response_type} with NO other text.",
            }
        ],
    )
    return message.content[0].text


def get_claude_choice(summary1, summary2, article, choice_type) -> str:
    match choice_type:
        case "comparison":
            prompt = COMPARISON_PROMPT_TEMPLATE.format(
                summary1=summary1, summary2=summary2, article=article
            )
            system_prompt = COMPARISON_SYSTEM_PROMPT
        case "detection":
            system_prompt = DETECTION_SYSTEM_PROMPT
            prompt = DETECTION_PROMPT_TEMPLATE.format(
                summary1=summary1, summary2=summary2, article=article
            )

    message = anthropic_client.beta.messages.create(
        model="claude-2.1",
        max_tokens=10,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def get_gpt_choice(
    summary1,
    summary2,
    article,
    choice_type,
    model="gpt4-1106-preview",
    return_logprobs=False,
) -> str:
    match choice_type:
        case "comparison":
            prompt = COMPARISON_PROMPT_TEMPLATE.format(
                summary1=summary1, summary2=summary2, article=article
            )
            system_prompt = COMPARISON_SYSTEM_PROMPT
        case "comparison_with_worse":
            prompt = COMPARISON_PROMPT_TEMPLATE_WITH_WORSE.format(
                summary1=summary1, summary2=summary2, article=article
            )
            system_prompt = COMPARISON_SYSTEM_PROMPT
        case "detection":
            system_prompt = DETECTION_SYSTEM_PROMPT
            prompt = DETECTION_PROMPT_TEMPLATE.format(
                summary1=summary1, summary2=summary2, article=article
            )
        case "detection_vs_human":
            system_prompt = DETECTION_SYSTEM_PROMPT
            prompt = DETECTION_PROMPT_TEMPLATE_VS_HUMAN.format(
                summary1=summary1, summary2=summary2, article=article
            )
        case "detection_vs_model":
            system_prompt = DETECTION_SYSTEM_PROMPT
            prompt = DETECTION_PROMPT_TEMPLATE_VS_MODEL.format(
                summary1=summary1, summary2=summary2, article=article
            )

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=history,
        max_tokens=10,
        temperature=0,
        logprobs=True,
        top_logprobs=2,
    )
    if return_logprobs:
        return response.choices[0].logprobs.content[0].top_logprobs
    return response.choices[0].message.content


def get_model_choice(
    summary1, summary2, article, choice_type, model, return_logprobs=False
):
    if model == "claude":
        return get_claude_choice(
            summary1,
            summary2,
            article,
            choice_type,
            return_logprobs=return_logprobs,
        )
    if model == "gpt4":
        return get_gpt_choice(
            summary1,
            summary2,
            article,
            choice_type,
            model="gpt-4-1106-preview",
            return_logprobs=return_logprobs,
        )
    if model.endswith("gpt35"):
        return get_gpt_choice(
            summary1,
            summary2,
            article,
            choice_type,
            model=GPT_MODEL_ID[model],
            return_logprobs=return_logprobs,
        )


def get_gpt_choice_logprobs_with_sources(
    summary1, summary2, source1, source2, article, model
) -> dict:
    prompt = COMPARISON_PROMPT_TEMPLATE_WITH_SOURCES.format(
        summary1=summary1,
        summary2=summary2,
        source1=source1,
        source2=source2,
        article=article,
    )
    system_prompt = COMPARISON_SYSTEM_PROMPT
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=history,
        max_tokens=1,
        temperature=0,
        logprobs=True,
        top_logprobs=2,
    )
    return response.choices[0].logprobs.content[0].top_logprobs


def get_logprobs_choice_with_sources(
    summary1, summary2, source1, source2, article, model
):
    if model == "gpt4":
        return get_gpt_choice_logprobs_with_sources(
            summary1, summary2, source1, source2, article, "gpt-4-1106-preview"
        )
    if model.endswith("gpt35"):
        return get_gpt_choice_logprobs_with_sources(
            summary1,
            summary2,
            source1,
            source2,
            article,
            GPT_MODEL_ID[model],
        )


def get_gpt_recognition_logprobs(summary, article, model) -> dict:
    history = [
        {"role": "system", "content": RECOGNITION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RECOGNITION_PROMPT_TEMPLATE.format(
                article=article, summary=summary
            ),
        },
    ]

    response = openai_client.chat.completions.create(
        model=GPT_MODEL_ID[model],
        messages=history,
        max_tokens=10,
        temperature=0,
        logprobs=True,
        top_logprobs=2,
    )
    return response.choices[0].logprobs.content[0].top_logprobs


def get_gpt_score(summary, article, model):
    history = [
        {"role": "system", "content": SCORING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Article:\n{article}\n\nSummary:\n{summary}\n\nProvide only the score with no other text.",
        },
    ]

    response = openai_client.chat.completions.create(
        model=GPT_MODEL_ID[model],
        messages=history,
        max_tokens=1,
        temperature=0,
        logprobs=True,
        top_logprobs=5,
    )
    return response.choices[0].logprobs.content[0].top_logprobs
