from data import load_data, save_to_json
from prompts import (
    DETECTION_PROMPT_TEMPLATE,
    DETECTION_SYSTEM_PROMPT,
    COMPARISON_PROMPT_TEMPLATE,
    COMPARISON_SYSTEM_PROMPT,
    LLAMA_PROMPT_TEMPLATE,
)

model = "llama"


def generate_llama_choice_prompt_data(
    dataset, file_name="llama_choice_prompt_data.json", starting_idx=0
):
    responses, articles, keys = load_data(dataset)

    questions = []
    for key in keys[starting_idx:]:
        article = articles[key]
        self_response = responses[model][key]
        for other in ["human", "claude", "gpt4", "gpt35"]:
            question = {"key": key, "model": other}
            other_response = responses[other][key]

            question["forward_detection_prompt"] = LLAMA_PROMPT_TEMPLATE.format(
                system_prompt=DETECTION_SYSTEM_PROMPT,
                user_prompt=DETECTION_PROMPT_TEMPLATE.format(
                    article=article, summary1=self_response, summary2=other_response
                ),
            )
            question["backward_detection_prompt"] = LLAMA_PROMPT_TEMPLATE.format(
                system_prompt=DETECTION_SYSTEM_PROMPT,
                user_prompt=DETECTION_PROMPT_TEMPLATE.format(
                    article=article, summary1=other_response, summary2=self_response
                ),
            )

            question["forward_comparison_prompt"] = LLAMA_PROMPT_TEMPLATE.format(
                system_prompt=COMPARISON_SYSTEM_PROMPT,
                user_prompt=COMPARISON_PROMPT_TEMPLATE.format(
                    article=article, summary1=self_response, summary2=other_response
                ),
            )
            question["backward_comparison_prompt"] = LLAMA_PROMPT_TEMPLATE.format(
                system_prompt=COMPARISON_SYSTEM_PROMPT,
                user_prompt=COMPARISON_PROMPT_TEMPLATE.format(
                    article=article, summary1=other_response, summary2=self_response
                ),
            )

            questions.append(question)

    save_to_json(questions, file_name)
