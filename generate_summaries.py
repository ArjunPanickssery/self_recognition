from data import load_data, save_to_json
from models import get_summary

xsum_responses, xsum_articles, xsum_keys = load_data("xsum")
cnn_responses, cnn_articles, cnn_keys = load_data("cnn")

main_models = ["gpt4", "gpt35", "llama"]
xsum_models_gpt35 = [
    "xsum_2_ft_gpt35",
    "xsum_10_ft_gpt35",
    "xsum_500_ft_gpt35",
    "xsum_always_1_ft_gpt35",
    "xsum_random_ft_gpt35",
    "xsum_readability_ft_gpt35",
    "xsum_length_ft_gpt35",
    "xsum_vowelcount_ft_gpt35",
]
cnn_models_gpt35 = [
    "cnn_2_ft_gpt35",
    "cnn_10_ft_gpt35",
    "cnn_500_ft_gpt35",
    "cnn_always_1_ft_gpt35",
    "cnn_random_ft_gpt35",
    "cnn_readability_ft_gpt35",
    "cnn_length_ft_gpt35",
    "cnn_vowelcount_ft_gpt35",
]

xsum_models_llama = [
    "xsum_2_ft_llama",
    "xsum_10_ft_llama",
    "xsum_500_ft_llama",
    "xsum_always_1_ft_llama",
    "xsum_random_ft_llama",
    "xsum_readability_ft_llama",
    "xsum_length_ft_llama",
    "xsum_vowelcount_ft_llama",
]
cnn_models_llama = [
    "cnn_2_ft_llama",
    "cnn_10_ft_llama",
    "cnn_500_ft_llama",
    "cnn_always_1_ft_llama",
    "cnn_random_ft_llama",
    "cnn_readability_ft_llama",
    "cnn_length_ft_llama",
    "cnn_vowelcount_ft_llama",
]

models = (
    main_models
    + xsum_models_gpt35
    + cnn_models_gpt35
    + xsum_models_llama
    + cnn_models_llama
)

print("Starting...")
for model in models:
    if model in main_models:
        continue
    if "llama" in model:
        continue
    results = {}
    for key in xsum_keys[:50]:
        results[key] = get_summary(xsum_articles[key], "xsum", model)
        save_to_json(results, f"summaries/xsum/{model}_responses.json")

    results = {}
    for key in cnn_keys[:50]:
        results[key] = get_summary(cnn_articles[key], "cnn", model)
        save_to_json(results, f"summaries/cnn/{model}_responses.json")
    print(model, "done!")

print("Done!")
