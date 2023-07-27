from rouge_score import rouge_scorer
import json

def cal_rouge_score(str1, str2):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(str1, str2)
    return scores

def cal_score_all(results1, results2):
    scores_rouge1 = []
    scores_rougeL = []
    scores_EM = []
    scores_F1 = []
    for i in range(len(results1)):
        query1, output1 = results1[i]["prompt"], results1[i]["output"]
        query2, output2 = results2[i]["prompt"], results2[i]["output"]
        if query1 != query2:
            raise ValueError("query1 and query2 are different", query1, query2)
        output1, output2 = normalize_text(output1), normalize_text(output2)
        scores_EM.append( compute_exact_match(output1, output2))
        scores_F1.append(compute_f1(output1,output2))
        score_rouge = cal_rouge_score(output1, output2)
        scores_rouge1.append(score_rouge['rouge1'].fmeasure)
        scores_rougeL.append(score_rouge['rougeL'].fmeasure)
    N = float(len(results1))

    return sum(scores_EM)/N, sum(scores_F1)/N, sum(scores_rouge1)/N, sum(scores_rougeL)/N

def load_file(file_path):
    with open(file_path, "r") as json_file:
        loaded_data_list = json.load(json_file)
    return loaded_data_list

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)

if __name__ == "__main__":
    file1_list = [ "hf_13b_torch.float32.json", "hf_13b_torch.float16.json", "hf_13b_torch.bfloat16.json"]
    file2_list = [ "ft_13b_torch.float32.json", "ft_13b_torch.float16.json", "ft_13b_torch.bfloat16.json"]
    files_list = file1_list + file2_list
    for i in range(len(files_list)):
        for j in range(i, len(files_list)):
            file1, file2 = files_list[i], files_list[j]
            results1, results2 = load_file(file1), load_file(file2)
            scores = cal_score_all(results1, results2)
            print(f"{file1.ljust(20)} {file2.ljust(20)} EM: {scores[0]:.2f} F1: {scores[2]:.2f} rouge1: {scores[2]:.2f} rougeL: {scores[3]:.2f}")





