from tqdm import tqdm
import matplotlib.pyplot as plt
from config import AGGS_STR_ORDERED, IDX2OP, Config


class EDA:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def print_stats(self, data):
        n = len(data)
        print("Computing tokens stats over {} examples...".format(n))
        data_stats = data.map(lambda x: map_to_length(x, self.tokenizer))
        output = data_stats.map(lambda x: compute_and_print_stats(x, n), batched=True, batch_size=-1)

    def eda(self, data):
        self.print_stats(data)
        plot_hists(data)


def map_to_length(x, tokenizer):
    x["input_len"] = len(tokenizer(x["input"]).input_ids)
    x["input_longer_256"] = int(x["input_len"] > 256)
    x["input_longer_128"] = int(x["input_len"] > 128)
    x["input_longer_64"] = int(x["input_len"] > 64)
    x["out_len"] = len(tokenizer(x["target"]).input_ids)
    x["out_longer_256"] = int(x["out_len"] > 256)
    x["out_longer_128"] = int(x["out_len"] > 128)
    x["out_longer_64"] = int(x["out_len"] > 64)
    return x


def compute_and_print_stats(x, len_data):
    if len(x["input_len"]) == len_data:
        print("\tINPUT/OUTPUT TOKEN DISTRIBUTION")
        print(
            "Input Mean: {:.2f}, %-Input > 256: {:.2f}%,  %-Input > 128: {:.2f}%, %-Input > 64: {:.2f}%\nOutput Mean: {:.2f}, %-Output > 256: {:.2f}%, %-Output > 128: {:.2f}%, %-Output > 64: {:.2f}%".format(
                sum(x["input_len"]) / len_data,
                100 * sum(x["input_longer_256"]) / len_data,
                100 * sum(x["input_longer_128"]) / len_data,
                100 * sum(x["input_longer_64"]) / len_data,
                sum(x["out_len"]) / len_data,
                100 * sum(x["out_longer_256"]) / len_data,
                100 * sum(x["out_longer_128"]) / len_data,
                100 * sum(x["out_longer_64"]) / len_data,
            )
        )


def plot_hists(data):
    print("Computing histogram stats...")
    lengths = []
    headers = []
    operators = []
    aggs = []
    no_conds = []

    for example in tqdm(data):
        sql = example["sql"]
        conds = sql["conds"]
        n_conds = len(conds["col"])
        no_conds.append(n_conds)
        aggs.append(AGGS_STR_ORDERED[sql["agg"]])

        for j in range(n_conds):
            operators.append(IDX2OP[conds["op"][j]])

        table = example["table"]
        lengths.append(len(table["rows"]))
        headers.append(len(table["header"]))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # Plot histogram of number of columns per table
    axes[0].hist(headers, bins=50, edgecolor='black')
    axes[0].set_title("Distribution of number of columns per table")
    axes[0].set_xlabel("Number of headers")
    axes[0].set_ylabel("Frequency")

    # Plot number of rows histogram
    axes[1].hist(lengths, bins=50, edgecolor='black')
    axes[1].set_title("Distribution of number of rows per table")
    axes[1].set_xlabel("Number of rows")
    axes[1].set_ylabel("Frequency")

    # Plot operators histogram
    axes[2].hist(operators, bins=50, edgecolor='black')
    axes[2].set_title("Distribution of operators")
    axes[2].set_xlabel("Operator")
    axes[2].set_ylabel("Frequency")

    # Plot number of conditions histogram
    axes[3].hist(no_conds, bins=50, edgecolor='black')
    axes[3].set_title("Distribution of number of conditions")
    axes[3].set_xlabel("Number of conditions")
    axes[3].set_ylabel("Frequency")

    # Plot aggs histogram
    axes[4].hist(aggs, bins=50, edgecolor='black')
    axes[4].set_title("Distribution of aggs")
    axes[4].set_xlabel("Agg")
    axes[4].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# if __name__ == '__main__':
#     from loader import load_dataset_from_json
#
#     # Loading Dataset
#     train_data, val_data, test_data = load_dataset_from_json()
#
#     # Load tokenizer and model
#     from model.utils import load_model_and_tokenizer
#     _, tokenizer, _ = load_model_and_tokenizer(Config.model_name)
#     print("special tokens map : ", tokenizer.special_tokens_map)
#
#     # Preprocess train data for EDA
#     from data.process import Processor
#     preprocessor = Processor(tokenizer, mode="human_readable_output", with_samples=False)
#     print("Preprocessing train data for EDA...")
#     train_data = train_data.map(lambda x: preprocessor.preprocess_data_row(x))
#
#     eda = EDA(tokenizer)
#     eda.eda(train_data)