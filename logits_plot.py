import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import json
import numpy as np

def plot_text_with_highlighting(words, magnitudes, words_per_line=13,max_magnitude=0.05):
    """
    Plot a text with (logit) magnitudes.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Set up the text with highlighting and underlining
    for i, (word, magnitude) in enumerate(zip(words, magnitudes)):
        color = cm.Reds(magnitude * max_magnitude)  # Use the Greens colormap to map magnitude to red color
        # Add background highlighting
        rect = patches.Rectangle((i % words_per_line - 0.5, -0.5 - 1.2 * (i // words_per_line)), 1, 1, linewidth=0,
                                 edgecolor='none', facecolor=color)
        ax.add_patch(rect)

        # Add text
        ax.text(i % words_per_line, - 1.2 * (i // words_per_line), word, color='black', ha='center', va='center',
                fontsize=10)

    ax.set_xlim(-0.5, words_per_line - 0.5)
    ax.set_ylim(- 1.2 * (len(words) // words_per_line) - 0.5, 0.5)
    ax.axis('off')  # Turn off the axis


def wrap_text(text, width=100):
    words = text.split()
    result = []
    current_line = []

    for word in words:
        if len(' '.join(current_line + [word])) <= width:
            current_line.append(word)
        else:
            result.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        result.append(' '.join(current_line))
    return '\n'.join(result)


if __name__ == "__main__":
    with open("results/output_data_qwen_new_detailed.json", "r", encoding="utf-8") as f:
        outputs = json.load(f)
    offset = 0  # If results are split across multiple files, this increases the question index when saving
    word_limit = 300  # np.inf

    for question_idx, output in enumerate(outputs):
        for type, summary in output["metrics"]["approach_2"].items():
            if len(summary["masked_out_infilling"]) > 0:
                # Collect KL per word
                words = []
                kls = []
                for answer_idx, answer in enumerate(summary["masked_out_infilling"]["detailed_logging"]["answers"]):
                    if answer_idx >= word_limit:
                        break
                    words.append(answer["masked_words"]["masked_word"])
                    kls.append(answer["masked_words"]["KL_div_sum"])

                # Plot + add title and subtitle
                plot_text_with_highlighting(words, kls)
                ax = plt.gca()
                plt.text(y=1.07, x=0.5, s=wrap_text(f"{type} summary: {output['summaries'][type]}"), transform=ax.transAxes, ha="center", va="center")
                plt.title(f"More red = Logits predicted given the summary have high KL divergence from logits predicted given samples of the distribution.\nSummed KL divergence: {np.array(kls).sum():.2f}", color="darkred", y=0.05, transform=ax.transAxes, ha="center", va="top")

                # Save
                plt.tight_layout()
                plt.savefig(f"plots/question_{question_idx + offset}_summary_{type}.png")
                plt.close()
