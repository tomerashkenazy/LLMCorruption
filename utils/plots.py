"""Plotting utilities for cross-model analysis."""

from typing import Dict, List

import numpy as np


def plot_cross_model_entropy_matrix(
    cross_model_matrix: np.ndarray,
    model_list: List[str],
    output_prefix: str = "cross_model_entropy_matrix",
) -> None:
    """Save a cross-model entropy heatmap (PNG/PDF/SVG)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    model_names_short = [m.split("/")[-1][:15] for m in model_list]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cross_model_matrix,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        xticklabels=model_names_short,
        yticklabels=model_names_short,
        cbar_kws={"label": "Entropy %"},
        vmin=0,
        vmax=100,
        ax=ax,
        annot_kws={"size": 9},
    )

    ax.set_xlabel("Target Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Source Model", fontsize=12, fontweight="bold")
    ax.set_title("Cross-Model Entropy Transfer Matrix", fontsize=14, fontweight="bold")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    for i in range(len(model_list)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="blue", lw=3))

    plt.tight_layout()

    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{output_prefix}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_comprehensive_results(
    cross_model_matrix: np.ndarray,
    cross_model_classifications: Dict[str, Dict],
    model_list: List[str],
    output_prefix: str = "comprehensive_results",
) -> None:
    """Generate the 3x3 comprehensive figure from the notebook."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    model_names_short = [m.split("/")[-1][:15] for m in model_list]

    corruption_matrix = np.zeros((len(model_list), len(model_list)))
    for source_idx, source_model in enumerate(model_list):
        for target_idx, target_model in enumerate(model_list):
            key = f"{source_model}→{target_model}"
            if key in cross_model_classifications:
                is_corrupted = cross_model_classifications[key]["is_corrupted"]
                corruption_matrix[source_idx, target_idx] = 1 if is_corrupted else 0

    corruption_type_counts = {}
    for data in cross_model_classifications.values():
        ct = data["corruption_type"]
        corruption_type_counts[ct] = corruption_type_counts.get(ct, 0) + 1

    corrupted_per_target = np.sum(corruption_matrix, axis=0)
    corrupted_per_source = np.sum(corruption_matrix, axis=1)

    diagonal_mask = np.eye(len(model_list), dtype=bool)
    self_corruption = corruption_matrix[diagonal_mask]
    transfer_corruption = corruption_matrix[~diagonal_mask]

    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.40)

    ax1 = fig.add_subplot(gs[0:2, 0:2])
    im = ax1.imshow(cross_model_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    for i in range(len(model_list)):
        for j in range(len(model_list)):
            entropy = cross_model_matrix[i, j]
            is_corrupted = corruption_matrix[i, j] == 1
            text_color = "white" if entropy > 50 else "black"
            text = f"{entropy:.1f}%"
            if is_corrupted:
                text += "\n[X]"
            ax1.text(j, i, text, ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    ax1.set_xticks(np.arange(len(model_list)))
    ax1.set_yticks(np.arange(len(model_list)))
    ax1.set_xticklabels(model_names_short, rotation=45, ha="right", fontsize=10)
    ax1.set_yticklabels(model_names_short, fontsize=10)
    ax1.set_xlabel("Target Model", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Source Model", fontsize=12, fontweight="bold")
    ax1.set_title("Cross-Model Entropy Matrix\n([X] = Corrupted Output)", fontsize=14, fontweight="bold", pad=15)

    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Entropy %", rotation=270, labelpad=20, fontsize=11, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 2])
    labels = list(corruption_type_counts.keys())
    sizes = list(corruption_type_counts.values())
    colors_pie = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#2ecc71", "#95a5a6", "#e67e22"]
    explode = [0.05 if label != "NORMAL" else 0 for label in labels]

    wedges, texts, autotexts = ax2.pie(
        sizes,
        labels=[l.replace("_OUTPUT", "").replace("_", " ")[:12] for l in labels],
        autopct="%1.1f%%",
        colors=colors_pie[: len(labels)],
        explode=explode,
        startangle=90,
        textprops={"fontsize": 9, "weight": "bold"},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(10)
    ax2.set_title("Corruption Type\nDistribution", fontsize=12, fontweight="bold", pad=10)

    ax3 = fig.add_subplot(gs[1, 2])
    vulnerability_order = np.argsort(corrupted_per_target)
    sorted_models = [model_names_short[i] for i in vulnerability_order]
    sorted_counts = corrupted_per_target[vulnerability_order]

    y_pos = np.arange(len(sorted_models))
    colors_vuln = ["#e74c3c" if c >= 4 else "#f39c12" if c >= 2 else "#2ecc71" for c in sorted_counts]
    bars = ax3.barh(y_pos, sorted_counts, color=colors_vuln, edgecolor="black", linewidth=1)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(sorted_models, fontsize=9)
    ax3.set_xlabel("Corrupted by N prompts", fontsize=10, fontweight="bold")
    ax3.set_title("Model Vulnerability\n(0-7 scale)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlim(0, 7.5)
    ax3.grid(axis="x", alpha=0.3, linestyle="--")

    for bar, count in zip(bars, sorted_counts):
        width = bar.get_width()
        ax3.text(width + 0.15, bar.get_y() + bar.get_height() / 2, f"{int(count)}/7", ha="left", va="center", fontsize=9, fontweight="bold")

    ax4 = fig.add_subplot(gs[2, 0])
    transferability_order = np.argsort(corrupted_per_source)
    sorted_models_trans = [model_names_short[i] for i in transferability_order]
    sorted_counts_trans = corrupted_per_source[transferability_order]

    y_pos_trans = np.arange(len(sorted_models_trans))
    colors_trans = ["#e74c3c" if c >= 4 else "#f39c12" if c >= 2 else "#2ecc71" for c in sorted_counts_trans]
    bars_trans = ax4.barh(y_pos_trans, sorted_counts_trans, color=colors_trans, edgecolor="black", linewidth=1)
    ax4.set_yticks(y_pos_trans)
    ax4.set_yticklabels(sorted_models_trans, fontsize=9)
    ax4.set_xlabel("Corrupts N models", fontsize=10, fontweight="bold")
    ax4.set_title("Prompt Transferability\n(0-7 scale)", fontsize=12, fontweight="bold", pad=10)
    ax4.set_xlim(0, 7.5)
    ax4.grid(axis="x", alpha=0.3, linestyle="--")

    for bar, count in zip(bars_trans, sorted_counts_trans):
        width = bar.get_width()
        ax4.text(width + 0.15, bar.get_y() + bar.get_height() / 2, f"{int(count)}/7", ha="left", va="center", fontsize=9, fontweight="bold")

    ax5 = fig.add_subplot(gs[2, 1])
    self_corrupted = int(np.sum(self_corruption))
    self_normal = len(self_corruption) - self_corrupted
    transfer_corrupted = int(np.sum(transfer_corruption))
    transfer_normal = len(transfer_corruption) - transfer_corrupted

    categories = ["Self-Optimization\n(7 tests)", "Cross-Model Transfer\n(42 tests)"]
    corrupted_counts = [self_corrupted, transfer_corrupted]
    normal_counts = [self_normal, transfer_normal]

    x_pos = np.arange(len(categories))
    width = 0.35

    bars1 = ax5.bar(x_pos - width / 2, corrupted_counts, width, label="Corrupted", color="#e74c3c", edgecolor="black", linewidth=1.5)
    bars2 = ax5.bar(x_pos + width / 2, normal_counts, width, label="Normal", color="#2ecc71", edgecolor="black", linewidth=1.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{int(height)}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax5.set_ylabel("Count", fontsize=10, fontweight="bold")
    ax5.set_title("Self vs Transfer Corruption", fontsize=12, fontweight="bold", pad=10)
    ax5.legend(fontsize=10, loc="upper right")
    ax5.grid(axis="y", alpha=0.3, linestyle="--")

    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis("off")

    total_tests = len(cross_model_classifications)
    total_corrupted = int(np.sum(corruption_matrix))
    corruption_rate = total_corrupted / total_tests * 100 if total_tests > 0 else 0

    self_corruption_rate = self_corrupted / len(self_corruption) * 100 if len(self_corruption) > 0 else 0
    transfer_corruption_rate = transfer_corrupted / len(transfer_corruption) * 100 if len(transfer_corruption) > 0 else 0

    most_vulnerable = model_names_short[np.argmax(corrupted_per_target)]
    most_transferable = model_names_short[np.argmax(corrupted_per_source)]

    avg_entropy_self = np.mean(np.diag(cross_model_matrix))
    avg_entropy_transfer = np.mean(cross_model_matrix[~diagonal_mask])

    stats_text = f"""
    === KEY STATISTICS ===
    {'-'*35}

    OVERALL RESULTS:
    > Total Tests: {total_tests} (7x7 matrix)
    > Corrupted: {total_corrupted} ({corruption_rate:.1f}%)
    > Normal: {total_tests - total_corrupted} ({100 - corruption_rate:.1f}%)

    SELF vs TRANSFER:
    > Self-Optimization: {self_corrupted}/7 ({self_corruption_rate:.1f}%)
    > Cross-Model: {transfer_corrupted}/42 ({transfer_corruption_rate:.1f}%)

    AVERAGE ENTROPY:
    > Self (diagonal): {avg_entropy_self:.1f}%
    > Transfer (off-diag): {avg_entropy_transfer:.1f}%

    TOP PERFORMERS:
    > Most Vulnerable: {most_vulnerable}
      (corrupted by {int(np.max(corrupted_per_target))}/7)

    > Most Transferable: {most_transferable}
      (corrupts {int(np.max(corrupted_per_source))}/7)

    INSIGHTS:
    > Self-optimization {'MORE' if self_corruption_rate > transfer_corruption_rate else 'LESS'}
      effective than transfer
    > Entropy {'HIGHER' if avg_entropy_self > avg_entropy_transfer else 'LOWER'} for self
    """

    ax6.text(
        0.05,
        0.95,
        stats_text,
        transform=ax6.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    fig.suptitle(
        "LLM Corruption Attack: Comprehensive Results\nGCG Optimization | Cross-Model Transferability | Corruption Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{output_prefix}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
