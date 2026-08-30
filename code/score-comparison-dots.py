import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path 
from scipy import stats

projectDir = Path(__file__).parent
dataDir    = projectDir / "data"
outputDir  = projectDir / "output"

interventionFile = Path(dataDir, "intervention.csv")
controlFile      = Path(dataDir, "control.csv")

COLORS = {
    "Intervention": "#0072B2",  # blue
    "Control":      "#5F6368",  # dark neutral gray
}

# Extract assessment scores from a given CSV file, validate them, and
# and return them as a DataFrame
#
def readScores(csvPath: Path, group: str) -> pd.DataFrame:
    data = pd.read_csv(csvPath)
    if "Score" not in data.columns:
        raise ValueError(f"{csvPath} has no 'Score' column.")

    scores = []

    for value in data["Score"]:
        parts = str(value).split("/")

        if len(parts) != 2 or parts[1].strip() != "18":
            raise ValueError(f"Unexpected score in {csvPath}: {value}")

        try:
            score = float(parts[0].strip())
        except ValueError:
            raise ValueError(f"Unexpected score in {csvPath}: {value}")

        if not 0 <= score <= 18:
            raise ValueError(f"Score outside the 0-18 range: {score}")

        scores.append(score)

    # Returns a DataFrame containing scores as rows; e.g.:
    #        Group  Score
    # Intervention   18.0
    # Intervention   18.0
    # Intervention   14.0
    return pd.DataFrame( {
        "Group": group,
        "Score": scores } )

# Return the mean and two-sided 95% one-sample t-CI.
#
def meanTConfidenceInterval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    n = values.size
    mean          = float(values.mean())
    standardError = float(stats.sem(values))
    critical      = float(stats.t.ppf(0.975, df = n-1))
    margin = critical * standardError
    return (mean,
            mean - margin,
            mean + margin)

# Return intervention-control mean difference and Welch 95% CI.
#
def welchDifferenceConfidenceInterval(
    intervention: np.ndarray,
    control:      np.ndarray,
) -> tuple[float, float, float]:
    intervention = np.asarray(intervention, dtype=float)
    control = np.asarray(control, dtype=float)
    
    interventionCount, controlCount = intervention.size, control.size

    interventionVariance = intervention.var(ddof=1)
    controlVariance      = control.var(ddof=1)

    difference = float(intervention.mean() - control.mean())
    
    standardError = float(
        np.sqrt( interventionVariance / interventionCount + controlVariance / controlCount ))
    
    degreesFreedom = (
        interventionVariance / interventionCount + controlVariance / controlCount
        ) ** 2 / (
        (interventionVariance / interventionCount) ** 2
        / (interventionCount - 1)
        + (controlVariance / controlCount) ** 2
        / (controlCount - 1) )
    
    critical = float(stats.t.ppf(0.975, df=degreesFreedom))
    margin = critical * standardError
    return (difference,
            difference - margin,
            difference + margin )

# Spread tied scores symmetrically without changing their y-values.
#
def tiedOffsets(
    values: np.ndarray,
    maximumWidth: float = 0.28,
) -> np.ndarray:
    values = np.asarray(values)
    offsets = np.zeros(values.size, dtype=float)

    for score in np.unique(values):
        indices = np.flatnonzero(values == score)
        if indices.size > 1:
            offsets[indices] = np.linspace(
                -maximumWidth / 2,
                maximumWidth / 2,
                indices.size )
    return offsets


def drawPanel(data: pd.DataFrame) -> None:
    groups = ["Control", "Intervention"]
    groupX = {"Control": 0.0, "Intervention": 1.0}

    plt.rcParams.update( {
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8 } )

    # Approximately half of a 6.5-inch text width.
    fig, ax = plt.subplots(figsize=(3.15, 3.05))
    fig.subplots_adjust(left=0.23, right=0.96, top=0.88, bottom=0.25)

    summaries: dict[str, tuple[float, float, float]] = {}

    for group in groups:
        scores = data.loc[data["Group"] == group, "Score"].to_numpy(dtype=float)
        x = groupX[group]

        # Shift participant points slightly left to leave room for mean/CI.
        pointX = x - 0.07 + tiedOffsets(scores)
        ax.scatter(
            pointX,
            scores,
            s=34,
            color=COLORS[group],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.94,
            zorder=3,
        )

        mean, lower, upper = meanTConfidenceInterval(scores)
        summaries[group] = (mean, lower, upper)
        summaryX = x + 0.21

        ax.errorbar(
            summaryX,
            mean,
            yerr=np.array([[mean - lower], [upper - mean]]),
            fmt="D",
            markersize=5.1,
            markerfacecolor="white",
            markeredgecolor=COLORS[group],
            markeredgewidth=1.3,
            ecolor=COLORS[group],
            elinewidth=1.5,
            capsize=3.5,
            capthick=1.2,
            zorder=4,
        )
        ax.annotate(
            f"{mean:.1f}",
            xy=(summaryX, mean),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=COLORS[group],
        )

    ax.set_title(
        "A   Total questionnaire scores",
        loc="left",
        fontweight="bold",
        pad=7,
    )
    ax.set_ylabel("Correct responses (0-18)")
    ax.set_xticks(
        [groupX[group] for group in groups],
        [
            f"{group}\n(n = {(data['Group'] == group).sum()})"
            for group in groups
        ],
    )
    ax.set_xlim(-0.42, 1.46)

    # The intervention t interval ends at 18.20. Show it rather than clipping
    # it at the questionnaire's maximum possible score of 18.
    ax.set_ylim(-0.25, 18.65)
    ax.set_yticks(np.arange(0, 19, 3))
    ax.grid(axis="y", color="#D9DDE1", linewidth=0.65)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#9AA0A6")
        ax.spines[spine].set_linewidth(0.7)
    ax.tick_params(axis="both", length=0, pad=4)

    fig.text(
        0.02,
        0.045,
        "Diamond and bar: mean and 95% CI",
        fontsize=6.8,
        color="#555555",
    )

    plt.show()
#     outputStem.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(
#         outputStem.with_suffix(".png"),
#         dpi=600,
#         facecolor="white",
#         bbox_inches="tight",
#         pad_inches=0.03,
#     )
#     fig.savefig(
#         outputStem.with_suffix(".svg"),
#         facecolor="white",
#         bbox_inches="tight",
#         pad_inches=0.03,
#     )
    plt.close(fig)

    for group in groups:
        scores = data.loc[data["Group"] == group, "Score"].to_numpy(dtype=float)
        mean, lower, upper = summaries[group]
        print(
            f"{group}: n={scores.size}, mean={mean:.3f}, "
            f"SD={scores.std(ddof=1):.3f}, "
            f"95% CI={lower:.3f} to {upper:.3f}"
        )

    control = data.loc[data["Group"] == "Control", "Score"].to_numpy(dtype=float)
    intervention = data.loc[
        data["Group"] == "Intervention", "Score"
    ].to_numpy(dtype=float)
    difference, lower, upper = welchDifferenceConfidenceInterval(
        intervention,
        control,
    )
    print(
        "Intervention - control mean difference: "
        f"{difference:.3f} (Welch 95% CI {lower:.3f} to {upper:.3f})"
    )


intervention = readScores(interventionFile, "Intervention")
control      = readScores(controlFile,      "Control")

combined = pd.concat([control, intervention], ignore_index=True)
drawPanel(combined)
