import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path

projectDir = Path(__file__).parent
dataDir    = projectDir / "data"
outputDir  = projectDir / "output"

interventionFile = Path(dataDir, "intervention.csv")
controlFile      = Path(dataDir, "control.csv")

colors = {
    "Intervention": "blue",
    "Control":      "black" }

objectiveLabels = [
    "LO1 Risk factors",
    "LO2 Prevention",
    "LO3 Warning signs",
    "LO4 Peer response",
    "LO5 Heat stroke",
    "LO6 Emergency response" ]

correctAnswers = [
    "A sunny, asphalt court with high humidity and little airflow",
    "During hot-weather activity, risk is generally greater early in a season because students may have had little time to adjust to the heat.",
    "None of the activities can be ranked.",
    "Gradually increase the duration and intensity of activity over several days.",
    "Drink fluids regularly throughout the day, including before and during the activity.",
    "Reduce the intensity or duration of the activity, add rest and cooling breaks, and adjust clothing or equipment when permitted.",
    "Stop participating and report the cramp to a responsible adult.",
    "Stop participating and report the symptoms to a responsible adult.",
    "Beginning to stumble and visibly slow down",
    "Encourage the student to stop participating, notify a responsible adult immediately, and stay with the student until the adult takes over.",
    "Encourage the student to remain out of the activity and involve a responsible adult until qualified medical personnel say it is safe to return.",
    "Refuse to hide the symptom and help the student report it to a responsible adult.",
    "A student develops a vacant stare and cannot follow a familiar instruction.",
    "This may be heat stroke and should be treated as a medical emergency.",
    "Collapse and confusion are sufficient warning signs to suspect possible heat stroke immediately, even without a thermometer reading.",
    "Alert a responsible adult, activate the emergency plan, ensure that 911 is called, and begin rapid cooling according to the plan.",
    "Move the affected student to a shaded area and apply ice packs on the affected student’s neck, armpits, and groin.",
    "Do not give fluids; alert a responsible adult and stay with the student."
]

# Parse and validate the total scores reported by Google Forms.
#
def readReportedScores(scoreValues: pd.Series, csvPath: Path) -> np.ndarray:
    scores = []

    for value in scoreValues:
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

    return np.asarray(scores, dtype=float)


# Score all 18 responses, confirm the answer key reproduces each participant's
# reported total, and return percent correct for the six learning objectives.
#
def readObjectiveResults(csvPath: Path, group: str) -> pd.DataFrame:
    data = pd.read_csv(csvPath)

    if "Score" not in data.columns:
        raise ValueError(f"{csvPath} has no 'Score' column.")

    questionColumns = list(data.columns[3:])
    if len(questionColumns) != len(correctAnswers):
        raise ValueError(
            f"Expected {len(correctAnswers)} questionnaire items in {csvPath}; "
            f"found {len(questionColumns)}.")

    responses = data.loc[:, questionColumns]
    if responses.isna().to_numpy().any():
        raise ValueError(f"Missing questionnaire responses in {csvPath}.")

    answerKey      = pd.Series(correctAnswers, index=questionColumns)
    correctMatrix  = responses.eq(answerKey, axis="columns")
    computedScores = correctMatrix.sum(axis="columns").to_numpy(dtype=float)
    reportedScores = readReportedScores(data["Score"], csvPath)

    if not np.array_equal(computedScores, reportedScores):
        mismatches = np.flatnonzero(computedScores != reportedScores) + 2
        raise ValueError(
            f"Answer-key totals do not match the reported scores in {csvPath} "
            f"at CSV row(s) {mismatches.tolist()}." )

    records = []
    questionsPerObjective = 3

    for objectiveIndex, objectiveLabel in enumerate(objectiveLabels):
        firstQuestion = objectiveIndex * questionsPerObjective
        lastQuestion  = firstQuestion + questionsPerObjective
        objectiveCorrect = correctMatrix.iloc[:, firstQuestion:lastQuestion]
        percentCorrect = 100 * float(objectiveCorrect.to_numpy().mean())

        records.append( {
            "Group":          group,
            "Objective":      objectiveLabel,
            "PercentCorrect": percentCorrect } )

    return pd.DataFrame(records)


def drawPanel(data: pd.DataFrame) -> None:
    plt.rcParams.update( {
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8 } )

    # Approximately half of a 6.5-inch text width.
#     fig, ax = plt.subplots(figsize=(3.15, 3.05))
    fig, ax = plt.subplots(figsize=(4, 3.05))
    controlData = data.loc[
        (data["Group"] == "Control") & (data["Objective"].isin(objectiveLabels))
    ].sort_values("Objective")
    control = controlData["PercentCorrect"].to_numpy(dtype=float)

    interventionData = data.loc[
        (data["Group"] == "Intervention") & (data["Objective"].isin(objectiveLabels))
    ].sort_values("Objective")
    intervention = interventionData["PercentCorrect"].to_numpy(dtype=float)
    
    # np.arange(6)[::-1] -> [0, 1, 2, 3, 4, 5][::-1] -> [5, 4, 3, 2, 1, 0]
    objectiveY = np.arange(len(objectiveLabels))[::-1]

    for objectiveIndex, y in enumerate(objectiveY):
        ax.plot(
            [control[objectiveIndex], intervention[objectiveIndex]],
            [y, y],
            color="gray",
            linewidth=1.2,
            zorder=1 )

    ax.scatter(
        control,
        objectiveY,
        s=38,
        color=colors["Control"],
        edgecolor="white",
        linewidth=0.6,
        label="Control",
        zorder=2
    )
    ax.scatter(
        intervention,
        objectiveY,
        s=38,
        color=colors["Intervention"],
        edgecolor="white",
        linewidth=0.6,
        label="Intervention",
        zorder=2
    )

    for objectiveIndex, y in enumerate(objectiveY):
        ax.annotate(
            f"{control[objectiveIndex]:.0f}",
            xy=(control[objectiveIndex], y),
            xytext=(0, -4),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color=colors["Control"]
        )
        ax.annotate(
            f"{intervention[objectiveIndex]:.0f}",
            xy=(intervention[objectiveIndex], y),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=colors["Intervention"]
        )

    ax.set_xlabel("Correct responses (%)")
    ax.set_xlim(0, 103)
    ax.set_xticks(np.arange(0, 120, 25))

    ax.set_yticks(objectiveY, objectiveLabels)
    ax.set_ylim(-0.45, len(objectiveLabels) - 0.55)
    ax.grid(axis="x", color="#D9DDE1", linewidth=0.65)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper left",
        handletextpad=0.4)

    fig.tight_layout(pad=0.2)
    plt.show()

    Path(outputDir).mkdir(parents=True, exist_ok=True)
    fig.savefig(
        outputDir / "score-comparison-objectives.png",
        bbox_inches="tight",
        pad_inches=0.05 )

    summary = data.pivot(
        index="Objective",
        columns="Group",
        values="PercentCorrect"
    ).loc[objectiveLabels]
    summary["Difference"] = summary["Intervention"] - summary["Control"]
    print(summary.round(1).to_string())


intervention = readObjectiveResults(interventionFile, "Intervention")
control      = readObjectiveResults(controlFile,      "Control")

combined = pd.concat([control, intervention], ignore_index=True)
drawPanel(combined)
