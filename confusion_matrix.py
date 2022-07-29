import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


def df_confusion_matrix(df):
    # df["label_class_eq"] = df["label"] == df["classified"]
    df["cm_value"] = df.apply(lambda d: "tw" if d["label"] == d["classified"] == "w" else
    "tn" if d["label"] == d["classified"] == "n" else
    "fn" if d["label"] == "w" and d["classified"] == "n" else
    "fw", axis=1)

    print(df["label"].value_counts())

    cm_values = df["cm_value"].value_counts()
    print(cm_values)

    tw = cm_values['tw'] if "tw" in cm_values.keys().tolist() else 0
    tn = cm_values['tn'] if "tn" in cm_values.keys().tolist() else 0
    fw = cm_values['fw'] if "fw" in cm_values.keys().tolist() else 0
    fn = cm_values['fn'] if "fn" in cm_values.keys().tolist() else 0
    w = df['label'].value_counts()['w'] if "w" in df['label'].value_counts().keys().tolist() else 0
    n = df['label'].value_counts()['n'] if "n" in df['label'].value_counts().keys().tolist() else 0

    show_stats(w, n, tw, tn)
    show_matrix(tw, fw, fn, tn)


def show_matrix(tw, fw, fn, tn):
    cm = pd.DataFrame([[tw, fw], [fn, tn]], index=["class as W", "class as N"], columns=["W label", "N label"])
    sn.heatmap(cm, cmap="Blues", annot=True)
    plt.show()


def show_percentage_matrix(tw, fw, fn, tn):
    sum = tw+fw+fn+tn
    cm = pd.DataFrame([[tw/sum, fw/sum], [fn/sum, tn/sum]], index=["class as W", "class as N"], columns=["W label", "N label"])
    sn.heatmap(cm, cmap="Blues", fmt='.2%', annot=True)
    plt.show()


def show_stats(w, n, tw, tn):
    print(f"w: {w}")
    print(f"n: {n}")
    print(f"correct W: {round(tw / w, 2)}")
    print(f"correct N: {round(tn / n, 2)}")
