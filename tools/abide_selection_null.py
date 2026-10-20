"""ABIDE I で、設定を試して選ぶところまで含めた並べ替え検定（ノートブック 04 の 6 章と補足で触れたもの）。

ノートブック 04 と同じ 871 人・17 施設で、施設を丸ごと除く交差検証の正解率を、次の 14 通りの設定で計算する。

    前処理の版 2 通り（filt_noglobal、nofilt_noglobal）×（線形 SVM の C 4 通り＋リッジ回帰の強さ 3 通り）

施設の中で診断のラベルを混ぜるたびに 14 通りすべてを計算し直し、その最大を「混ぜたときの値」とする。
本物の値には、ノートブック 04 の 5 章の設定（filt_noglobal、C = 0.001）で、施設を丸ごと除く交差検証の正解率を使う。

リッジ回帰は、scikit-learn の RidgeClassifier(alpha=...) と同じ計算（切片は正則化しない）を、内積の表を使って速く行う。
始めに、RidgeClassifier そのものと予測が一致することを確かめて表示する。

データ（PCP の CC200 の時系列。2 つの版で合わせて約 680 MB）は、無ければ <data_dir> の下に取ってくる。
data/ の下に置く（.gitignore で Git の管理の外にしてある）。

使い方：
    python tools/abide_selection_null.py <data_dir> [並べ替えの回数（既定 100）]
    例：python tools/abide_selection_null.py data/abide-selection 100

手元（Python 3.13.14、numpy 2.1.3、pandas 2.2.3、scikit-learn 1.9.1）で、取得を除いて約 10 分かかった。
100 回の結果：混ぜたときの最大は平均 0.540・標準偏差 0.006・最大 0.566、本物 0.689、p = 0.0099（100 回で出せる最小）。
"""
import socket
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC

BASE = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
STRATEGIES = ("filt_noglobal", "nofilt_noglobal")
SVM_C = (1.0, 0.01, 0.001, 0.0001)
RIDGE_ALPHA = (1e2, 1e3, 1e4)
SEED = 2

socket.setdefaulttimeout(60)  # 取得の途中で通信が 60 秒止まったら、やり直す


def download(url, path):
    """1 つのファイルを取ってくる。すでに取ってあれば取り直さない。通信が切れたら 3 回まで試す"""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")  # 途中で切れても壊れたファイルが残らないように
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, part)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2)
        part.rename(path)
    return path


def load_people(data_dir):
    """参加者の表を取ってきて、ノートブック 04 と同じ品質確認の条件で絞る"""
    phenotypes = pd.read_csv(download(f"{BASE}/Phenotypic_V1_0b_preprocessed1.csv",
                                      data_dir / "Phenotypic_V1_0b_preprocessed1.csv"))
    passed = (
        (phenotypes["FILE_ID"] != "no_filename")
        & (phenotypes["qc_rater_1"] == "OK")
        & phenotypes["qc_anat_rater_2"].isin(["OK", "maybe"])
        & phenotypes["qc_func_rater_2"].isin(["OK", "maybe"])
        & (phenotypes["qc_anat_rater_3"] == "OK")
        & (phenotypes["qc_func_rater_3"] == "OK")
    )
    return phenotypes[passed].reset_index(drop=True)


def load_features(data_dir, strategy, file_ids):
    """1 つの前処理の版について、全員分の 19,900 個の結合の強さ（Fisher の z）の表"""
    jobs = [(f"{BASE}/Outputs/cpac/{strategy}/rois_cc200/{f}_rois_cc200.1D",
             data_dir / strategy / "rois_cc200" / f"{f}_rois_cc200.1D") for f in file_ids]
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda job: download(*job), jobs))
    upper = np.triu_indices(200, k=1)
    rows = []
    for path in paths:
        with np.errstate(invalid="ignore", divide="ignore"):  # 信号がずっと 0 の領域があっても止めない
            matrix = np.corrcoef(np.loadtxt(path).T)
        rows.append(np.arctanh(np.nan_to_num(matrix[upper])))  # 計算できない結合（NaN）は 0 にする
    return np.vstack(rows)


def prepare_ridge(K, splits):
    """リッジ回帰のために、学習に使う人の平均で中心化した内積の表を、交差検証の組ごとに固有値分解しておく"""
    folds = []
    for train, test in splits:
        K_train, K_test = K[np.ix_(train, train)], K[np.ix_(test, train)]
        column_mean = K_train.mean(axis=0)
        overall_mean = column_mean.mean()
        K_train_c = K_train - column_mean[None, :] - column_mean[:, None] + overall_mean
        K_test_c = K_test - K_test.mean(axis=1, keepdims=True) - column_mean[None, :] + overall_mean
        eigenvalues, eigenvectors = np.linalg.eigh(K_train_c)
        folds.append((train, test, eigenvalues, eigenvectors, K_test_c @ eigenvectors))
    return folds


def ridge_predict(folds, labels, alpha):
    """RidgeClassifier(alpha) と同じ予測（ラベルは +1 か −1）"""
    predicted = np.empty(len(labels))
    for train, test, eigenvalues, eigenvectors, K_test_v in folds:
        label_mean = labels[train].mean()
        coef = (eigenvectors.T @ (labels[train] - label_mean)) / (eigenvalues + alpha)
        predicted[test] = np.where(K_test_v @ coef + label_mean > 0, 1, -1)
    return predicted


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python tools/abide_selection_null.py <data_dir> [n_permutations (default 100)]")
    data_dir = Path(sys.argv[1])
    n_permutations = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    people = load_people(data_dir)
    y = np.where(people["DX_GROUP"] == 1, 1, -1)  # ASD が +1、定型が −1
    site = people["SITE_ID"].str.replace(r"_[12]$", "", regex=True).to_numpy()  # 同じ大学の 2 組をまとめた 17 施設
    print(len(y), "people,", len(np.unique(site)), "sites", flush=True)

    X = {strategy: load_features(data_dir, strategy, people["FILE_ID"]) for strategy in STRATEGIES}
    K = {strategy: X[strategy] @ X[strategy].T for strategy in STRATEGIES}
    splits = list(LeaveOneGroupOut().split(np.zeros(len(y)), y, site))
    ridge_folds = {strategy: prepare_ridge(K[strategy], splits) for strategy in STRATEGIES}

    for strategy in STRATEGIES:
        for alpha in RIDGE_ALPHA:
            reference = np.empty(len(y))
            for train, test in splits:
                reference[test] = RidgeClassifier(alpha=alpha).fit(X[strategy][train], y[train]).predict(X[strategy][test])
            mine = ridge_predict(ridge_folds[strategy], y.astype(float), alpha)
            print(f"check {strategy} ridge alpha={alpha:g}: predictions different from RidgeClassifier: "
                  f"{int((reference != mine).sum())}", flush=True)

    def accuracies(labels):
        out = {}
        for strategy in STRATEGIES:
            for C in SVM_C:
                predicted = np.empty(len(labels))
                for train, test in splits:
                    svm = SVC(kernel="precomputed", C=C).fit(K[strategy][np.ix_(train, train)], labels[train])
                    predicted[test] = svm.predict(K[strategy][np.ix_(test, train)])
                out[f"{strategy} svm C={C:g}"] = (predicted == labels).mean()
            for alpha in RIDGE_ALPHA:
                out[f"{strategy} ridge alpha={alpha:g}"] = (ridge_predict(ridge_folds[strategy], labels.astype(float), alpha)
                                                            == labels).mean()
        return out

    observed = accuracies(y)
    for name, accuracy in observed.items():
        print(f"observed {name}: {accuracy:.3f}")
    score = observed["filt_noglobal svm C=0.001"]

    rng = np.random.default_rng(SEED)
    start = time.time()
    null_max = []
    for i in range(n_permutations):
        shuffled = y.copy()
        for s in np.unique(site):
            rows = np.where(site == s)[0]
            shuffled[rows] = y[rng.permutation(rows)]  # 施設の中で混ぜる
        null_max.append(max(accuracies(shuffled).values()))
        if (i + 1) % 10 == 0:
            print(f"{i + 1} permutations, {time.time() - start:.0f}s, running max {max(null_max):.3f}", flush=True)
    null_max = np.array(null_max)
    p_value = (np.sum(null_max >= score) + 1) / (n_permutations + 1)
    print(f"maximum over {len(observed)} settings with shuffled labels: mean {null_max.mean():.3f}, "
          f"sd {null_max.std():.3f}, max {null_max.max():.3f}")
    print(f"observed (filt_noglobal, C=0.001) {score:.3f}; best observed {max(observed.values()):.3f}; p = {p_value:.4f}")


if __name__ == "__main__":
    main()
