"""
Extractive Text Summarization
==============================
Scores each sentence using:
  1. TF-IDF importance
  2. Position bias (earlier sentences score higher)
  3. Query relevance (cosine similarity to query vector)

Selection strategies:
  - Greedy TF-IDF ranking
  - MMR (Maximal Marginal Relevance): argmax[λ*sim(s,query) - (1-λ)*max_{j∈selected}sim(s,j)]

Evaluation: ROUGE-1, ROUGE-2, ROUGE-L
"""

import os
import json
import math
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    tokens = re.findall(r"[ぁ-ん]+|[ァ-ン]+|[一-龥]+|[a-z]+", text)
    return tokens


def split_sentences(text: str) -> List[str]:
    """Split Japanese/English text into sentences."""
    # Split on。? ! . with optional whitespace
    parts = re.split(r"(?<=[。？！\.\?!])\s*", text.strip())
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

class TFIDF:
    def __init__(self):
        self.idf: Dict[str, float] = {}
        self.vocab: List[str] = []

    def fit(self, documents: List[List[str]]):
        N = len(documents)
        df: Counter = Counter()
        for doc in documents:
            for term in set(doc):
                df[term] += 1
        self.vocab = list(df.keys())
        self.idf = {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in self.vocab}

    def transform(self, document: List[str]) -> np.ndarray:
        tf = Counter(document)
        total = max(len(document), 1)
        vec = np.zeros(len(self.vocab), dtype=np.float64)
        for i, term in enumerate(self.vocab):
            vec[i] = (tf[term] / total) * self.idf.get(term, 0.0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-10 else vec

    def fit_transform(self, documents: List[List[str]]) -> np.ndarray:
        self.fit(documents)
        return np.stack([self.transform(doc) for doc in documents])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Sentence scoring
# ---------------------------------------------------------------------------

def score_sentences_tfidf(sentences: List[str],
                           query: Optional[str] = None,
                           position_weight: float = 0.3,
                           query_weight: float = 0.3) -> Tuple[np.ndarray, TFIDF, np.ndarray]:
    """
    Returns scores (N,), fitted TFIDF, sentence vectors (N, V).
    """
    tokenized = [tokenize(s) for s in sentences]
    tfidf = TFIDF()
    vecs = tfidf.fit_transform(tokenized)  # (N, V)

    # TF-IDF importance: sum of TF-IDF values per sentence
    tfidf_scores = vecs.sum(axis=1)  # (N,)
    max_t = tfidf_scores.max() if tfidf_scores.max() > 0 else 1.0
    tfidf_scores = tfidf_scores / max_t

    # Position bias: exponential decay
    n = len(sentences)
    pos_scores = np.exp(-np.arange(n) * 3.0 / n)
    pos_scores = pos_scores / pos_scores.max()

    # Query relevance
    if query is not None:
        q_tokens = tokenize(query)
        q_vec = tfidf.transform(q_tokens)
        query_scores = np.array([cosine_similarity(v, q_vec) for v in vecs])
    else:
        query_scores = np.zeros(n)

    # Combine
    total_weight = 1.0 - position_weight - query_weight
    scores = (total_weight * tfidf_scores
              + position_weight * pos_scores
              + query_weight * query_scores)
    return scores, tfidf, vecs


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def select_greedy(scores: np.ndarray, k: int) -> List[int]:
    """Simple top-k by score, preserving original order."""
    ranked = np.argsort(scores)[::-1][:k]
    return sorted(ranked.tolist())


def select_mmr(vecs: np.ndarray, scores: np.ndarray, k: int,
               lambda_mmr: float = 0.7) -> List[int]:
    """
    Maximal Marginal Relevance:
      At each step: argmax_i [ λ * score(i) - (1-λ) * max_{j∈selected} sim(i,j) ]
    """
    n = len(scores)
    selected: List[int] = []
    remaining = list(range(n))

    for _ in range(k):
        if not remaining:
            break
        best_idx = -1
        best_val = -np.inf
        for i in remaining:
            if selected:
                max_sim = max(cosine_similarity(vecs[i], vecs[j]) for j in selected)
            else:
                max_sim = 0.0
            mmr_val = lambda_mmr * scores[i] - (1 - lambda_mmr) * max_sim
            if mmr_val > best_val:
                best_val = mmr_val
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)

    return sorted(selected)


def select_position(sentences: List[str], k: int) -> List[int]:
    """Baseline: always pick first k sentences."""
    return list(range(min(k, len(sentences))))


# ---------------------------------------------------------------------------
# ROUGE metrics
# ---------------------------------------------------------------------------

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def rouge_n(hypothesis: str, reference: str, n: int) -> Dict[str, float]:
    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)
    hyp_ng = _ngrams(hyp_tokens, n)
    ref_ng = _ngrams(ref_tokens, n)
    overlap = sum((hyp_ng & ref_ng).values())
    precision = overlap / max(sum(hyp_ng.values()), 1)
    recall = overlap / max(sum(ref_ng.values()), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-optimized LCS
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    return prev[n]


def rouge_l(hypothesis: str, reference: str) -> Dict[str, float]:
    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / max(len(hyp_tokens), 1)
    recall = lcs / max(len(ref_tokens), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_summary(hypothesis: str, reference: str) -> Dict[str, Dict[str, float]]:
    return {
        "ROUGE-1": rouge_n(hypothesis, reference, 1),
        "ROUGE-2": rouge_n(hypothesis, reference, 2),
        "ROUGE-L": rouge_l(hypothesis, reference),
    }


# ---------------------------------------------------------------------------
# Synthetic Japanese dataset: 20 articles + reference summaries
# ---------------------------------------------------------------------------

ARTICLES = [
    {
        "title": "気候変動と農業への影響",
        "text": (
            "地球温暖化により、世界各地で農業生産量が変化している。"
            "特に熱帯地域では気温上昇による干ばつが深刻化し、食料不足が懸念される。"
            "一方で高緯度地域では耕作可能面積が拡大する地域もある。"
            "農業の適応策として、耐熱性品種の開発や灌漑技術の向上が進んでいる。"
            "国際機関は各国に対し、温室効果ガスの削減と同時に適応策の策定を求めている。"
            "農家の所得支援も重要な政策課題として浮上している。"
            "食料安全保障を確保するため、技術革新と国際協力が不可欠である。"
        ),
        "reference": (
            "気候変動は農業生産に深刻な影響を与えており、耐熱性品種開発や灌漑技術向上などの適応策と国際協力が求められている。"
        ),
        "query": "農業への気候変動の影響と対策"
    },
    {
        "title": "人工知能と医療診断",
        "text": (
            "人工知能を活用した医療診断システムが急速に発展している。"
            "特に画像認識技術は放射線科や病理診断において高い精度を示している。"
            "AIは膨大なデータから微細な異常を検出する能力で人間の医師を補助する。"
            "一方でAIの判断根拠が不透明であるという説明可能性の問題も指摘される。"
            "患者のプライバシー保護やデータセキュリティも重要な課題だ。"
            "医師とAIの協働モデルが最も効果的であると多くの研究が示している。"
            "規制当局はAI医療機器の承認基準を整備し始めている。"
        ),
        "reference": (
            "AI医療診断は画像認識で高精度を示す一方、説明可能性やプライバシーの課題もあり、医師との協働モデルが有効とされる。"
        ),
        "query": "AIの医療診断における役割"
    },
    {
        "title": "再生可能エネルギーの普及",
        "text": (
            "太陽光発電と風力発電のコストはここ10年で大幅に低下した。"
            "多くの国がカーボンニュートラルを目標に再生可能エネルギー導入を加速している。"
            "蓄電池技術の進歩により、電力の安定供給が可能になりつつある。"
            "再生可能エネルギーへの転換は雇用創出にも寄与している。"
            "送電網のインフラ整備が普及加速の鍵を握る。"
            "化石燃料からの転換は既存産業への影響も考慮が必要だ。"
            "国際エネルギー機関は2050年までの脱炭素化シナリオを示している。"
        ),
        "reference": (
            "再生可能エネルギーはコスト低下と蓄電技術進歩で普及が加速しており、送電インフラ整備と産業転換への対応が課題となっている。"
        ),
        "query": "再生可能エネルギーの普及と課題"
    },
    {
        "title": "少子高齢化と社会保障",
        "text": (
            "日本の少子高齢化は世界で最も進んだ水準にある。"
            "高齢者人口の増加は医療費・介護費の膨張をもたらしている。"
            "現役世代の負担増加が経済成長の足かせとなっている。"
            "外国人労働者の受け入れ拡大が労働力不足の解消策として検討されている。"
            "デジタル技術を活用した介護の効率化が注目されている。"
            "年金制度の持続可能性を確保するための改革が急務とされる。"
            "子育て支援策の充実が出生率回復への鍵とされている。"
        ),
        "reference": (
            "日本の少子高齢化は医療・介護費増大と現役世代負担増をもたらし、年金改革や子育て支援、外国人労働者受け入れが課題である。"
        ),
        "query": "少子高齢化対策"
    },
    {
        "title": "サイバーセキュリティの脅威",
        "text": (
            "デジタル化の進展とともにサイバー攻撃の件数と規模が増大している。"
            "ランサムウェアによる被害は医療機関や重要インフラにも及んでいる。"
            "国家ぐるみのサイバー攻撃が国際的な安全保障問題となっている。"
            "企業のセキュリティ投資不足が脆弱性を生む主要因とされる。"
            "ゼロトラストセキュリティモデルへの移行が推奨されている。"
            "セキュリティ人材の不足は世界的な問題となっている。"
            "国際的な法執行機関の連携による摘発事例も増えている。"
        ),
        "reference": (
            "サイバー攻撃は規模・複雑性ともに増大しており、企業のセキュリティ強化とゼロトラスト導入、専門人材育成が急務である。"
        ),
        "query": "サイバーセキュリティの課題と対策"
    },
]

# Extend to 20 articles with additional entries
EXTRA_ARTICLES = [
    {
        "title": "量子コンピュータの現状",
        "text": (
            "量子コンピュータは従来のコンピュータでは不可能な計算を実現しうる技術だ。"
            "GoogleやIBMなど大手企業が量子優位性の実証に取り組んでいる。"
            "現時点ではノイズの多い中規模量子デバイスの時代にある。"
            "量子アルゴリズムの実用化には誤り訂正技術の成熟が必要だ。"
            "創薬や材料科学への応用が最も有望視されている。"
            "量子暗号は従来の暗号を無効化する可能性があり安全保障上の懸念もある。"
            "日本政府も量子技術を重点投資分野として位置付けている。"
        ),
        "reference": "量子コンピュータは誤り訂正の課題が残るが創薬・材料科学への応用が期待され、各国が重点投資している。",
        "query": "量子コンピュータの応用と課題"
    },
    {
        "title": "宇宙開発の民間参入",
        "text": (
            "SpaceXやBlue Originなど民間企業が宇宙開発を牽引する時代になった。"
            "打ち上げコストの劇的な低下が宇宙ビジネスの裾野を広げている。"
            "衛星インターネットサービスは地球上の通信格差解消に貢献する可能性がある。"
            "月や火星への有人探査計画が現実味を帯びてきた。"
            "宇宙デブリ問題は持続可能な宇宙利用の大きな障害だ。"
            "国際宇宙法の整備が新たな宇宙時代の課題となっている。"
            "宇宙資源の採掘をめぐる法的枠組みの議論も始まっている。"
        ),
        "reference": "民間参入でコスト低下が進む宇宙開発は衛星通信や有人探査の可能性を広げる一方、デブリ問題と法整備が課題だ。",
        "query": "民間宇宙開発の現状"
    },
    {
        "title": "フードテックと代替タンパク質",
        "text": (
            "人口増加と食料不足への対応策として代替タンパク質が注目されている。"
            "植物由来の代替肉は味や食感で急速に改善が進んでいる。"
            "培養肉は動物を屠殺せずに肉を生産できる革新的技術だ。"
            "昆虫食はタンパク質効率が高く環境負荷も低い。"
            "消費者の受容性と価格競争力が普及の鍵を握る。"
            "食品安全規制の整備が各国で進められている。"
            "伝統的畜産業との共存モデルも模索されている。"
        ),
        "reference": "代替タンパク質は植物由来・培養肉・昆虫食などが開発されており、消費者受容性と規制整備が普及の課題となっている。",
        "query": "代替タンパク質技術"
    },
    {
        "title": "メンタルヘルスとデジタル支援",
        "text": (
            "コロナ禍以降、メンタルヘルス問題への関心が世界的に高まっている。"
            "スマートフォンアプリによる認知行動療法が手軽な支援ツールとして普及している。"
            "AIチャットボットが初期相談の受け皿として機能し始めている。"
            "デジタル支援は専門家へのアクセス障壁を下げる効果がある。"
            "一方でデジタルツールの効果と安全性についての科学的検証が求められる。"
            "対面カウンセリングとデジタル支援の組み合わせが最善とされる。"
            "職場でのメンタルヘルス対策も重要な経営課題となっている。"
        ),
        "reference": "デジタルツールはメンタルヘルス支援のアクセス改善に貢献するが、効果検証と対面ケアとの組み合わせが重要だ。",
        "query": "デジタルメンタルヘルス支援"
    },
    {
        "title": "電気自動車の普及と課題",
        "text": (
            "電気自動車の販売台数は世界的に急増しており主要自動車市場で存在感を増している。"
            "バッテリーコストの低下が電気自動車の価格競争力を高めている。"
            "充電インフラの整備不足が普及の最大の障壁のひとつだ。"
            "電力源が再生可能エネルギーでない限り完全な脱炭素化にはならない。"
            "レアメタルの供給チェーンと採掘環境問題が課題として残る。"
            "自動運転技術との統合で移動の概念が変わりつつある。"
            "内燃機関車からの転換に伴う雇用移行支援が必要だ。"
        ),
        "reference": "電気自動車はコスト低下で普及が進む一方、充電インフラ不足、レアメタル問題、電力源の脱炭素化が課題である。",
        "query": "電気自動車普及の現状と課題"
    },
    {
        "title": "教育のデジタル化",
        "text": (
            "コロナ禍はオンライン教育の急速な普及を促した。"
            "個別最適化学習はAIを活用して各生徒の理解度に合わせた指導を実現する。"
            "デジタル格差が教育機会の不平等を生む懸念がある。"
            "対面授業の社会的・情動的価値はデジタルでは代替しにくい。"
            "教師のデジタルリテラシー向上が不可欠な課題となっている。"
            "ゲーミフィケーションが学習意欲の向上に有効であることが示されている。"
            "オープン教育リソースの普及が教育の民主化に貢献している。"
        ),
        "reference": "教育デジタル化はAI個別学習やゲーミフィケーションで効果を上げる一方、デジタル格差と教師研修が課題である。",
        "query": "教育のデジタル化"
    },
    {
        "title": "都市の持続可能性",
        "text": (
            "スマートシティ構想が世界各地で進められ都市インフラのデジタル化が加速している。"
            "センサーとデータ分析により交通渋滞や廃棄物管理の効率化が図られている。"
            "緑の都市計画は熱島現象の緩和と住民の心身健康に寄与する。"
            "15分都市の概念は生活に必要な機能を徒歩15分圏内に集約するものだ。"
            "都市への人口集中と地方の過疎化は同時並行して進んでいる。"
            "スマートシティのデータ管理と市民プライバシーの保護が課題だ。"
            "気候変動への適応として洪水対策や熱波対策の整備が急がれる。"
        ),
        "reference": "スマートシティは交通や廃棄物管理の効率化を進めるが、データプライバシーと気候変動への都市適応が重要課題だ。",
        "query": "スマートシティの取り組み"
    },
    {
        "title": "バイオテクノロジーと医療革命",
        "text": (
            "mRNAワクチン技術はコロナ禍で初めて実用化され医療の歴史を変えた。"
            "CRISPR-Cas9ゲノム編集技術は遺伝性疾患の根本治療を可能にしつつある。"
            "細胞療法や遺伝子治療は従来薬では治療不可能だった疾患に希望をもたらす。"
            "バイオ医薬品のコストは依然として高く普及の障壁となっている。"
            "ゲノム編集の倫理的課題、特に生殖細胞への応用は国際的議論を呼んでいる。"
            "パーソナライズド医療は個人のゲノム情報に基づく最適治療を目指す。"
            "バイオセキュリティの観点から研究規制の強化も求められている。"
        ),
        "reference": "mRNAとCRISPRなどのバイオ技術は医療を革新するが、コスト、倫理課題、セキュリティ規制への対応が必要だ。",
        "query": "バイオテクノロジーの医療応用"
    },
    {
        "title": "デジタル通貨と金融の未来",
        "text": (
            "中央銀行デジタル通貨(CBDC)の研究・実証実験が世界中で進んでいる。"
            "ビットコインなど暗号資産は投機資産としての性格が強い。"
            "分散型金融(DeFi)は中間業者を排除した金融サービスを提供する。"
            "CBDCは金融包摂の促進と決済の効率化に貢献しうる。"
            "マネーロンダリング対策と匿名性のバランスが規制上の難題だ。"
            "ステーブルコインの規制をめぐって各国で議論が続いている。"
            "デジタル通貨の普及は既存銀行ビジネスモデルを変容させる。"
        ),
        "reference": "CBDCや暗号資産、DeFiが金融を変革しつつあるが、マネーロンダリング対策と規制整備、既存業界への影響が課題だ。",
        "query": "デジタル通貨と金融規制"
    },
    {
        "title": "プラスチック汚染と循環経済",
        "text": (
            "毎年800万トン以上のプラスチックが海洋に流入していると推定されている。"
            "マイクロプラスチックは食物連鎖を通じて人体への蓄積が懸念されている。"
            "循環経済モデルは廃棄物を出さない生産・消費サイクルを目指す。"
            "生分解性プラスチックや代替素材の開発が活発になっている。"
            "企業の拡大生産者責任(EPR)制度の導入が各国で進んでいる。"
            "消費者意識の変化とリサイクルインフラの整備が鍵となる。"
            "国際条約によるプラスチック汚染防止の取り組みが始まっている。"
        ),
        "reference": "海洋プラスチック汚染は深刻で、循環経済への移行、代替素材開発、EPR制度と国際条約による包括的対策が求められている。",
        "query": "プラスチック汚染対策"
    },
    {
        "title": "遠隔医療の可能性",
        "text": (
            "テレメディシンはオンラインで医師の診察を受けられるサービスだ。"
            "地方や離島など医療アクセスが困難な地域での活用が期待される。"
            "コロナ禍で遠隔医療の利用が急増し規制緩和も進んだ。"
            "ウェアラブルデバイスとの連携で継続的な健康モニタリングが可能になる。"
            "対面診察が必要な場合との判断が医師にとって重要な課題だ。"
            "診療報酬制度の整備が遠隔医療普及の前提条件となっている。"
            "個人健康データの管理と国際的な情報共有のルール作りも必要だ。"
        ),
        "reference": "遠隔医療は医療アクセス改善に有効で普及が進むが、対面との使い分け判断、報酬制度整備、データ管理が課題だ。",
        "query": "遠隔医療の普及と課題"
    },
    {
        "title": "自動化と雇用の未来",
        "text": (
            "ロボットとAIによる自動化は多くの職種を代替する可能性がある。"
            "定型的・反復的な業務ほど自動化の影響を受けやすい。"
            "一方で新技術は新たな職種と雇用機会を生み出す歴史的経緯がある。"
            "リスキリング(学び直し)への支援が労働者保護の重要策となっている。"
            "ユニバーサルベーシックインカムの議論が自動化と関連して再燃している。"
            "自動化の恩恵を社会全体に分配する税制・社会保障の再設計が必要だ。"
            "人間にしかできない創造性や共感力が価値を持つ時代になりつつある。"
        ),
        "reference": "自動化は雇用を代替する一方で新職種を生み、リスキリング支援と自動化の恩恵を分配する社会制度の再設計が課題だ。",
        "query": "自動化と雇用変化"
    },
    {
        "title": "水資源の危機",
        "text": (
            "世界人口の約20億人が水ストレスの高い地域に居住している。"
            "農業が世界の淡水使用量の約70%を占めている。"
            "気候変動は降水パターンを変化させ水資源の偏在を悪化させる。"
            "海水淡水化技術はコスト低下が進んでいるが依然エネルギー集約的だ。"
            "農業の滴下灌漑など節水技術の普及が急務となっている。"
            "越境水資源をめぐる国際紛争リスクが高まっている。"
            "都市の水道インフラ老朽化による漏水損失も大きな問題だ。"
        ),
        "reference": "水資源危機は農業用水の効率化、海水淡水化の推進、国際的な水紛争防止と老朽インフラ更新で対応が求められる。",
        "query": "水資源不足の対策"
    },
    {
        "title": "メタバースの可能性と限界",
        "text": (
            "メタバースは仮想空間上での社会活動・経済活動を可能にする概念だ。"
            "VRヘッドセットの価格低下と性能向上が普及の条件を整えつつある。"
            "ゲームや教育、医療訓練など特定用途での有用性が実証されている。"
            "アバターを通じた匿名性はハラスメントや詐欺のリスクも生む。"
            "大規模なメタバース構築には膨大なエネルギーと計算資源が必要だ。"
            "デジタル資産の権利保護と所有権の法的整備が課題として残る。"
            "現実空間との融合により新しい社会インフラとなる可能性もある。"
        ),
        "reference": "メタバースは教育・医療訓練などで有用性を示すが、ハラスメントリスク、エネルギー消費、デジタル資産の法整備が課題だ。",
        "query": "メタバースの活用と課題"
    },
    {
        "title": "核融合エネルギーの展望",
        "text": (
            "核融合発電は太陽と同じ原理で無限に近いクリーンエネルギーを提供しうる。"
            "2022年にNIFが初めて投入エネルギーを上回る核融合エネルギーを得た。"
            "ITER国際プロジェクトは2035年の実験炉稼動を目指している。"
            "民間スタートアップが核融合炉の商業化を競い合っている。"
            "核融合は放射性廃棄物が少なく安全性が高いとされる。"
            "発電コストの見通しと実用化時期については依然不確実性が高い。"
            "核融合実現は気候変動問題のゲームチェンジャーになりうると期待される。"
        ),
        "reference": "核融合エネルギーはNIF成果と民間投資で現実味が増すが、発電コストと実用化時期には依然不確実性が残る。",
        "query": "核融合エネルギーの現状"
    },
]

ALL_ARTICLES = ARTICLES + EXTRA_ARTICLES


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(k: int = 2, lambda_mmr: float = 0.7) -> Dict:
    results = []
    method_scores = {"tfidf": [], "mmr": [], "position": []}

    for article in ALL_ARTICLES:
        sentences = split_sentences(article["text"])
        reference = article["reference"]
        query = article.get("query", "")

        if len(sentences) < 2:
            continue

        k_actual = min(k, len(sentences))

        # Score sentences
        scores, tfidf, vecs = score_sentences_tfidf(sentences, query=query)

        # Method 1: Greedy TF-IDF
        idx_tfidf = select_greedy(scores, k_actual)
        summary_tfidf = "".join([sentences[i] for i in idx_tfidf])

        # Method 2: MMR
        idx_mmr = select_mmr(vecs, scores, k_actual, lambda_mmr=lambda_mmr)
        summary_mmr = "".join([sentences[i] for i in idx_mmr])

        # Method 3: Position
        idx_pos = select_position(sentences, k_actual)
        summary_pos = "".join([sentences[i] for i in idx_pos])

        # Evaluate
        ev_tfidf = evaluate_summary(summary_tfidf, reference)
        ev_mmr = evaluate_summary(summary_mmr, reference)
        ev_pos = evaluate_summary(summary_pos, reference)

        for metric in ["ROUGE-1", "ROUGE-2", "ROUGE-L"]:
            method_scores["tfidf"].append(ev_tfidf[metric]["f1"])
            method_scores["mmr"].append(ev_mmr[metric]["f1"])
            method_scores["position"].append(ev_pos[metric]["f1"])

        results.append({
            "title": article["title"],
            "n_sentences": len(sentences),
            "selected_k": k_actual,
            "tfidf": {
                "selected_indices": idx_tfidf,
                "summary": summary_tfidf,
                "scores": ev_tfidf
            },
            "mmr": {
                "selected_indices": idx_mmr,
                "summary": summary_mmr,
                "scores": ev_mmr
            },
            "position": {
                "selected_indices": idx_pos,
                "summary": summary_pos,
                "scores": ev_pos
            }
        })

    # Aggregate
    agg = {}
    for method in ["tfidf", "mmr", "position"]:
        scores_arr = method_scores[method]
        agg[method] = {
            "mean_f1": float(np.mean(scores_arr)),
            "std_f1": float(np.std(scores_arr)),
        }

    return {"results": results, "aggregate": agg, "k": k, "lambda_mmr": lambda_mmr}


def print_report(eval_output: Dict):
    print("\n" + "=" * 60)
    print("Extractive Summarization Evaluation Report")
    print("=" * 60)
    print(f"Top-k = {eval_output['k']}, MMR λ = {eval_output['lambda_mmr']}")
    print(f"Articles evaluated: {len(eval_output['results'])}")
    print("\nAggregate ROUGE F1 (averaged over all articles & metrics):")
    for method, vals in eval_output["aggregate"].items():
        print(f"  {method:10s}: mean={vals['mean_f1']:.4f}  std={vals['std_f1']:.4f}")

    print("\nSample article summaries:")
    for res in eval_output["results"][:3]:
        print(f"\n  [{res['title']}]")
        print(f"  TF-IDF:   {res['tfidf']['summary'][:80]}...")
        print(f"  MMR:      {res['mmr']['summary'][:80]}...")
        print(f"  Position: {res['position']['summary'][:80]}...")
        print(f"  ROUGE-1 F1: tfidf={res['tfidf']['scores']['ROUGE-1']['f1']:.3f} "
              f"mmr={res['mmr']['scores']['ROUGE-1']['f1']:.3f} "
              f"pos={res['position']['scores']['ROUGE-1']['f1']:.3f}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "experiments", "06-summarization")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "results.json")

    print("Running extractive summarization evaluation...")
    output = run_evaluation(k=2, lambda_mmr=0.7)

    print_report(output)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to: {save_path}")
