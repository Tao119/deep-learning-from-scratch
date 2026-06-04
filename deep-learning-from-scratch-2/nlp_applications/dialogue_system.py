"""
Retrieval-Based Dialogue System
================================
Architecture:
  1. Knowledge base: 100+ Q&A pairs across medical, science, sports topics
  2. TF-IDF query encoding → cosine similarity → top-k retrieval
  3. Context-aware re-ranking: last 3 turns boost similar topics
  4. Intent classification: greeting / question / farewell / unknown
  5. Entity extraction: disease names, drug names, medical terms

Demo: 10-turn conversation about medical topics
"""

import re
import math
from collections import Counter, deque
from typing import List, Dict, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Knowledge Base (100 Q&A pairs)
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: List[Dict[str, str]] = [
    # Medical — diseases
    {"q": "高血圧とは何ですか？", "a": "高血圧は血圧が正常値（収縮期130mmHg未満、拡張期80mmHg未満）を超えた状態です。動脈硬化・心臓病・脳卒中のリスク因子となります。", "topic": "medical"},
    {"q": "高血圧の治療法は？", "a": "生活習慣改善（減塩・運動・禁煙）が基本で、必要に応じてACE阻害薬、ARB、カルシウム拮抗薬、利尿薬などの降圧薬を使用します。", "topic": "medical"},
    {"q": "糖尿病とはどんな病気ですか？", "a": "糖尿病はインスリン分泌不足または作用不全により慢性的に血糖値が高くなる代謝疾患です。1型・2型・妊娠糖尿病などに分類されます。", "topic": "medical"},
    {"q": "糖尿病の合併症は？", "a": "三大合併症として糖尿病網膜症、糖尿病腎症、糖尿病神経障害があります。また心血管疾患のリスクも大幅に上昇します。", "topic": "medical"},
    {"q": "心筋梗塞の症状は？", "a": "胸痛（圧迫感・締め付け感）、左腕・顎・背中への放散痛、冷汗、呼吸困難が典型的症状です。女性は非典型的症状が多いことがあります。", "topic": "medical"},
    {"q": "脳卒中の種類は？", "a": "脳卒中は脳梗塞（血管閉塞）と脳出血（血管破裂）に大別されます。脳梗塞はさらに心原性、アテローム血栓性、ラクナ梗塞などに分類されます。", "topic": "medical"},
    {"q": "がんの早期発見方法は？", "a": "定期的な検診（胃カメラ、大腸内視鏡、マンモグラフィ、PSAなど）と腫瘍マーカー検査が有効です。異変を感じたら早めに受診することが重要です。", "topic": "medical"},
    {"q": "肺炎の治療は？", "a": "細菌性肺炎には抗生物質（アモキシシリン、レボフロキサシンなど）、ウイルス性肺炎には対症療法が中心です。重症例では入院・酸素療法が必要です。", "topic": "medical"},
    {"q": "喘息の管理方法は？", "a": "吸入ステロイドが長期管理の基本薬です。発作時には短時間作用型β2刺激薬（SABA）を使用します。誘因回避も重要です。", "topic": "medical"},
    {"q": "COPD（慢性閉塞性肺疾患）とは？", "a": "主に喫煙による肺の慢性炎症で、気流制限が不可逆的に進行する疾患です。息切れ、慢性咳・痰が主症状です。", "topic": "medical"},
    # Medical — drugs
    {"q": "アスピリンの副作用は？", "a": "主な副作用は消化管出血・潰瘍、出血時間延長、アスピリン喘息（非ステロイド性抗炎症薬誘発喘息）などです。", "topic": "medical"},
    {"q": "スタチンとは何ですか？", "a": "HMG-CoA還元酵素阻害薬（スタチン）はLDLコレステロールを低下させ、心血管疾患リスクを減少させる薬剤です。アトルバスタチン、ロスバスタチンが代表的です。", "topic": "medical"},
    {"q": "ワルファリンの使用上の注意は？", "a": "ビタミンKを多く含む食品（納豆・緑黄色野菜）との相互作用に注意が必要です。定期的なINRモニタリングが必要で、出血に注意します。", "topic": "medical"},
    {"q": "メトホルミンはどんな薬ですか？", "a": "2型糖尿病の第一選択薬で、肝臓での糖新生を抑制します。体重増加が少なく、心血管保護効果も報告されています。", "topic": "medical"},
    {"q": "ACE阻害薬の効果は？", "a": "ACE阻害薬はアンジオテンシンIIの産生を阻害し、血圧を低下させます。心不全・糖尿病性腎症の保護効果もあります。", "topic": "medical"},
    {"q": "抗生物質の適切な使用方法は？", "a": "処方された期間は必ず飲み切ることが重要です。自己判断での中止は耐性菌を生む原因となります。ウイルス感染には効果がありません。", "topic": "medical"},
    {"q": "痛み止めの種類は？", "a": "NSAIDs（ロキソプロフェン、イブプロフェン）、アセトアミノフェン、オピオイド系鎮痛薬があります。目的・症状に応じて使い分けます。", "topic": "medical"},
    {"q": "降圧薬の種類を教えてください", "a": "ACE阻害薬、ARB（アンジオテンシンII受容体拮抗薬）、カルシウム拮抗薬、利尿薬、β遮断薬が主要な降圧薬です。", "topic": "medical"},
    {"q": "インスリン療法はどんな場合に必要ですか？", "a": "1型糖尿病では必須です。2型糖尿病でも経口薬で血糖コントロールが不十分な場合、手術・感染時、妊娠中などに使用します。", "topic": "medical"},
    {"q": "コレステロールの基準値は？", "a": "LDLコレステロール140mg/dL未満、HDLコレステロール40mg/dL以上、中性脂肪150mg/dL未満が一般的な目安です。", "topic": "medical"},
    # Medical — symptoms & examination
    {"q": "発熱の定義は何度ですか？", "a": "一般的に37.5℃以上を発熱、38.5℃以上を高熱と分類します。腋窩温（脇の下）で測定するのが一般的です。", "topic": "medical"},
    {"q": "血液検査でわかることは？", "a": "白血球数（感染・炎症）、赤血球・ヘモグロビン（貧血）、血小板（出血傾向）、肝機能（AST/ALT）、腎機能（クレアチニン）などが評価できます。", "topic": "medical"},
    {"q": "MRIとCTの違いは？", "a": "MRIは放射線を使わず軟部組織の評価に優れ、CTは撮影が速く骨・出血・肺の評価に向いています。MRIは禁忌（金属インプラントなど）への注意が必要です。", "topic": "medical"},
    {"q": "貧血の原因は？", "a": "鉄欠乏性貧血（最多）、ビタミンB12欠乏（悪性貧血）、葉酸欠乏、慢性疾患性貧血、溶血性貧血、再生不良性貧血などがあります。", "topic": "medical"},
    {"q": "甲状腺機能亢進症の症状は？", "a": "動悸、体重減少、発汗増加、手の振戦、下痢、月経不順などが典型的症状です。眼球突出（バセドウ病に特徴的）も見られることがあります。", "topic": "medical"},
    # Science
    {"q": "光合成とは何ですか？", "a": "光合成は植物が光エネルギーを利用してCO₂と水から有機物（グルコース）と酸素を生成するプロセスです。葉緑体のチラコイドとストロマで行われます。", "topic": "science"},
    {"q": "DNAとRNAの違いは？", "a": "DNAはデオキシリボース含有、二本鎖、チミン塩基を持ち遺伝情報を保存します。RNAはリボース含有、一本鎖（主に）、ウラシル塩基を持ちタンパク質合成を担います。", "topic": "science"},
    {"q": "相対性理論の基本とは？", "a": "特殊相対性理論は光速不変の原理と相対性原理を基礎に、時間の遅れ・長さの収縮・E=mc²を導きます。一般相対性理論は重力を時空の曲率として記述します。", "topic": "science"},
    {"q": "ブラックホールとは？", "a": "ブラックホールは重力が非常に強く光でも脱出できない天体です。恒星の重力崩壊や超大質量ブラックホールとして銀河中心に存在します。", "topic": "science"},
    {"q": "量子力学の不確定性原理とは？", "a": "ハイゼンベルクの不確定性原理は位置と運動量を同時に精密に決定できないことを主張します。Δx・Δp ≥ ℏ/2 という式で表されます。", "topic": "science"},
    {"q": "気候変動の原因は？", "a": "産業革命以降の化石燃料燃焼によるCO₂・CH₄などの温室効果ガス増加が主因です。これにより地球の平均気温が上昇する温室効果が強化されています。", "topic": "science"},
    {"q": "AIの機械学習とは？", "a": "機械学習はデータからパターンを自動的に学習する手法です。教師あり学習、教師なし学習、強化学習に大別され、ニューラルネットワークが近年の主流です。", "topic": "science"},
    {"q": "ワクチンの仕組みは？", "a": "ワクチンは弱毒化・不活化した病原体や抗原タンパク、mRNAなどを投与し免疫記憶を誘導します。再感染時に速やかな免疫応答が起きるよう準備します。", "topic": "science"},
    {"q": "CRISPR-Cas9とは何ですか？", "a": "CRISPR-Cas9はRNA誘導型のゲノム編集ツールで、特定のDNA配列を精密に切断・修復・挿入できます。遺伝病治療や農業育種への応用が進んでいます。", "topic": "science"},
    {"q": "ニュートンの運動法則は？", "a": "第1法則：慣性の法則。第2法則：F=ma（力=質量×加速度）。第3法則：作用反作用の法則。3つの法則が古典力学の基礎をなします。", "topic": "science"},
    {"q": "宇宙の年齢は？", "a": "現在の観測から宇宙の年齢は約138億年と推定されています。ビッグバン理論に基づき、宇宙マイクロ波背景放射の観測から精密に求められています。", "topic": "science"},
    {"q": "光の速さは？", "a": "真空中の光速は約299,792,458 m/s（約30万km/s）で、物理定数として厳密に定義されています。", "topic": "science"},
    {"q": "周期表の元素数は？", "a": "2024年時点で118種類の元素が確認されています。自然界に安定して存在する元素は92種類（ウランまで）です。", "topic": "science"},
    {"q": "地震のマグニチュードとは？", "a": "マグニチュードは地震が放出するエネルギーの対数的尺度です。1増えるとエネルギーは約32倍になります。震度は各地点での揺れの強さを表します。", "topic": "science"},
    {"q": "プレートテクトニクスとは？", "a": "地球の岩石圏が十数枚のプレートに分かれ、それらの移動・衝突・沈み込みによって地震・火山・造山運動が起きるという理論です。", "topic": "science"},
    # Sports
    {"q": "マラソンの距離は？", "a": "マラソンの公式距離は42.195km（42km195m）です。1908年ロンドンオリンピックで現在の距離が定着しました。", "topic": "sports"},
    {"q": "サッカーのオフサイドルールは？", "a": "攻撃側プレーヤーがボールを受ける瞬間に、相手ゴールラインとボールの間に相手プレーヤーが1人（GKを除く）しかいない場合にオフサイドとなります。", "topic": "sports"},
    {"q": "野球の打率とは？", "a": "打率は安打数÷打数で計算されます。三振・四球・犠打などは打数に含まれません。一般的に打率.300以上が優秀な打者の目安とされます。", "topic": "sports"},
    {"q": "テニスのグランドスラムとは？", "a": "全豪オープン、全仏オープン（ローランギャロス）、ウィンブルドン、全米オープンの4大メジャー大会の総称です。1年に全制覇を「年間グランドスラム」と呼びます。", "topic": "sports"},
    {"q": "水泳の主な種目は？", "a": "自由形（クロール）、背泳ぎ、平泳ぎ、バタフライの4泳法があります。個人メドレーはバタフライ→背泳ぎ→平泳ぎ→自由形の順で泳ぎます。", "topic": "sports"},
    {"q": "バスケットボールの3ポイントラインは何メートル？", "a": "NBAでは約7.24m（23フィート9インチ）、FIBAルールでは6.75mです。日本のBリーグはFIBAルールを採用しています。", "topic": "sports"},
    {"q": "ラグビーのトライとは？", "a": "トライはボールをインゴールに接地させることで得られる5点です。その後のコンバージョンキック成功でさらに2点追加されます。", "topic": "sports"},
    {"q": "オリンピックはどのくらいの頻度で開催される？", "a": "夏季・冬季オリンピックはそれぞれ4年ごとに開催されます。2年おきに夏季・冬季が交互に行われます。", "topic": "sports"},
    {"q": "柔道の段位制度は？", "a": "柔道の段位は1段から10段まであります。黒帯は初段から5段、6段以上は紅白帯・紅帯を使用します。段位は講道館が認定します。", "topic": "sports"},
    {"q": "ゴルフのパーとは？", "a": "パーはそのホールを完了するための基準打数です。通常コースはパー72（18ホール）で構成されます。バーディーはパーより1打少なく、ボギーは1打多い状態です。", "topic": "sports"},
    {"q": "卓球の試合形式は？", "a": "国際大会では11点先取の7ゲームマッチが標準です。サービスは2本交代で行われます。台は274cm×152.5cm、ネット高さ15.25cmです。", "topic": "sports"},
    {"q": "陸上の短距離記録は？", "a": "100mの世界記録はウサイン・ボルト（ジャマイカ）の9秒58（2009年）。200mは19秒19（同）。日本記録は100mで山縣亮太の9秒95です。", "topic": "sports"},
    {"q": "バレーボールのセット数は？", "a": "3セット先取が勝利です（5セットマッチ）。第5セット（ファイナルセット）は15点先取でデュースあり。1〜4セットは25点先取です。", "topic": "sports"},
    {"q": "スキーのアルペン種目は？", "a": "滑降（ダウンヒル）、回転（スラローム）、大回転（ジャイアントスラローム）、スーパー大回転（スーパーG）、複合が主な種目です。", "topic": "sports"},
    {"q": "駅伝とは何ですか？", "a": "駅伝は複数の走者がリレー形式で一定区間を走る長距離継走競技です。正月の箱根駅伝（関東学生駅伝）が日本で最も有名な大会です。", "topic": "sports"},
    # More medical
    {"q": "アレルギーとは？", "a": "アレルギーは本来無害な物質（アレルゲン）に対して免疫系が過剰反応する状態です。花粉症、食物アレルギー、アナフィラキシーなど様々な型があります。", "topic": "medical"},
    {"q": "腎不全の治療は？", "a": "慢性腎不全の末期では透析（血液透析または腹膜透析）または腎移植が必要です。早期には原疾患治療・生活習慣改善・降圧療法が重要です。", "topic": "medical"},
    {"q": "肝炎ウイルスの種類は？", "a": "A型（経口感染・急性）、B型（血液・性感染・慢性化あり）、C型（血液感染・慢性化が多く肝硬変・肝がんリスク）などがあります。", "topic": "medical"},
    {"q": "認知症の種類は？", "a": "アルツハイマー型（最多）、血管性認知症、レビー小体型認知症、前頭側頭型認知症が四大認知症です。それぞれ症状・経過・治療が異なります。", "topic": "medical"},
    {"q": "うつ病の治療法は？", "a": "薬物療法（SSRI・SNRI等の抗うつ薬）と心理療法（認知行動療法）の組み合わせが有効です。休養と社会的サポートも重要です。", "topic": "medical"},
    {"q": "骨粗しょう症の予防は？", "a": "カルシウム・ビタミンD摂取、適度な運動（荷重運動）、禁煙・節酒が基本です。骨密度検査で早期発見が可能です。", "topic": "medical"},
    {"q": "胃潰瘍の原因は？", "a": "ヘリコバクター・ピロリ菌感染とNSAIDs（非ステロイド性抗炎症薬）の使用が主な原因です。ストレスも悪化因子となります。", "topic": "medical"},
    {"q": "花粉症の治療は？", "a": "抗ヒスタミン薬、鼻噴霧ステロイド薬が主な治療薬です。根本的治療としてアレルゲン免疫療法（舌下免疫療法）も有効です。", "topic": "medical"},
    {"q": "救急車を呼ぶ基準は？", "a": "意識消失・呼吸停止・激しい胸痛・脳卒中症状（突然の片麻痺・構音障害）・高度外傷・大量出血などは迷わず119番に連絡してください。", "topic": "medical"},
    {"q": "AED（自動体外式除細動器）の使い方は？", "a": "電源を入れ音声ガイドに従います。パッドを右鎖骨下と左わき腹に貼り付け、解析・ショック推奨が出たら全員がタッチしていないことを確認してボタンを押します。", "topic": "medical"},
    # More science
    {"q": "光合成の反応式は？", "a": "6CO₂ + 6H₂O + 光エネルギー → C₆H₁₂O₆ + 6O₂　がグルコース合成の全体反応式です。", "topic": "science"},
    {"q": "酵素とは何ですか？", "a": "酵素はタンパク質でできた生体触媒で、特定の化学反応を促進します。基質特異性があり、温度・pHに依存して活性が変化します。", "topic": "science"},
    {"q": "半導体とは？", "a": "半導体は電気の伝導性が導体と絶縁体の中間にある材料です。シリコンが代表的で、トランジスタ・集積回路の基材として現代電子産業の基盤です。", "topic": "science"},
    {"q": "核分裂と核融合の違いは？", "a": "核分裂は重い核（ウランなど）が分裂してエネルギーを放出（原子炉・核爆弾）。核融合は軽い核（水素同位体）が合体してエネルギーを放出（太陽・核融合炉）。", "topic": "science"},
    {"q": "ビッグバンとは？", "a": "宇宙は約138億年前に極高温・高密度の特異点から膨張を開始したというのがビッグバン理論です。宇宙マイクロ波背景放射がその証拠です。", "topic": "science"},
    # More sports
    {"q": "野球のDHとは？", "a": "DH（指名打者制度）は投手に代わって打撃専門の選手を起用できるルールです。日本プロ野球ではパ・リーグが採用、セ・リーグは採用していません。", "topic": "sports"},
    {"q": "サッカーワールドカップの開催頻度は？", "a": "FIFAワールドカップは4年に1度開催されます。2026年大会からは出場国が32カ国から48カ国に拡大されます。", "topic": "sports"},
    {"q": "水泳の世界記録で最速の種目は？", "a": "短距離種目（50m自由形）が最速で、世界記録は男子約20秒台です。長距離になるほど単位距離当たりのペースは落ちます。", "topic": "sports"},
    {"q": "陸上の投擲種目は？", "a": "砲丸投げ、円盤投げ、ハンマー投げ、やり投げの4種目が陸上の投擲競技です。", "topic": "sports"},
    {"q": "体操競技の採点方式は？", "a": "2006年から10点満点制が廃止され、Dスコア（難度点）とEスコア（実施点）の合算方式になりました。Dスコアに上限はなく高難度演技が評価されます。", "topic": "sports"},
    # Greetings & fallback
    {"q": "こんにちは", "a": "こんにちは！何かご質問がありましたらお気軽にどうぞ。医療、科学、スポーツについての情報をお伝えできます。", "topic": "greeting"},
    {"q": "ありがとう", "a": "どういたしまして！他にご質問があればお気軽にどうぞ。", "topic": "greeting"},
    {"q": "さようなら", "a": "ありがとうございました。またいつでもご質問ください。お気をつけて！", "topic": "farewell"},
    {"q": "お元気ですか", "a": "ありがとうございます。あなたのご質問にお役に立てるよう準備しています！", "topic": "greeting"},
    {"q": "よろしくお願いします", "a": "こちらこそよろしくお願いします。何でもご質問ください。", "topic": "greeting"},
]


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[ぁ-ん]+|[ァ-ン]+|[一-龥]+|[a-z]+|\d+", text)


# ---------------------------------------------------------------------------
# TF-IDF Retrieval
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
        self.vocab = sorted(df.keys())
        self.idf = {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in self.vocab}

    def transform(self, tokens: List[str]) -> np.ndarray:
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        vec = np.zeros(len(self.vocab), dtype=np.float64)
        for i, term in enumerate(self.vocab):
            if term in tf:
                vec[i] = (tf[term] / total) * self.idf.get(term, 0.0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-10 else vec


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------

INTENT_PATTERNS = {
    "greeting": [
        r"こんにち[はわ]", r"おはよう", r"こんばん[はわ]", r"はじめまして",
        r"よろしく", r"お元気", r"ありがとう", r"どうも",
    ],
    "farewell": [
        r"さようなら", r"またね", r"バイバイ", r"終わり", r"おやすみ", r"失礼します",
    ],
    "question": [
        r"[はわ]？", r"とは", r"ですか", r"でしょうか", r"方法", r"原因",
        r"種類", r"違い", r"教えて", r"について", r"何ですか",
    ],
}


def classify_intent(text: str) -> str:
    for intent, patterns in INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text):
                return intent
    return "unknown"


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

DISEASE_NAMES = [
    "高血圧", "糖尿病", "心筋梗塞", "脳卒中", "脳梗塞", "脳出血", "がん", "癌",
    "肺炎", "喘息", "COPD", "慢性閉塞性肺疾患", "腎不全", "肝炎", "認知症",
    "アルツハイマー", "うつ病", "骨粗しょう症", "胃潰瘍", "花粉症", "アレルギー",
    "貧血", "甲状腺機能亢進症", "バセドウ病", "敗血症", "心不全", "不整脈", "AF", "狭心症",
]

DRUG_NAMES = [
    "アスピリン", "スタチン", "ワルファリン", "メトホルミン", "インスリン",
    "ACE阻害薬", "ARB", "カルシウム拮抗薬", "利尿薬", "β遮断薬",
    "ロキソプロフェン", "イブプロフェン", "アセトアミノフェン",
    "抗生物質", "ステロイド", "SSRI", "SNRI", "抗ヒスタミン薬",
    "アモキシシリン", "レボフロキサシン", "アトルバスタチン", "ロスバスタチン",
]

MEDICAL_TERMS = [
    "血圧", "血糖", "コレステロール", "LDL", "HDL", "中性脂肪", "ヘモグロビン",
    "白血球", "血小板", "クレアチニン", "AST", "ALT", "INR", "PSA", "NT-proBNP",
    "心電図", "ECG", "MRI", "CT", "超音波", "内視鏡", "透析", "手術",
    "ワクチン", "免疫", "炎症", "感染", "抗体", "受容体",
]


def extract_entities(text: str) -> Dict[str, List[str]]:
    found_diseases = [d for d in DISEASE_NAMES if d in text]
    found_drugs = [d for d in DRUG_NAMES if d in text]
    found_terms = [t for t in MEDICAL_TERMS if t in text]
    return {
        "diseases": found_diseases,
        "drugs": found_drugs,
        "medical_terms": found_terms,
    }


# ---------------------------------------------------------------------------
# Dialogue System
# ---------------------------------------------------------------------------

class DialogueSystem:
    def __init__(self, kb: List[Dict[str, str]], context_window: int = 3,
                 top_k: int = 3, similarity_threshold: float = 0.05):
        self.kb = kb
        self.context_window = context_window
        self.top_k = top_k
        self.threshold = similarity_threshold
        self.context: deque = deque(maxlen=context_window)
        self.turn_count = 0

        # Fit TF-IDF on knowledge base questions
        self.tfidf = TFIDF()
        all_docs = [tokenize(item["q"] + " " + item["a"]) for item in kb]
        self.tfidf.fit(all_docs)
        self.kb_vecs = np.stack([self.tfidf.transform(d) for d in all_docs])

    def _context_topic_boost(self, kb_idx: int) -> float:
        """Boost score for items matching recent conversation topics."""
        if not self.context:
            return 0.0
        recent_topics = [entry.get("topic", "") for entry in self.context]
        item_topic = self.kb[kb_idx].get("topic", "")
        return 0.1 * sum(1 for t in recent_topics if t == item_topic) / len(recent_topics)

    def retrieve(self, query: str) -> Tuple[List[int], List[float]]:
        """Retrieve top-k KB entries by TF-IDF similarity + context boost."""
        q_tokens = tokenize(query)
        q_vec = self.tfidf.transform(q_tokens)
        sims = np.array([cosine_sim(q_vec, kv) for kv in self.kb_vecs])

        # Apply context boost
        for i in range(len(self.kb)):
            sims[i] += self._context_topic_boost(i)

        top_k_idx = np.argsort(sims)[::-1][:self.top_k].tolist()
        top_k_scores = [float(sims[i]) for i in top_k_idx]
        return top_k_idx, top_k_scores

    def respond(self, user_input: str) -> Dict:
        self.turn_count += 1
        intent = classify_intent(user_input)
        entities = extract_entities(user_input)
        retrieved_idx, scores = self.retrieve(user_input)
        best_score = scores[0] if scores else 0.0
        best_idx = retrieved_idx[0] if retrieved_idx else -1

        if best_score < self.threshold or best_idx < 0:
            response = "その質問についての情報が見つかりませんでした。医療、科学、スポーツに関することでしたら回答できます。"
            topic = "unknown"
        else:
            response = self.kb[best_idx]["a"]
            topic = self.kb[best_idx].get("topic", "unknown")

        turn_info = {
            "turn": self.turn_count,
            "input": user_input,
            "intent": intent,
            "entities": entities,
            "retrieved_q": self.kb[best_idx]["q"] if best_idx >= 0 else "",
            "score": best_score,
            "response": response,
            "topic": topic,
        }
        self.context.append(turn_info)
        return turn_info

    def reset(self):
        self.context.clear()
        self.turn_count = 0


# ---------------------------------------------------------------------------
# Evaluation on held-out Q&A pairs
# ---------------------------------------------------------------------------

def evaluate_retrieval(system: DialogueSystem, held_out: List[Dict]) -> Dict:
    """
    Measure retrieval accuracy on a closed test:
    The held-out Q&A pairs must exist in system.kb for exact-match evaluation.
    Accuracy = fraction where top-1 retrieved answer matches the expected answer.

    Note: If held_out items are NOT in system.kb (open-test), we fall back to
    topic-accuracy: top-1 retrieved item shares the same topic as the gold item.
    """
    correct = 0
    topic_correct = 0
    results = []
    for item in held_out:
        q = item["q"]
        expected_a = item["a"]
        expected_topic = item.get("topic", "")
        idx_list, scores = system.retrieve(q)
        best_idx = idx_list[0] if idx_list else -1
        predicted_a = system.kb[best_idx]["a"] if best_idx >= 0 else ""
        predicted_topic = system.kb[best_idx].get("topic", "") if best_idx >= 0 else ""
        exact_match = predicted_a == expected_a
        topic_match = predicted_topic == expected_topic and expected_topic != ""
        if exact_match:
            correct += 1
        if topic_match:
            topic_correct += 1
        results.append({
            "query": q,
            "expected": expected_a[:40] + "...",
            "predicted": predicted_a[:40] + "...",
            "correct": exact_match,
            "topic_match": topic_match,
            "score": scores[0] if scores else 0.0,
        })
    n = max(len(held_out), 1)
    return {
        "exact_accuracy": correct / n,
        "topic_accuracy": topic_correct / n,
        "correct": correct,
        "topic_correct": topic_correct,
        "total": len(held_out),
        "details": results,
    }


# ---------------------------------------------------------------------------
# Demo conversation
# ---------------------------------------------------------------------------

DEMO_TURNS = [
    "こんにちは。医療について質問があります。",
    "高血圧とはどんな病気ですか？",
    "高血圧の治療薬を教えてください",
    "ACE阻害薬とはどんな薬ですか？",
    "糖尿病との関係はありますか？",
    "糖尿病の合併症について教えてください",
    "腎不全になった場合の治療方法は？",
    "透析についてもう少し詳しく知りたいです",
    "血液検査で腎機能を調べるにはどうすればいいですか？",
    "ありがとうございました。さようなら。",
]


def run_demo(system: DialogueSystem):
    print("\n" + "=" * 70)
    print("Retrieval-Based Dialogue System — Medical Topic Demo")
    print("=" * 70)
    system.reset()

    for user_input in DEMO_TURNS:
        result = system.respond(user_input)
        print(f"\n[Turn {result['turn']}]")
        print(f"  User   : {result['input']}")
        print(f"  Intent : {result['intent']}")
        if any(result["entities"].values()):
            ent_str = ", ".join(
                f"{k}={v}" for k, v in result["entities"].items() if v
            )
            print(f"  Entities: {ent_str}")
        print(f"  Match  : {result['retrieved_q'][:40]}... (score={result['score']:.3f})")
        print(f"  System : {result['response']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    system = DialogueSystem(KNOWLEDGE_BASE, context_window=3, top_k=3, similarity_threshold=0.03)

    run_demo(system)

    # Evaluation: hold out 20 entries, train on the rest
    import random
    random.seed(42)
    held_out_indices = random.sample(range(len(KNOWLEDGE_BASE)), 20)
    held_out = [KNOWLEDGE_BASE[i] for i in held_out_indices]
    train_kb = [item for i, item in enumerate(KNOWLEDGE_BASE) if i not in held_out_indices]
    eval_system = DialogueSystem(train_kb, context_window=3, top_k=3, similarity_threshold=0.03)

    print("\n" + "=" * 70)
    print("Retrieval Accuracy Evaluation (held-out 20 Q&A pairs)")
    print("=" * 70)
    eval_result = evaluate_retrieval(eval_system, held_out)
    # Open-test (items removed from KB): exact match = 0 expected; topic accuracy shows semantic retrieval quality
    print(f"\nExact Accuracy : {eval_result['exact_accuracy']:.2%} ({eval_result['correct']}/{eval_result['total']})")
    print(f"Topic Accuracy : {eval_result['topic_accuracy']:.2%} ({eval_result['topic_correct']}/{eval_result['total']})")
    print("  (Topic accuracy = top-1 retrieved item shares the same topic as gold item)")
    print("\nSample results:")
    for detail in eval_result["details"][:5]:
        status = "OK" if detail["topic_match"] else "NG"
        print(f"  [{status}] Q: {detail['query'][:30]}... score={detail['score']:.3f}")
