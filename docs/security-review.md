# セキュリティレビュー: rgbx (RGB↔X) デモコード

- **対象**: `rgbx` リポジトリ（`zheng95z/rgbx` のクローン）の Python デモ群
- **実施日**: 2026-06-30
- **レビュアー**: Claude Code 自動セキュリティレビュー
- **対象環境**: `gradio==3.48.0`（`environment.yml`）/ gradio デモ・画像ローダ・パイプライン
- **総括**: 実運用上の急所は **「`share=True` での公開」**。下記2件はローカル限定なら低リスクだが、公式 README / `phase1-baseline-rgbx/SETUP.md` が Colab フォールバックとして勧める `share=True` 公開時に顕在化する。

> ⚠️ 注意: これは他者リポジトリ（`zheng95z/rgbx`, Adobe / SIGGRAPH 2024）由来のコードに対する指摘です。GitHub で公開・流用する際は、**upstream への責任ある開示（issue / security advisory）を先に検討**してください。

---

## 🔴 Finding 1: 任意パスへのファイル書き込み / パストラバーサル

- **深刻度**: Medium（ローカル）→ High（`share=True` 公開時）
- **場所**: `rgb2x/gradio_demo_rgb2x.py:98-107`（`trigger_save_all`）

```python
def trigger_save_all(save_dir, return_list):
    save_dir = str(save_dir).strip()
    os.makedirs(save_dir, exist_ok=True)               # ← 任意ディレクトリ作成
    for image, label in return_list:
        label = label.split("Generated ")[-1].replace(" ", "_") + ".png"
        image.save(os.path.join(save_dir, label))      # ← 任意パスへ書き込み
```

### 内容

- `save_dir` は UI テキストボックス（`save_dir_input`, 154行）由来で **完全にユーザー制御**。`/`・`..`・絶対パスの検証が一切ない。
- `os.makedirs` で任意ディレクトリを作成し、`image.save` で **プロセス権限で書ける任意の場所に PNG を書き込める**（既存 `.png` 資産の上書き、ホームディレクトリ配下への書き込み等）。
- デフォルト `demo.launch()`（180行）は 127.0.0.1 バインドのためローカル限定。しかし `SETUP.md` が勧める **`share=True`** で gradio.live 公開トンネルになると **リモートから悪用可能**。
- 他2デモ（`x2rgb` / `x2rgb_inpainting`）はこの `save_dir` パターンを持たず、本問題は **rgb2x のみ**。

### 修正案

```python
BASE = os.path.join(current_directory, "outputs")
def trigger_save_all(save_dir, return_list):
    dest = os.path.realpath(os.path.join(BASE, os.path.basename(str(save_dir).strip())))
    if not dest.startswith(os.path.realpath(BASE) + os.sep):
        return "Invalid folder"
    os.makedirs(dest, exist_ok=True)
    for image, label in return_list:
        label = label.split("Generated ")[-1].replace(" ", "_") + ".png"
        image.save(os.path.join(dest, label))
    return "Images saved"
```

---

## 🟠 Finding 2: 既知 CVE を含む古い gradio 依存

- **深刻度**: Medium（`share=True` / 非信頼ネットワーク公開時に顕在化）
- **場所**: `environment.yml` → `gradio==3.48.0`

### 内容

- このバージョンは **パストラバーサル / 任意ローカルファイル読み取り**の既知脆弱性の影響範囲。
  - 例: **CVE-2023-51449**（`/file=` エンドポイント経由の任意ファイル読み取り。3.41〜4.x 初期系が対象）
  - 後続の LFI 系（例: CVE-2024-1561）も同様に該当しうる
- Finding 1 と同じく **`share=True` で公開した瞬間にサーバ上のファイルを読まれ得る**点が実害。
- ※ 正確な影響バージョン範囲は各 CVE の一次情報で要確認。

### 修正案

- gradio をパッチ済みリリースへ更新（最低でも 4.x 系の最新、可能なら現行 5.x）。
- 更新できない場合は **`share=True` を使わない／信頼ネットワーク外に晒さない**。

---

## 🟡 補足（軽微・参考）

- `rgb2x/load_image.py:83` 他: `cv2.cvtColor(cv2.imread(path), ...)` は、読み込み失敗時に `cv2.imread` が `None` を返し例外になる（堅牢性の問題、セキュリティ影響は小）。同パターンは `x2rgb/`・`x2rgb_inpainting/` の `load_image.py` にも存在。

---

## 推奨アクション（優先度順）

1. **公開するなら gradio 更新 + `save_dir` 検証をセットで**（最優先）。
2. `share=True` を常用しない運用に。Colab 疎通確認のみに限定。
3. upstream（`zheng95z/rgbx`）へ責任ある開示を検討してから公開・流用する。
