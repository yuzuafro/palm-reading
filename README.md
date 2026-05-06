# 手相占い

PC と Android 端末で動く、PWA 対応の AR 手相占い Web アプリです。  
MediaPipe Hand Landmarker で手のランドマークを検出し、生命線・知能線・感情線を推定してカメラ映像へ重ねます。

## 機能

- カメラ映像上に手の骨格と推定された主要 3 線を AR オーバーレイ表示
- 手の開き方や指のバランスから簡易な手相診断を表示
- Android / PC Chrome でインストール可能な PWA 対応
- 静的ホスティングだけで動作し、GitHub Pages にデプロイ可能

## ローカル起動

```bash
npm install
npm run dev
```

`npm run dev` / `npm run build` の前に MediaPipe のモデルと Wasm を確認し、ローカルに無ければ自動で補充します。

ブラウザで表示したらカメラ権限を許可してください。  
カメラ API は HTTPS または `localhost` でのみ利用できます。

## ビルド

```bash
npm run build
```

生成物は `dist/` に出力されます。

## GitHub Pages デプロイ

このリポジトリには GitHub Actions による Pages デプロイ workflow (`.github/workflows/deploy.yml`) を含めています。  
`main` ブランチへ push すると workflow が `npm ci` → `npm run build` を実行し、生成された `dist/` を Pages に公開します。

1. GitHub に push する
2. リポジトリ設定で **Pages > Build and deployment > Source** を **GitHub Actions** にする
3. `main` ブランチへ push すると `dist/` が Pages に公開される

### MediaPipe asset の運用

- バージョンは `mediapipe-assets.config.json` の `assetVersion` で管理します
- ビルド時は `public/models` と `public/wasm` を自動補充し、その内容を Pages に含めます
- 必要なファイルが手元にない場合は、スクリプトが取得して `public/` 配下へ配置します

## 技術構成

- Vite
- React + TypeScript
- MediaPipe Tasks Vision (Hand Landmarker)
- build 前に MediaPipe asset を補充する Node.js スクリプト
- 手動登録の service worker / Web App Manifest

## アイコン編集

- `public/favicon.drawio.svg` が draw.io / diagrams.net で再編集する元データです
- `public/favicon.svg`、`icon-192.png`、`icon-512.png`、`apple-touch-icon.png` は `public/favicon.drawio.svg` を元にした配布用アセットです
- アイコン書き出し時は、紫とピンクの円が角丸アイコン領域からはみ出す部分を含めない前提です

## 注意

- 診断は手の線そのものを画像解析するものではなく、手のランドマークから推定した簡易占いです
- 初回読み込み時にモデルと Wasm を取得し、その後は service worker が主要アセットをキャッシュします

