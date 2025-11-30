結論として、**ローカルファイル（C:...）をブラウザで開いた場合、そのままでは WebSocket で `ws://localhost:8081` にアクセスできます。**

ただし、URL の指定方法には注意点があります。

---

# ✅ 1. “file://” で開く場合の URL の書き方

あなたのファイル：

```
C:\Users\yoshinaga\Documents\Python Scripts\hard-related\code-optimization\swich_url\src\page.html
```

は、ブラウザでは次の URL で開けます：

```
file:///C:/Users/yoshinaga/Documents/Python%20Scripts/hard-related/code-optimization/swich_url/src/page.html
```

**ポイント：**

* Windows のパスは「\」ではなく「/」
* 先頭は **file:///**
* フォルダ名にスペースがある場合は `%20` に変換される

---

# ✅ 2. WebSocket はそのまま使える（問題なし）

client.html 内のコードは  **そのままで動きます** ：

```js
const ws = new WebSocket("ws://localhost:8081");
```

file:// でも ws:// は使用可能なので問題ありません。

---

# ⭐ 3. 実際の手順

## STEP 1

ブラウザで次の URL を直接入力：

```
file:///C:/Users/yoshinaga/Documents/Python%20Scripts/hard-related/code-optimization/swich_url/src/page.html
```

## STEP 2

ページが開いたら console に：

```
WebSocket 接続済み
```

などと表示されるはず。

## STEP 3

別端末または curl で `/api/update` を叩く：

```
curl -X POST http://localhost:3000/api/update \
  -H "Content-Type: application/json" \
  -d '{"url": "/page2"}'
```

## STEP 4

ブラウザが指定した URL へ自動で移動する。

---

# ❗ 注意：遷移先 URL がローカルの場合

`"url": "/page2"` とした場合は *HTTP サーバで提供されるパス* を意味します。

もしローカルの別の HTML ファイルに飛びたい場合は以下のように指定できます：

```
file:///C:/path/to/another.html
```

例：

```json
{
  "url": "file:///C:/Users/yoshinaga/Documents/Python%20Scripts/hard-related/code-optimization/swich_url/src/another.html"
}
```

---

# 🔍 まとめ

| 用途                                      | URLの指定例                        |
| ----------------------------------------- | ---------------------------------- |
| ローカル HTML をブラウザで開く            | `file:///C:/Users/.../page.html` |
| WebSocket 接続                            | `ws://localhost:8081`            |
| API の URL にローカルファイルを指定したい | `file:///C:/.../xxx.html`        |

---

必要であれば：

* 特定フォルダを Express で配信してローカル HTML を `http://localhost/...` で扱う方法
* Next.js / React 版のページ切り替え
* 1 台だけ URL 切り替えしたい場合の識別方法

もお作りできます！
