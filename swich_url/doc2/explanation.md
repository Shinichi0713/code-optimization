了解しました！

**「数秒遅延しても OK。クライアント HTML は完全静的で JS なし」**

という条件なら、**100% 実現可能**です。

そのための唯一の方法は：

---

# ✅ **方法：クライアント側に meta refresh を入れて、サーバーが返す HTML を動的に切り替える**

HTML 側に追加するのは  **meta refresh 1行だけ** 。

例：

```html
<meta http-equiv="refresh" content="5">
```

→ この場合、5秒ごとにブラウザがページを自動リロードします。

---

# 🎯 **切り替えの仕組み**

1. クライアントは「現在の表示中ページ」を定期的に再読み込みする
2. Express サーバーの `/current` ルートが「今表示すべきページ」を返す
3. `/api/update` が呼ばれたら、その「今表示すべきページ」を変更
4. 次のリロードのタイミングで、新しい画面へ切り替わる

---

# 📘 システム構成図

```
client.html  ← meta refreshで5秒ごとに自動リロード
      ↑
      │ GET /current
      │
Node.js
  ├─ /current  → 現在のページのHTMLを返す
  └─ /api/update → 次に返すHTMLを切り替える
```

---

# 🟢 Node.js（Express）実装例

以下はあなたの server.js に追加できる形です。

```js
const express = require("express");
const bodyParser = require("body-parser");
const fs = require("fs");
const path = require("path");

const app = express();
app.use(bodyParser.json());

// 現在表示すべきページをここで管理
let currentPage = "page1.html";

// /api/update で currentPage を切り替える
app.post("/api/update", (req, res) => {
  const { page } = req.body;

  if (!page) {
    return res.status(400).json({ message: "page is required" });
  }

  currentPage = page;  // ページ切替
  console.log("Set current page to:", page);

  res.json({ message: "page updated", page });
});

// 現在表示すべきHTMLを返す
app.get("/current", (req, res) => {
  const filePath = path.join(__dirname, currentPage);
  res.sendFile(filePath);
});

// サーバー起動
app.listen(3000, () => {
  console.log("Server running on 3000");
});
```

---

# 🟦 2つのクライアントページ例

## page1.html

```html
<html>
  <head>
    <meta http-equiv="refresh" content="5">
  </head>
  <body>
    <h1>Page 1</h1>
  </body>
</html>
```

## page2.html

```html
<html>
  <head>
    <meta http-equiv="refresh" content="5">
  </head>
  <body>
    <h1>Page 2</h1>
  </body>
</html>
```

---

# 🟣 クライアントのアクセス方法

ブラウザで以下を開くだけ：

```
http://localhost:3000/current
```

これで5秒ごとにページが自動再読込されます。

---

# 🟠 ページ切り替え方法（curl）

例：page2.html に切り替える

### PowerShell

```powershell
curl -X POST http://localhost:3000/api/update `
  -H "Content-Type: application/json" `
  -d "{\"page\":\"page2.html\"}"
```

---

# 🔄 遅延はどれくらい？

`<meta http-equiv="refresh" content="5">`

なら最大 5 秒後に切り替わります。

もっと早くしたいなら：

```
content="2"
```

や

```
content="1"
```

に変更できます。

---

# 🎉 **これでクライアント側の HTML に JS を追加せず、数秒遅延で画面切り替えが可能です！**

---

必要であれば：

* page.html をテンプレート化して共通化する方法
* 表示中ページが増える場合の自動ルーティング
* 切り替え履歴の保存
* 1台のみ切り替えたい（クッキー識別）

なども作れます。
