以下のような構成にすれば、**外部から update API を叩くと、Web アプリの表示中画面を別サイトへ強制的に切り替える**仕組みを実現できます。

Node.js + WebSocket（または SSE）を使うのが定番です。

---

# ✅ 要求仕様

* 外部に「update API」を公開する（例：`POST /api/update`）
* API にリクエストが来たら、

  **現在 Web サイトを見ている全ユーザーの画面を別 URL に切り替える**
* サーバ側で URL を更新 → クライアント画面が自動で切り替わる

---

# ✅ 実現する方法（最も一般的）

## **1. サーバは Node.js (Express) + WebSocket を使用**

* 外部 API で「切り替え先 URL」を更新
* WebSocket で全クライアントへ「画面切り替え」をブロードキャスト

---

# ✅ 全体イメージ

```
ユーザー端末 → Webページにアクセス
                    ↑
                    | WebSocket接続
                    |
Node.js サーバ（Express）
   ├─ API: POST /api/update   ← 外部サービスから呼ばれる
   └─ WebSocket サーバ : 全クライアントへ通知
```

ブラウザ側は WebSocket から「change_url: (URL)」の通知を受け取ったら

`window.location.href = URL;` で自動遷移。

---

# ✅ Node.js サーバ側コード例（Express + ws）

```javascript
// 1. ライブラリ
const express = require("express");
const bodyParser = require("body-parser");
const WebSocket = require("ws");

const app = express();
app.use(bodyParser.json());

// 2. WebSocket サーバ
const wss = new WebSocket.Server({ port: 8081 });

// 全クライアントにメッセージ送信
function broadcast(data) {
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(data));
    }
  });
}

// 3. 外部向け API
app.post("/api/update", (req, res) => {
  const { url } = req.body;

  if (!url) {
    return res.status(400).json({ message: "url is required" });
  }

  // 全クライアントへ通知
  broadcast({ type: "change_url", url });

  res.json({ message: "URL broadcasted", url });
});

// 4. 普通の HTTP サーバ
app.listen(3000, () => {
  console.log("API server running on port 3000");
  console.log("WebSocket running on port 8081");
});
```

---

# ✅ クライアント側 JavaScript（Webページ側）

```html
<script>
  // WebSocket に接続
  const ws = new WebSocket("ws://yourserver:8081");

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === "change_url") {
      console.log("Switching to:", data.url);
      window.location.href = data.url; // 別サイトにリダイレクト
    }
  };
</script>
```

---

# 📝 どう動くか？

1. 外部サービスが

   ```bash
   curl -X POST http://yourserver:3000/api/update -H "Content-Type: application/json" -d '{"url":"https://newsite.com"}'
   ```

   を実行
2. Node.js が WebSocket で全クライアントへ通知
3. ブラウザ側が通知を受け取り、表示しているページを

   `https://newsite.com` へ自動で変更

---

# 🔒 注意点

* 別サイトへ自動遷移するので、ブラウザのセキュリティ仕様により

  クロスドメインの条件次第では警告が出る場合があります
* 商用環境なら HTTPS + 認証付き API にすべき

  （トークン等を使う）

---

# 🔧 必要なら…

* Vue / React 用の書き方
* API 認証（例：Bearer Token）
* 特定ユーザーだけ切り替える仕組み
* Docker 構成

なども作成できます。

必要であれば教えてください！
