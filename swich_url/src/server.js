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
