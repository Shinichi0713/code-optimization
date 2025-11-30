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
