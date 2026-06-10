/**
 * server.js — Express backend for the Legal RAG UI
 *
 * Acts as a thin proxy between the browser and your Python FastAPI backend.
 * Run with: node server.js
 * Requires: npm install express multer axios form-data
 *
 * Make sure your FastAPI backend is running:
 *   uvicorn src.api:app --port 8000
 */

const express  = require("express");
const multer   = require("multer");
const axios    = require("axios");
const FormData = require("form-data");
const path     = require("path");

const app    = express();
const upload = multer({ storage: multer.memoryStorage() }); // keep file in RAM, forward to FastAPI

// URL of your Python FastAPI backend
const API_BASE = process.env.API_BASE || "http://localhost:8000";

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// ── Proxy: upload + index a document ────────────────────────────────────────
app.post("/api/index", upload.single("file"), async (req, res) => {
  try {
    const form = new FormData();
    form.append("file", req.file.buffer, {
      filename:    req.file.originalname,
      contentType: req.file.mimetype,
    });
    const { data } = await axios.post(`${API_BASE}/index`, form, {
      headers: form.getHeaders(),
      timeout: 120_000,
    });
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.response?.data?.detail || e.message });
  }
});

// ── Proxy: ask a question ────────────────────────────────────────────────────
app.post("/api/query", async (req, res) => {
  try {
    const { data } = await axios.post(`${API_BASE}/query`, req.body, { timeout: 300_000 });
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.response?.data?.detail || e.message });
  }
});

// ── Proxy: summarise ─────────────────────────────────────────────────────────
app.post("/api/summarise", async (req, res) => {
  try {
    const { data } = await axios.post(`${API_BASE}/summarise`, req.body, { timeout: 300_000 });
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.response?.data?.detail || e.message });
  }
});

// ── Proxy: list documents ────────────────────────────────────────────────────
app.get("/api/documents", async (req, res) => {
  try {
    const { data } = await axios.get(`${API_BASE}/documents`, { timeout: 10_000 });
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.response?.data?.detail || e.message });
  }
});

// ── Proxy: delete document ───────────────────────────────────────────────────
app.delete("/api/documents/:doc_id", async (req, res) => {
  try {
    const { data } = await axios.delete(`${API_BASE}/documents/${req.params.doc_id}`, { timeout: 10_000 });
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.response?.data?.detail || e.message });
  }
});

// ── Proxy: run evaluation ────────────────────────────────────────────────────
app.post("/api/evaluate", async (req, res) => {
  try {
    const { data } = await axios.post(`${API_BASE}/evaluate`, req.body, { timeout: 600_000 });
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.response?.data?.detail || e.message });
  }
});

// ── Health ───────────────────────────────────────────────────────────────────
app.get("/api/health", async (req, res) => {
  try {
    const { data } = await axios.get(`${API_BASE}/health`, { timeout: 5_000 });
    res.json(data);
  } catch (e) {
    res.status(503).json({ error: "Backend unreachable" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Legal RAG UI → http://localhost:${PORT}`));
