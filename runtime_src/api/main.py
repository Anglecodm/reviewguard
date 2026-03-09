from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from reviewguard.features import extract_features
from reviewguard.scrape import scrape_reviews
from reviewguard.service import ReviewGuardService

MODEL_PATH = os.getenv("MODEL_PATH", "models/reviewguard_model.joblib")
METADATA_PATH = os.getenv("METADATA_PATH", "models/reviewguard_metadata.json")

app = FastAPI(title="Review Guard API", version="1.0")

service = ReviewGuardService(MODEL_PATH, METADATA_PATH)


def _overall_probability(preds) -> float:
    if not preds:
        return 0.0
    return float(sum(float(p.score) for p in preds) / len(preds))


def _overall_verdict(probability: float, threshold: float) -> str:
    return "fraud" if float(probability) >= float(threshold) else "authentic"


class PredictRequest(BaseModel):
    text: str
    threshold: Optional[float] = None


class BatchPredictRequest(BaseModel):
    texts: List[str]
    threshold: Optional[float] = None


class IngestRequest(BaseModel):
    url: str
    limit: Optional[int] = 1000
    threshold: Optional[float] = None
    platform: Optional[str] = "auto"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Review Guard Console</title>
    <link rel="manifest" href="/manifest.json" />
    <style>
      :root {
        --bg: #eef2f8;
        --panel: #ffffff;
        --ink: #18212f;
        --muted: #5f6b7c;
        --primary: #1359c8;
        --primary-dark: #0b3d8f;
        --accent: #2bb673;
        --danger: #e5484d;
        --shadow: 0 16px 40px rgba(13, 35, 71, 0.12);
        --border: #e1e6ef;
        --card: #f7f9fc;
      }

      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Sora", "Space Grotesk", "IBM Plex Sans", "Segoe UI", sans-serif;
        color: var(--ink);
        background: radial-gradient(1000px 500px at 10% -10%, #e2ebff, transparent),
                    radial-gradient(1200px 600px at 90% -10%, #ddfff2, transparent),
                    var(--bg);
      }
      .wrap {
        max-width: 1260px;
        margin: 24px auto 56px;
        padding: 0 20px;
      }
      .window {
        background: var(--panel);
        border-radius: 16px;
        box-shadow: var(--shadow);
        overflow: hidden;
        border: 1px solid var(--border);
      }
      .window-bar {
        height: 42px;
        background: linear-gradient(90deg, #f9fbff, #f4f7fb);
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 0 14px;
      }
      .dot { width: 12px; height: 12px; border-radius: 50%; }
      .dot.red { background: #ff5f57; }
      .dot.yellow { background: #ffbd2e; }
      .dot.green { background: #28c840; }

      header {
        display: grid;
        grid-template-columns: 1.2fr 1.6fr;
        gap: 18px;
        padding: 20px;
        border-bottom: 1px solid var(--border);
        align-items: center;
      }
      .brand {
        display: flex;
        align-items: center;
        gap: 12px;
      }
      .logo {
        width: 36px;
        height: 36px;
        background: linear-gradient(145deg, #1359c8, #0b3d8f);
        border-radius: 10px;
        display: grid;
        place-items: center;
        color: white;
        font-weight: 700;
        letter-spacing: 0.5px;
      }
      .brand h1 {
        font-size: 18px;
        margin: 0;
        font-weight: 700;
      }
      .brand span { color: var(--primary); }
      .hero {
        display: flex;
        flex-direction: column;
        gap: 6px;
        justify-content: center;
      }
      .hero h2 {
        margin: 0;
        font-size: 18px;
      }
      .hero p {
        margin: 0;
        color: var(--muted);
        font-size: 13px;
      }
      .status {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 12px;
        color: var(--muted);
      }
      .pill {
        background: #e9f1ff;
        color: var(--primary);
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 700;
        letter-spacing: 0.3px;
      }
      .live {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: var(--accent);
        font-weight: 700;
      }
      .pulse {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 0 0 rgba(37, 165, 95, 0.6);
        animation: pulse 1.5s infinite;
      }
      @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(37, 165, 95, 0.6); }
        70% { box-shadow: 0 0 0 10px rgba(37, 165, 95, 0); }
        100% { box-shadow: 0 0 0 0 rgba(37, 165, 95, 0); }
      }

      .cards {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        padding: 18px 20px 0;
      }
      .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px;
        position: relative;
        overflow: hidden;
        min-height: 98px;
      }
      .card h3 {
        margin: 0;
        font-size: 13px;
        color: var(--muted);
        font-weight: 600;
      }
      .card .value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 8px;
        color: var(--primary);
      }
      .card.red .value { color: var(--danger); }
      .spark {
        position: absolute;
        right: -10px;
        bottom: -6px;
        opacity: 0.5;
      }

      .filters {
        margin: 16px 20px;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px;
        background: #f9fbff;
        display: grid;
        gap: 12px;
      }
      .filters h4 {
        margin: 0;
        font-size: 14px;
        color: var(--primary);
      }
      .filter-row {
        display: grid;
        grid-template-columns: 2fr 1fr 1fr auto;
        gap: 12px;
        align-items: center;
      }
      select, input[type="text"], input[type="url"] {
        width: 100%;
        padding: 10px 12px;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: white;
        color: var(--ink);
        font-size: 13px;
      }
      .btn {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 700;
        cursor: pointer;
      }
      .btn:active { transform: translateY(1px); }

      .content {
        display: grid;
        grid-template-columns: 2.2fr 1fr;
        gap: 16px;
        padding: 0 20px 20px;
      }
      .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
      }
      .panel h3 {
        margin: 0;
        padding: 12px 16px;
        background: linear-gradient(135deg, #1359c8, #2f7ee6);
        color: white;
        font-size: 14px;
        letter-spacing: 0.3px;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }
      th, td {
        padding: 10px 12px;
        border-bottom: 1px solid var(--border);
        text-align: left;
      }
      th { color: var(--muted); font-weight: 600; background: #f8fafc; }
      tr:hover { background: #f3f6ff; cursor: pointer; }
      .badge {
        background: var(--danger);
        color: white;
        font-weight: 700;
        padding: 4px 8px;
        border-radius: 8px;
        display: inline-block;
        min-width: 36px;
        text-align: center;
      }
      .user {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .avatar {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #dbe7ff;
        color: #1f5fbf;
        display: grid;
        place-items: center;
        font-weight: 700;
        font-size: 12px;
      }
      .details {
        padding: 12px 16px;
        font-size: 13px;
      }
      .details h4 { margin: 0 0 12px; }
      .details p { margin: 8px 0; color: var(--muted); }
      .details .strong { color: var(--ink); font-weight: 600; }
      .details .callout {
        margin-top: 12px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #ffe5e8;
        color: var(--danger);
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 700;
      }
      .cta {
        margin-top: 14px;
        width: 100%;
      }

      .analyze {
        margin: 0 20px 18px;
        padding: 14px 16px;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: #ffffff;
        display: grid;
        grid-template-columns: 2fr 1fr auto;
        gap: 12px;
        align-items: center;
      }
      .analyze textarea {
        width: 100%;
        min-height: 70px;
        border-radius: 10px;
        border: 1px solid var(--border);
        padding: 10px 12px;
        font-size: 13px;
      }
      .score-box {
        border: 1px dashed var(--border);
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        font-weight: 700;
        color: var(--primary);
      }
      .muted { color: var(--muted); }
      .fade-in { animation: fadeIn 0.7s ease; }
      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
      }

      @media (max-width: 980px) {
        .cards { grid-template-columns: 1fr; }
        .filter-row { grid-template-columns: 1fr; }
        .content { grid-template-columns: 1fr; }
        .analyze { grid-template-columns: 1fr; }
        header { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="window">
        <div class="window-bar">
          <span class="dot red"></span>
          <span class="dot yellow"></span>
          <span class="dot green"></span>
        </div>
        <header>
          <div class="brand">
            <div class="logo">RG</div>
            <div>
              <h1><span>ReviewGuard</span> Signal Console</h1>
              <div class="muted">Prototype fraud screening for e-commerce reviews</div>
            </div>
          </div>
          <div class="hero">
            <h2>Scan a storefront URL and get a final fraud verdict</h2>
            <p>This console returns explicit fraud probability and a hard authenticity verdict.</p>
            <div class="status">
              <span class="pill">PIPELINE READY</span>
              <span class="live"><span class="pulse"></span>ACTIVE</span>
              <span>Last updated: <span id="lastUpdated">just now</span></span>
            </div>
          </div>
        </header>

        <section class="cards">
          <div class="card">
            <h3>Total Reviews</h3>
            <div class="value" id="totalReviews">0</div>
            <svg class="spark" width="160" height="60" viewBox="0 0 160 60" fill="none">
              <path d="M0 50 C20 40, 40 45, 60 35 C80 25, 100 30, 120 18 C140 6, 160 12, 160 12 V60 H0 Z" fill="#cfe0ff"/>
            </svg>
          </div>
          <div class="card red">
            <h3>Flagged as Fraudulent</h3>
            <div class="value" id="fraudCount">0</div>
            <svg class="spark" width="160" height="60" viewBox="0 0 160 60" fill="none">
              <path d="M0 45 C25 38, 45 44, 65 30 C90 12, 120 40, 160 10 V60 H0 Z" fill="#ffd5db"/>
            </svg>
          </div>
          <div class="card">
            <h3>Authentic Reviews</h3>
            <div class="value" id="authCount">0</div>
            <svg class="spark" width="160" height="60" viewBox="0 0 160 60" fill="none">
              <path d="M0 52 C30 40, 70 54, 95 32 C120 12, 140 26, 160 8 V60 H0 Z" fill="#dff2ff"/>
            </svg>
          </div>
          <div class="card">
            <h3>Overall Fraud Probability</h3>
            <div class="value" id="overallFraudPct">0%</div>
            <svg class="spark" width="160" height="60" viewBox="0 0 160 60" fill="none">
              <path d="M0 48 C26 32, 56 46, 92 26 C116 12, 136 18, 160 6 V60 H0 Z" fill="#ffecc8"/>
            </svg>
          </div>
        </section>

        <section class="filters">
          <h4>Scan Target</h4>
          <div class="filter-row">
            <input id="storeUrl" type="url" placeholder="Paste e-commerce URL e.g. https://www.jumia.co.ke/product/..." />
            <select id="platformSelect">
              <option value="auto">Platform: Auto</option>
              <option value="jumia">Platform: Jumia (Live)</option>
              <option value="kilimall">Platform: Kilimall (Live)</option>
              <option value="amazon">Platform: Amazon (API)</option>
              <option value="ebay">Platform: eBay (API Live Beta)</option>
              <option value="etsy">Platform: Etsy (API Live - Key Required)</option>
              <option value="shopify">Platform: Shopify (API Live Beta)</option>
              <option value="woocommerce">Platform: WooCommerce (API Live)</option>
              <option value="walmart">Platform: Walmart (API)</option>
            </select>
            <select id="windowSelect">
              <option>Window: Last 7 Days</option>
              <option>Window: Last 30 Days</option>
              <option>Window: Last 90 Days</option>
            </select>
            <button class="btn" onclick="scanUrl()">Scan URL</button>
          </div>
          <div class="muted">Live URL scraping: Jumia + Kilimall. eBay and Shopify API modes are live beta; WooCommerce API mode is live; Etsy API mode is live with API key. Other API platforms require integration setup.</div>
          <div class="muted" id="scanStatus">Paste a URL and press Scan.</div>
          <div class="muted" id="finalVerdictSummary">Final Verdict: -</div>
        </section>

        <section class="content">
          <div class="panel">
            <h3>Flagged Reviews</h3>
            <table id="reviewsTable">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>User</th>
                  <th>Review Text</th>
                  <th>Exclamations</th>
                  <th>Emojis</th>
                  <th>Promo Words</th>
                  <th>Final Verdict</th>
                  <th>Fraud Probability</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>
          <div class="panel">
            <h3>Suspicious Review Details</h3>
            <div class="details" id="details">
              <h4>Pick a review</h4>
              <p class="muted">Click a row to see details here.</p>
            </div>
          </div>
        </section>
      </div>
    </div>

    <script>
      if ("serviceWorker" in navigator) {
        window.addEventListener("load", () => {
          navigator.serviceWorker.register("/service-worker.js");
        });
      }
      let sampleRows = [];

      function initTable() {
        const tbody = document.querySelector("#reviewsTable tbody");
        tbody.innerHTML = "";
        if (!sampleRows.length) {
          const tr = document.createElement("tr");
          tr.innerHTML = `<td colspan="8" class="muted">No reviews loaded yet. Paste a URL and click Scan.</td>`;
          tbody.appendChild(tr);
          return;
        }
        for (const row of sampleRows) {
          const tr = document.createElement("tr");
          tr.classList.add("fade-in");
          tr.innerHTML = `
            <td>${row.ts}</td>
            <td>
              <div class="user">
                <span class="avatar">${row.user.slice(0,2).toUpperCase()}</span>
                ${row.user}
              </div>
            </td>
            <td>${row.text}</td>
            <td>${row.ex}</td>
            <td>${row.em}</td>
            <td>${row.promo}</td>
            <td>${row.final_verdict || row.decision}</td>
            <td><span class="badge">${Number(row.fraud_probability_pct ?? row.score ?? 0).toFixed(1)}%</span></td>
          `;
          tr.addEventListener("click", () => showDetails(row));
          tbody.appendChild(tr);
        }
      }

      function showDetails(row) {
        const el = document.getElementById("details");
        el.innerHTML = `
          <h4>User: <span class="strong">${row.user}</span></h4>
          <p>Timestamp: <span class="strong">${row.ts}</span></p>
          <p>Review: <span class="strong">${row.text}</span></p>
          <p>Exclamations: <span class="strong">${row.ex}</span></p>
          <p>Promo Words: <span class="strong">${row.promo}</span></p>
          <p>Emoji Count: <span class="strong">${row.em}</span></p>
          <p>Final Verdict: <span class="strong">${row.final_verdict || row.decision}</span></p>
          <p>Confidence Signal: <span class="strong">${row.confidence_signal || row.reason || "-"}</span></p>
          <div class="callout">Fraud Probability: ${Number(row.fraud_probability_pct ?? row.score ?? 0).toFixed(1)}%</div>
          <button class="btn cta" onclick="markReviewed()">Mark as Reviewed</button>
        `;
      }

      async function scanUrl() {
        const url = document.getElementById("storeUrl").value.trim();
        const platform = document.getElementById("platformSelect").value;
        if (!url) {
          alert("Paste an e-commerce URL first.");
          return;
        }
        document.getElementById("scanStatus").textContent = "Scanning... generating prototype results.";
        const res = await fetch("/ingest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url, limit: 1000, platform })
        });
        const data = await res.json();
        if (data.error) {
          document.getElementById("scanStatus").textContent =
            `Scan failed: ${data.error}`;
          return;
        }
        sampleRows = data.rows || [];
        document.getElementById("totalReviews").textContent = data.total_reviews;
        document.getElementById("fraudCount").textContent = data.flagged;
        document.getElementById("authCount").textContent = data.authentic;
        document.getElementById("overallFraudPct").textContent =
          `${Number(data.fraud_probability_pct || 0).toFixed(1)}%`;
        document.getElementById("lastUpdated").textContent = "just now";
        document.getElementById("scanStatus").textContent =
          `Loaded ${data.total_reviews} reviews from ${data.source_url}.`;
        document.getElementById("finalVerdictSummary").textContent =
          `Final Verdict: ${(data.final_verdict || "-").toUpperCase()} | Overall Fraud Probability: ${Number(data.fraud_probability_pct || 0).toFixed(1)}%`;
        initTable();
        if (sampleRows.length) {
          showDetails(sampleRows[0]);
        }
      }

      function markReviewed() {
        alert("Marked as reviewed.");
      }

      document.getElementById("scanStatus").textContent = "Paste a URL and press Scan.";
      document.getElementById("finalVerdictSummary").textContent = "Final Verdict: -";
      initTable();
    </script>
  </body>
</html>
"""


@app.get("/manifest.json", response_class=JSONResponse)
def manifest():
    return {
        "name": "Review Guard Console",
        "short_name": "ReviewGuard",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#eef2f8",
        "theme_color": "#1359c8",
        "icons": [
            {"src": "/icon.svg", "sizes": "192x192", "type": "image/svg+xml"},
            {"src": "/icon.svg", "sizes": "512x512", "type": "image/svg+xml"},
        ],
    }


@app.get("/service-worker.js")
def service_worker():
    js = """
self.addEventListener("install", event => {
  event.waitUntil(
    caches.open("reviewguard-v1").then(cache => cache.addAll(["/"]))
  );
});

self.addEventListener("fetch", event => {
  event.respondWith(
    caches.match(event.request).then(response => response || fetch(event.request))
  );
});
"""
    return Response(content=js, media_type="application/javascript")


@app.get("/icon.svg")
def icon_svg():
    svg = """
<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#1359c8"/>
      <stop offset="1" stop-color="#0b3d8f"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" rx="96" fill="url(#g)"/>
  <path d="M256 110l126 48v86c0 86-54 166-126 190-72-24-126-104-126-190v-86l126-48z" fill="#fff" opacity="0.95"/>
  <path d="M224 270l-36-36 22-22 36 36 70-70 22 22-92 92z" fill="#1359c8"/>
</svg>
"""
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/predict")
def predict(request: PredictRequest):
    threshold = request.threshold
    pred = service.predict([request.text], threshold=threshold)[0]
    effective_threshold = threshold if threshold is not None else service.default_threshold
    fraud_probability_pct = round(float(pred.score) * 100.0, 2)
    return {
        "final_verdict": pred.decision,
        "decision": pred.decision,
        "label": pred.label,
        "fraud_probability": pred.score,
        "fraud_probability_pct": fraud_probability_pct,
        "score": pred.score,
        "confidence_signal": pred.reason,
        "reason": pred.reason,
        "threshold": effective_threshold,
    }


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest):
    threshold = request.threshold
    preds = service.predict(request.texts, threshold=threshold)
    effective_threshold = threshold if threshold is not None else service.default_threshold
    overall_prob = _overall_probability(preds)
    fraud_reviews = sum(1 for p in preds if p.decision == "fraud")
    authentic_reviews = sum(1 for p in preds if p.decision == "authentic")
    return {
        "threshold": effective_threshold,
        "overall_fraud_probability": overall_prob,
        "overall_fraud_probability_pct": round(overall_prob * 100.0, 2),
        "final_verdict": _overall_verdict(overall_prob, effective_threshold),
        "fraud_reviews": fraud_reviews,
        "authentic_reviews": authentic_reviews,
        "results": [
            {
                "final_verdict": p.decision,
                "decision": p.decision,
                "label": p.label,
                "fraud_probability": p.score,
                "fraud_probability_pct": round(float(p.score) * 100.0, 2),
                "score": p.score,
                "confidence_signal": p.reason,
                "reason": p.reason,
            }
            for p in preds
        ],
    }


@app.post("/ingest")
def ingest(request: IngestRequest):
    threshold = request.threshold
    limit = max(1, min(request.limit or 1000, 1000))
    effective_threshold = threshold if threshold is not None else service.default_threshold

    error = None
    try:
        scraped = scrape_reviews(request.url, limit=limit, platform_hint=request.platform)
        rows = [item.text for item in scraped]
    except Exception as exc:
        rows = []
        error = str(exc)

    if not rows:
        return {
            "source_url": request.url,
            "total_reviews": 0,
            "flagged": 0,
            "authentic": 0,
            "needs_review": 0,
            "overall_fraud_probability": 0.0,
            "fraud_probability_pct": 0.0,
            "final_verdict": "authentic",
            "threshold": effective_threshold,
            "rows": [],
            "error": error or "No reviews found. The page may block scraping or needs a headless browser.",
        }

    preds = service.predict(rows, threshold=threshold)
    now = datetime.now()

    results = []
    fraud_count = 0
    authentic_count = 0
    for idx, (item, pred) in enumerate(zip(scraped, preds)):
        text = item.text
        feats = extract_features(text, keywords=service.keywords)
        score_pct = round(float(pred.score) * 100.0, 2)
        if pred.decision == "fraud":
            fraud_count += 1
        else:
            authentic_count += 1
        if item.date:
            ts = item.date
        else:
            stamp = now - timedelta(minutes=idx * 7)
            ts = stamp.strftime("%a, %I:%M %p")
        results.append(
            {
                "ts": ts,
                "user": item.user or f"User{idx + 1:02d}",
                "text": text,
                "ex": int(feats.get("exclamation_count", 0)),
                "em": int(feats.get("emoji_count", 0)),
                "promo": int(feats.get("promo_keyword_count", 0)),
                "final_verdict": pred.decision,
                "decision": pred.decision,
                "confidence_signal": pred.reason,
                "reason": pred.reason,
                "fraud_probability": pred.score,
                "fraud_probability_pct": score_pct,
                "score": score_pct,
            }
        )

    total = len(results)
    overall_prob = _overall_probability(preds)
    overall_prob_pct = round(overall_prob * 100.0, 2)
    flagged_ratio_pct = round((fraud_count / total) * 100.0, 2) if total else 0.0
    return {
        "source_url": request.url,
        "total_reviews": total,
        "flagged": fraud_count,
        "authentic": authentic_count,
        "needs_review": 0,
        "flagged_ratio_pct": flagged_ratio_pct,
        "overall_fraud_probability": overall_prob,
        "fraud_probability_pct": overall_prob_pct,
        "final_verdict": _overall_verdict(overall_prob, effective_threshold),
        "threshold": effective_threshold,
        "rows": results,
        "error": error,
    }
