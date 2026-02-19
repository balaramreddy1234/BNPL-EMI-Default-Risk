/**
 * BNPL Manual Risk Assessment Logic
 * Features: INR Formatting, Smart Suggestion Display, and PDF Integration
 */
async function predict() {
  const fields = ["income", "loan", "emi", "tenure", "ontime", "delays", "credit"];
  const data = {};
  const resultEl = document.getElementById("result");

  // Validate Input Fields
  for (const id of fields) {
    const fieldObj = document.getElementById(id);
    const val = fieldObj.value.trim();
    if (!val) {
      resultEl.style.color = "#dc3545";
      resultEl.innerHTML = "❌ <strong>Error:</strong> All financial fields are required.";
      fieldObj.style.borderColor = "#dc3545";
      return;
    }
    fieldObj.style.borderColor = "#ddd";
    data[id] = parseFloat(val);
  }

  resultEl.innerHTML = "⏳ Assessing financial risk & generating deep-dive report...";
  resultEl.style.color = "#2c3e50";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    const result = await res.json();

    if (!result.success) {
      throw new Error(result.error || "Prediction failed");
    }

    const isHigh = result.risk.includes("HIGH");
    const isMed  = result.risk.includes("MEDIUM");
    const themeColor = isHigh ? "#dc3545" : isMed ? "#ff9800" : "#28a745";

    

    // Render the Unified Result Card
    resultEl.innerHTML = `
      <div style="padding:25px; border-radius:12px; border:2px solid ${themeColor}; background: ${themeColor}05; font-family: 'Segoe UI', sans-serif;">
        <h3 style="margin-top:0; color:${themeColor};">🛡️ Executive Verdict: ${result.risk}</h3>
        <p style="font-size:1.1em; color:#2c3e50;">Default Probability: <strong>${(result.probability * 100).toFixed(2)}%</strong></p>
        
        <div style="margin:20px 0; padding:15px; background:white; border-left:5px solid ${themeColor}; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <strong style="display:block; margin-bottom:5px;">💡 Proactive Financial Advice:</strong>
            <span style="color:#4a5568; line-height:1.5;">${result.suggestion}</span>
        </div>

        <div style="display:flex; gap:15px; align-items:center; margin-top:20px;">
            <a href="/download-report/${result.report_url}" target="_blank"
               style="background:${themeColor}; color:white; padding:12px 25px; border-radius:8px; text-decoration:none; font-weight:bold; box-shadow: 0 4px 10px ${themeColor}44;">
               📥 Download PDF Report
            </a>

            <a href="/pages/emi_history.html" style="color:#2c3e50; font-weight:bold; text-decoration:none; border-bottom:1px solid #2c3e50;">
               View Full History ➔
            </a>
        </div>
      </div>
    `;
  } catch (err) {
    resultEl.style.color = "#dc3545";
    resultEl.innerHTML = "❌ <strong>System Error:</strong> " + err.message;
  }
}