/**
 * BNPL Strategic Decision Engine
 * Features: PDF Table Extraction, Drill-Down Modal, and IST Time-stamping.
 */

// 1. Global Filter Logic: Category Toggles + Search
function applyFilters() {
    const searchQuery = document.getElementById('memberSearch').value.toLowerCase();
    const activeCategory = document.getElementById('categoryFilter').value;
    const cards = document.querySelectorAll('.member-card');

    cards.forEach(card => {
        const text = card.innerText.toLowerCase();
        const category = card.getAttribute('data-category');
        const matchesSearch = text.includes(searchQuery);
        const matchesCategory = (activeCategory === 'all' || activeCategory === category);
        card.style.display = (matchesSearch && matchesCategory) ? 'flex' : 'none';
    });
}

// 2. Individual PDF Generator [Outcome: Transparency]
async function generateMemberPDF(memberData) {
    const btn = document.getElementById('pdfBtn');
    const originalText = btn.innerHTML;
    try {
        btn.innerHTML = "⏳ Generating Report...";
        btn.disabled = true;

        const res = await fetch("/generate-member-pdf", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(memberData)
        });
        
        const result = await res.json();
        if (result.success) {
            window.open(`/download-report/${result.report_url}`, '_blank');
        } else {
            alert("Error: " + result.error);
        }
    } catch (err) {
        alert("System error generating report.");
    } finally {
        if(btn) {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    }
}

// 3. Drill-Down Modal: Analytics + Suggestions
function showMemberDetails(encodedData) {
    const m = JSON.parse(decodeURIComponent(encodedData));
    const emiBurden = ((m.emi / (m.income || 1)) * 100).toFixed(1);
    
    const overlay = document.createElement('div');
    overlay.id = "modalOverlay";
    overlay.style = "position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.85); display:flex; align-items:center; justify-content:center; z-index:3000; font-family: 'Segoe UI', sans-serif; backdrop-filter: blur(5px);";
    
    const modal = document.createElement('div');
    modal.style = "background:white; padding:30px; border-radius:16px; width:550px; position:relative; box-shadow:0 15px 50px rgba(0,0,0,0.5); max-height: 90vh; overflow-y: auto;";
    
    modal.innerHTML = `
        <button onclick="document.getElementById('modalOverlay').remove()" style="position:absolute; top:20px; right:20px; border:none; background:none; font-size:26px; cursor:pointer; color:#7f8c8d;">&times;</button>
        <h3 style="margin-top:0; color:#2c3e50; border-bottom:2px solid #eee; padding-bottom:12px;">👤 Profile Analysis: ${m.name}</h3>
        
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; margin-top:20px;">
            <div><small style="color:#7f8c8d;">MONTHLY INCOME</small><div style="font-weight:bold;">₹${Number(m.income).toLocaleString('en-IN')}</div></div>
            <div><small style="color:#7f8c8d;">LOAN AMOUNT</small><div style="font-weight:bold;">₹${Number(m.loan).toLocaleString('en-IN')}</div></div>
            <div><small style="color:#7f8c8d;">TENURE</small><div style="font-weight:bold;">${m.tenure} Months</div></div>
            <div><small style="color:#7f8c8d;">CIBIL SCORE</small><div style="font-weight:bold; color:#2980b9;">${m.credit}</div></div>
            <div><small style="color:#7f8c8d;">ON-TIME PMTS</small><div style="font-weight:bold; color:#27ae60;">${m.ontime}</div></div>
            <div><small style="color:#7f8c8d;">PAST DELAYS</small><div style="font-weight:bold; color:#e74c3c;">${m.delays}</div></div>
        </div>

        <div style="margin-top:25px; padding:15px; border-radius:10px; background:#f8f9fa;">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="font-size:0.85em; font-weight:bold;">EMI BURDEN RATIO</span>
                <span style="font-size:0.85em; font-weight:bold; color:${emiBurden > 50 ? '#e74c3c' : '#27ae60'};">${emiBurden}%</span>
            </div>
            <div style="width:100%; background:#eee; height:10px; border-radius:5px;">
                <div style="width:${Math.min(emiBurden, 100)}%; background:${emiBurden > 50 ? '#e74c3c' : '#27ae60'}; height:10px; border-radius:5px;"></div>
            </div>
        </div>

        <div style="margin-top:25px; padding:18px; background:#f0f7ff; border-radius:10px; border-left:6px solid #2980b9;">
            <strong style="color:#2c3e50; display:block; margin-bottom:5px;">💡 Strategic Advice:</strong>
            <span style="color:#34495e; font-size:0.95em;">${m.suggestion || "Manual verification required."}</span>
        </div>

        <button id="pdfBtn" onclick='generateMemberPDF(${JSON.stringify(m)})' 
                style="margin-top:20px; width:100%; background:#2c3e50; color:white; padding:14px; border-radius:10px; border:none; cursor:pointer; font-weight:bold;">
            📥 Download Detailed PDF Report
        </button>
    `;
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
    overlay.onclick = (e) => { if(e.target === overlay) overlay.remove(); };
}

// 4. Batch Upload (Supports CSV & PDF)
async function upload() {
    const fileInput = document.getElementById("fileInput");
    const analysisEl = document.getElementById("analysis");
    if (!fileInput.files[0]) return;
    
    analysisEl.innerHTML = `<div style="text-align:center; padding:40px;"><div class="spinner"></div><p>Engine is calculating risk probabilities...</p></div>`;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("/upload", { method: "POST", body: formData });
        const data = await response.json();

        if (data.success) {
            const buildList = (members, color, catLabel) => {
                if (!members || members.length === 0) return `<li style="list-style:none; color:#999; text-align:center; padding:15px;">No records</li>`;
                return members.map(m => {
                    const encoded = encodeURIComponent(JSON.stringify(m));
                    return `
                        <li class="member-card" data-category="${catLabel}" onclick="showMemberDetails('${encoded}')"
                            style="margin-bottom:10px; cursor:pointer; list-style:none; padding:12px; background:white; border-radius:10px; border:1px solid #eee; display:flex; justify-content:space-between; align-items:center;">
                            <div><strong style="color:#2c3e50;">${m.name}</strong><br><small>Credit: ${m.credit}</small></div>
                            <div style="text-align:right;"><div style="color:${color}; font-weight:bold;">₹${Number(m.loan).toLocaleString('en-IN')}</div></div>
                        </li>`;
                }).join('');
            };

            analysisEl.innerHTML = `
                <div style="display:flex; gap:15px; margin-bottom:20px; background:#fff; padding:15px; border-radius:12px; box-shadow:0 4px 10px rgba(0,0,0,0.05);">
                    <input type="text" id="memberSearch" onkeyup="applyFilters()" placeholder="Search members..." style="flex:2; padding:10px; border:1px solid #ddd; border-radius:8px;">
                    <select id="categoryFilter" onchange="applyFilters()" style="flex:1; padding:10px; border:1px solid #ddd; border-radius:8px;">
                        <option value="all">All Status</option>
                        <option value="eligible">✅ Eligible</option>
                        <option value="pending">🟡 Pending</option>
                        <option value="declined">🚫 Stopped</option>
                    </select>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                    <div style="background:#f0fff4; padding:15px; border-radius:12px; border-top:5px solid #28a745;">
                        <strong style="color:#22543d;">✅ ELIGIBLE (${data.low_risk})</strong>
                        <ul style="padding:0; margin-top:10px; max-height:450px; overflow-y:auto;">${buildList(data.eligible_members, '#28a745', 'eligible')}</ul>
                    </div>
                    <div style="background:#fffaf0; padding:15px; border-radius:12px; border-top:5px solid #ffc107;">
                        <strong style="color:#744210;">🟡 PENDING (${data.medium_risk})</strong>
                        <ul style="padding:0; margin-top:10px; max-height:450px; overflow-y:auto;">${buildList(data.pending_members, '#d69e2e', 'pending')}</ul>
                    </div>
                    <div style="background:#fff5f5; padding:15px; border-radius:12px; border-top:5px solid #dc3545;">
                        <strong style="color:#742a2a;">🚫 STOPPED (${data.high_risk})</strong>
                        <ul style="padding:0; margin-top:10px; max-height:450px; overflow-y:auto;">${buildList(data.declined_members, '#dc3545', 'declined')}</ul>
                    </div>
                </div>
            `;
        } else {
            alert("Upload Error: " + data.error);
        }
    } catch (err) {
        analysisEl.innerHTML = `<p style="color:red; padding:20px;">❌ System error processing file.</p>`;
    }
}