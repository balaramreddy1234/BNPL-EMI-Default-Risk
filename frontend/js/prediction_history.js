/**
 * BNPL Strategic Decision Engine - History Management
 * Updated Features: Batch Drill-down, Individual PDF access, and Bulk Delete logic.
 */

document.addEventListener("DOMContentLoaded", () => {
    fetchHistory();
});

/**
 * 1. Fetch History and Render Rows [Outcome 4: Transparency]
 */
function fetchHistory() {
    const tableBody = document.getElementById("historyBody");
    const noData = document.getElementById("noData");
    const enableDeleteBtn = document.getElementById("enableDeleteBtn");

    fetch("/user/prediction-history")
        .then(res => res.json())
        .then(data => {
            if (!data || data.length === 0) {
                noData.style.display = "block";
                if (enableDeleteBtn) enableDeleteBtn.style.display = "none";
                tableBody.innerHTML = "";
                return;
            }
            
            noData.style.display = "none";
            if (enableDeleteBtn) enableDeleteBtn.style.display = "block";

            tableBody.innerHTML = data.map(item => {
                const isManual = item.type === "manual";
                const recordId = item.id;
                
                // Styling based on risk level for transparency
                const riskColor = item.risk === "HIGH RISK" ? "#dc3545" : (item.risk === "MEDIUM RISK" ? "#ff9800" : "#28a745");
                
                return `
                    <tr id="row-${recordId}" style="border-bottom: 1px solid #edf2f7; transition: background 0.2s;">
                        <td class="checkbox-col" style="text-align:center; padding: 15px;">
                            <input type="checkbox" class="delete-checkbox" value="${recordId}" onchange="updateBtnState()" style="transform: scale(1.2); cursor: pointer;">
                        </td>
                        <td style="padding: 15px;">${item.created_at}</td>
                        <td style="padding: 15px;">
                            <span style="padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; background: ${isManual ? '#ebf8ff' : '#faf5ff'}; color: ${isManual ? '#3182ce' : '#805ad5'};">
                                ${isManual ? 'MANUAL' : 'BATCH'}
                            </span>
                        </td>
                        <td style="padding: 15px; font-weight: 500;">
                            ${isManual ? `₹${Number(item.income).toLocaleString('en-IN')}` : item.filename}
                        </td>
                        <td style="padding: 15px; font-weight: bold; color: ${riskColor};">
                            ${isManual ? item.risk : `H: ${item.high} | L: ${item.low}`}
                        </td>
                        <td style="padding: 15px;">
                            ${isManual ? (item.probability * 100).toFixed(1) + '%' : `<small style="color:#718096;">Total:</small> ${item.total}`}
                        </td>
                        <td style="padding: 15px;">
                            ${isManual 
                                ? `<a href="/download-report/${item.report}" target="_blank" 
                                      style="text-decoration: none; color: white; background: #2c3e50; padding: 6px 12px; border-radius: 6px; font-size: 0.85em; font-weight: bold;">📥 PDF</a>` 
                                : `<button onclick="viewBatchDetail(${recordId})" 
                                           style="border: 1px solid #2c3e50; color: #2c3e50; background: transparent; padding: 5px 12px; border-radius: 6px; font-size: 0.85em; font-weight: bold; cursor: pointer; transition: 0.2s;">
                                   📊 View Dataset
                                   </button>`
                            }
                        </td>
                    </tr>`;
            }).join("");
        })
        .catch(err => {
            console.error("History Load Error:", err);
            noData.innerHTML = "<p style='color:red;'>Failed to load history.</p>";
            noData.style.display = "block";
        });
}

/**
 * 2. New: Batch Detail View [Outcome 3: Better Decision Making]
 */
function viewBatchDetail(batchId) {
    // Redirects to the Upload module with the batch context to view individual members
    // This allows you to use the existing 'eligible-member' view features
    window.location.href = `/pages/upload.html?batch_id=${batchId}`;
}

/**
 * 3. Mode Switching Logic
 */
function enterDeleteMode() {
    const table = document.getElementById("historyTable");
    if (!table) return;
    table.classList.remove("delete-mode-off");
    table.classList.add("delete-mode-on");
    document.getElementById("enableDeleteBtn").style.display = "none";
    document.getElementById("bulkDeleteBtn").style.display = "inline-block";
    document.getElementById("cancelDeleteBtn").style.display = "inline-block";
    updateBtnState();
}

function exitDeleteMode() {
    const table = document.getElementById("historyTable");
    if (!table) return;
    table.classList.remove("delete-mode-on");
    table.classList.add("delete-mode-off");
    const selectAll = document.getElementById("selectAll");
    if (selectAll) selectAll.checked = false;
    document.querySelectorAll('.delete-checkbox').forEach(cb => cb.checked = false);
    document.getElementById("enableDeleteBtn").style.display = "inline-block";
    document.getElementById("bulkDeleteBtn").style.display = "none";
    document.getElementById("cancelDeleteBtn").style.display = "none";
}

/**
 * 4. Selection Helpers
 */
function toggleAll(source) {
    const checkboxes = document.querySelectorAll('.delete-checkbox');
    checkboxes.forEach(cb => cb.checked = source.checked);
    updateBtnState();
}

function updateBtnState() {
    const count = document.querySelectorAll('.delete-checkbox:checked').length;
    const btn = document.getElementById("bulkDeleteBtn");
    if (!btn) return;
    btn.innerText = count > 0 ? `🗑️ Confirm Delete (${count})` : "🗑️ Confirm Delete";
    if (count > 0) {
        btn.style.opacity = "1";
        btn.style.pointerEvents = "auto";
    } else {
        btn.style.opacity = "0.5";
        btn.style.pointerEvents = "none";
    }
}

/**
 * 5. Execute Deletion [Outcome 5: Scalability]
 */
async function deleteSelected() {
    const selected = Array.from(document.querySelectorAll('.delete-checkbox:checked')).map(cb => cb.value);
    if (!selected.length || !confirm(`Permanently delete ${selected.length} records?`)) return;

    const btn = document.getElementById("bulkDeleteBtn");
    btn.innerText = "⏳ Deleting...";
    btn.disabled = true;

    try {
        const res = await fetch("/user/delete-history", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ids: selected })
        });
        const result = await res.json();
        if (result.success) {
            selected.forEach(id => {
                const row = document.getElementById(`row-${id}`);
                if (row) row.remove();
            });
            exitDeleteMode();
            if (document.querySelectorAll('.delete-checkbox').length === 0) fetchHistory();
        } else {
            alert("Error: " + (result.error || "Could not delete records"));
        }
    } catch (err) {
        console.error("Delete Failed:", err);
        alert("Server error. Please try again later.");
    } finally {
        btn.disabled = false;
        btn.innerText = "🗑️ Confirm Delete";
    }
}