/**
 * BNPL Strategic Engine - EMI History Management
 * Optimized for Telangana (IST) Time Display
 */

document.addEventListener("DOMContentLoaded", () => {
    const tableBody = document.getElementById("historyBody");
    const noData = document.getElementById("noData");
    const bulkDeleteBtn = document.getElementById("bulkDeleteBtn");

    /* ==========================================
       1. FORMAT DATE → IST (HYDERABAD)
       Converts UTC to: 2026-02-08 05:13:43 PM
       ========================================== */
    const formatToIST = (dateString) => {
        if (!dateString) return "N/A";
        try {
            const date = new Date(dateString);

            const formatter = new Intl.DateTimeFormat("en-IN", {
                timeZone: "Asia/Kolkata",
                year: "numeric",
                month: "2-digit",
                day: "2-digit",
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
                hour12: true
            });

            const parts = formatter.formatToParts(date).reduce((acc, part) => {
                acc[part.type] = part.value;
                return acc;
            }, {});

            // Construct: YYYY-MM-DD hh:mm:ss AM/PM
            const dayPeriod = parts.dayPeriod ? parts.dayPeriod.toUpperCase() : "";
            
            return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}:${parts.second} ${dayPeriod}`;

        } catch (err) {
            return dateString;
        }
    };

    /* ==========================================
       2. FETCH & RENDER HISTORY
       ========================================== */
    fetch("/user/prediction-history")
        .then(res => res.json())
        .then(data => {
            if (!Array.isArray(data) || data.length === 0) {
                if (noData) noData.style.display = "block";
                return;
            }

            if (noData) noData.style.display = "none";
            if (bulkDeleteBtn) bulkDeleteBtn.style.display = "inline-block";

            tableBody.innerHTML = data.map(item => {
                const recordId = item.id;
                // CRITICAL: Convert the raw UTC string to IST here
                const istTime = formatToIST(item.created_at);
                const isManual = item.type === "manual";
                const riskClass = item.risk && item.risk.toUpperCase().includes("HIGH") ? "risk-high" : "risk-low";

                return `
                    <tr id="row-${recordId}">
                        <td><input type="checkbox" class="delete-checkbox" value="${recordId}" onchange="checkSelection()"></td>
                        <td style="white-space:nowrap; font-family: monospace;">${istTime}</td>
                        <td><span class="source-badge ${isManual ? '' : 'batch'}">${isManual ? 'Manual' : 'Batch'}</span></td>
                        <td class="font-bold">
                            ${isManual ? `₹${Number(item.income || 0).toLocaleString("en-IN")}` : (item.filename || "N/A")}
                        </td>
                        <td><span class="risk-badge ${riskClass}">
                            ${isManual ? (item.risk || "N/A") : `H: ${item.high} | L: ${item.low}`}
                        </span></td>
                        <td>${isManual ? (Number(item.probability || 0) * 100).toFixed(1) + '%' : `Total: ${item.total}`}</td>
                        <td>
                            ${isManual 
                                ? `<a href="/download-report/${item.report}" target="_blank" class="action-btn">PDF</a>`
                                : `<button class="action-btn details" onclick="alert('Batch ID: ${recordId}')">Details</button>`
                            }
                        </td>
                    </tr>`;
            }).join("");
        })
        .catch(err => console.error("Fetch failed:", err));
});

function checkSelection() {
    const selectedCount = document.querySelectorAll(".delete-checkbox:checked").length;
    const btn = document.getElementById("bulkDeleteBtn");
    if (!btn) return;
    btn.innerText = selectedCount > 0 ? `🗑️ Delete Selected (${selectedCount})` : "🗑️ Delete Selected";
    btn.style.opacity = selectedCount > 0 ? "1" : "0.6";
}

async function deleteSelected() {
    const ids = Array.from(document.querySelectorAll(".delete-checkbox:checked")).map(cb => cb.value);
    if (ids.length === 0 || !confirm(`Delete ${ids.length} records?`)) return;
    try {
        const res = await fetch("/user/delete-history", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ids })
        });
        if ((await res.json()).success) {
            ids.forEach(id => document.getElementById(`row-${id}`)?.remove());
            if (document.querySelectorAll(".delete-checkbox").length === 0) location.reload();
        }
    } catch (err) { console.error(err); }
}