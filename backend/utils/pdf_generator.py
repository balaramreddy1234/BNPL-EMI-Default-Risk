import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import io
import os
import requests
from datetime import datetime

# ======================================================
# FONT SETUP — UTF-8 SUPPORT FOR ₹ SYMBOL
# ======================================================
def setup_fonts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    regular_font = os.path.join(base_dir, "DejaVuSans.ttf")
    bold_font = os.path.join(base_dir, "DejaVuSans-Bold.ttf")

    font_urls = {
        regular_font: "https://github.com/mhardcastle/DejaVu-Sans-TTF/raw/master/DejaVuSans.ttf",
        bold_font: "https://github.com/mhardcastle/DejaVu-Sans-TTF/raw/master/DejaVuSans-Bold.ttf"
    }

    for path, url in font_urls.items():
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            try:
                response = requests.get(url, timeout=15)
                with open(path, "wb") as f:
                    f.write(response.content)
            except: pass

    try:
        pdfmetrics.registerFont(TTFont("DejaVu", regular_font))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", bold_font))
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
        return "DejaVu", "DejaVu-Bold"
    except:
        return "Helvetica", "Helvetica-Bold"

MAIN_FONT, BOLD_FONT = setup_fonts()

# ======================================================
# PDF REPORT GENERATOR (OPTIMIZED LAYOUT & SPACING)
# ======================================================
def create_visual_report(data, prob, risk, report_path):
    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter
    curr = "₹" if MAIN_FONT == "DejaVu" else "Rs."

    # ---------------- DATA SANITIZATION ----------------
    try:
        income = float(str(data.get('income', 1)).replace(',', '')) or 1
        emi = float(str(data.get('emi', 0)).replace(',', ''))
        loan = float(str(data.get('loan', 0)).replace(',', ''))
        tenure = int(data.get('tenure', 0))
        on_time = int(data.get('ontime', 0))
        delays = int(data.get('delays', 0))
        credit = int(data.get('credit', 0))
    except:
        income, emi, loan, tenure, on_time, delays, credit = 1, 0, 0, 0, 0, 0, 0

    # ---------------- HEADER ----------------
    c.setFillColorRGB(0.17, 0.24, 0.31)
    c.rect(0, 730, width, 70, fill=1)
    c.setFillColor(colors.white)
    c.setFont(BOLD_FONT, 20)
    c.drawCentredString(width / 2, 755, "BNPL Risk Assessment Report")

    # ---------------- 1. EXECUTIVE SUMMARY ----------------
    c.setFillColor(colors.black)
    c.setFont(BOLD_FONT, 13)
    c.drawString(50, 710, "1. Executive Summary")
    c.line(50, 705, 562, 705)

    c.setFont(BOLD_FONT, 11)
    c.drawString(60, 685, f"Report Profile: {data.get('name', 'N/A')}")

    verdict_color = colors.red if "HIGH" in risk.upper() else colors.orange if "MEDIUM" in risk.upper() else colors.green
    c.setFillColor(verdict_color)
    c.roundRect(350, 675, 180, 25, 10, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont(BOLD_FONT, 10)
    c.drawCentredString(440, 682, f"Final Risk Verdict: {risk.upper()}")

    c.setFillColor(colors.black)
    c.setFont(BOLD_FONT, 10)
    c.drawString(60, 660, "Financial Snapshot")
    c.setFont(MAIN_FONT, 9)
    y_snapshot = 645
    for text in [f"• Monthly Income: {curr}{income:,.2f}", f"• Requested Loan: {curr}{loan:,.2f}", 
                 f"• Monthly EMI: {curr}{emi:,.2f}", f"• Loan Tenure: {tenure} Months"]:
        c.drawString(70, y_snapshot, text)
        y_snapshot -= 13

    c.drawString(360, 660, f"System Confidence: {round(prob * 100, 1)}%")
    c.drawString(360, 647, f"Assessment Date: {datetime.now().strftime('%d-%m-%Y')}")

    # ---------------- 2. ASSESSMENT PARAMETERS ----------------
    c.setFont(BOLD_FONT, 13)
    c.drawString(50, 585, "2. Assessment Parameters")
    c.line(50, 580, 562, 580)

    c.setFillColorRGB(0.96, 0.97, 0.98)
    c.roundRect(55, 505, 500, 70, 10, fill=1, stroke=0)

    c.setFillColor(colors.black)
    c.setFont(BOLD_FONT, 10)
    c.drawString(75, 555, "Debt-to-Income Analysis")
    c.drawString(320, 555, "Behavioral Metrics")

    c.setFont(MAIN_FONT, 9)
    emi_ratio = (emi / income) * 100
    c.drawString(75, 540, f"• EMI Ratio: {round(emi_ratio, 1)}% ({'Healthy' if emi_ratio <= 40 else 'High'})")
    c.drawString(75, 527, f"• Credit Score: {credit}")
    c.drawString(320, 540, f"• Reliability: {on_time} On-time payments")
    c.drawString(320, 527, f"• Delays: {delays} Total delays")

    # ---------------- 3. STRATEGIC PROFESSIONAL ANALYSIS ----------------
    c.setFont(BOLD_FONT, 13)
    c.drawString(50, 480, "3. Strategic Professional Analysis")
    c.line(50, 475, 562, 475)

    def draw_chart(fig, x, y, w, h):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches='tight')
        buf.seek(0)
        c.drawImage(ImageReader(buf), x, y, w, h)
        plt.close(fig)

    # Probability & EMI Graphs (Top Visualization Row)
    fig1, ax1 = plt.subplots(figsize=(4, 2.5))
    ax1.barh(["Risk Level"], [prob], color='red' if prob > 0.5 else 'green')
    ax1.set_xlim(0, 1)
    ax1.set_title("Probability of Default", fontsize=10, fontweight='bold')
    draw_chart(fig1, 310, 340, 240, 130)

    fig2, ax2 = plt.subplots(figsize=(4, 2.5))
    ax2.bar(["User Ratio"], [emi_ratio], color='skyblue', width=0.4)
    ax2.axhline(40, color='red', linestyle="--")
    ax2.set_ylim(0, 100)
    ax2.set_title("EMI to Income Ratio (%)", fontsize=10, fontweight='bold')
    draw_chart(fig2, 60, 340, 240, 130)

    # Pie Chart (Shifted Down for focus)
    fig3, ax3 = plt.subplots(figsize=(4, 4))
    # Safety check for pie chart data
    total_pmts = on_time + delays
    p_data = [on_time, delays] if total_pmts > 0 else [1, 0]
    p_labels = ["On-time", "Delays"] if total_pmts > 0 else ["No History", ""]
    
    ax3.pie(p_data, labels=p_labels, colors=['#2ecc71', '#e74c3c'], autopct="%1.1f%%", startangle=140, textprops={'fontsize': 9})
    ax3.set_title("Historical Payment Reliability", fontsize=10, fontweight='bold')
    draw_chart(fig3, 185, 160, 240, 180) 

    # ---------------- 4. PROFESSIONAL SUGGESTIONS ----------------
    # Positioned clearly above the footer (starts at y=50)
    c.setFont(BOLD_FONT, 13)
    c.drawString(50, 155, "4. Professional Suggestions") 
    c.line(50, 150, 562, 150)

    suggestions = []
    if emi > (income * 0.4):
        suggestions.append("• Income vs EMI: Monthly burden exceeds 40%; suggest reducing EMI or increasing income.")
    else:
        suggestions.append("• Income vs EMI: Your monthly EMI commitment is within a healthy, affordable range.")

    if delays > on_time:
        suggestions.append("• Repayment: Delays exceed on-time payments; suggest stricter discipline or auto-debit.")
    else:
        suggestions.append("• Repayment: You have a solid record of consistent and reliable repayment behavior.")

    if credit < 650:
        suggestions.append("• Credit Profile: Score below 650; focus on clearing outstanding dues to improve profile.")
    elif 650 <= credit <= 750:
        suggestions.append("• Credit Profile: Score is good; maintain current habits to reach an 'Excellent' status.")
    else:
        suggestions.append("• Credit Profile: Strong credit profile; continue your responsible financial management.")

    if loan > (income * 5):
        suggestions.append("• Debt Capacity: Total loan exceeds 5x income; suggest avoiding further debt restructuring.")
    else:
        suggestions.append("• Debt Capacity: Requested loan amount aligns well with your monthly earning capacity.")

    c.setFont(MAIN_FONT, 9)
    y_sugg = 135 
    for s in suggestions:
        c.drawString(65, y_sugg, s)
        y_sugg -= 14

    # ---------------- FOOTER ----------------
    c.setFillColorRGB(0.17, 0.24, 0.31)
    c.rect(0, 0, width, 50, fill=1)
    c.setFillColor(colors.white)
    c.setFont(MAIN_FONT, 8)
    c.drawCentredString(width / 2, 25, "Strategic Financial Suggestion Engine | Secured System Decision Report")
    c.drawCentredString(width / 2, 12, "Hyderabad, Telangana, India | Automated Financial Review")

    c.showPage()
    c.save()