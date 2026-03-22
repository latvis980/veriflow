// src/components/ReportRenderer.tsx
import { useRef } from "react";
import type { AnalysisMode } from "@/lib/api";
import ComprehensiveReport from "./reports/ComprehensiveReport";
import KeyClaimsReport from "./reports/KeyClaimsReport";
import BiasReport from "./reports/BiasReport";
import DeceptionReport from "./reports/DeceptionReport";
import ManipulationReport from "./reports/ManipulationReport";
import LLMOutputReport from "./reports/LLMOutputReport";

type Props = {
  mode: AnalysisMode;
  data: any;
  onReset: () => void;
  sourceUrl?: string;
};

const modeLabel: Record<string, string> = {
  "comprehensive": "Comprehensive Analysis",
  "key-claims": "Key Claims",
  "bias-analysis": "Bias Analysis",
  "lie-detection": "Lie Detection",
  "manipulation": "Manipulation Detection",
  "llm-output": "LLM Output Verification",
};

const ReportRenderer = ({ mode, data, onReset, sourceUrl }: Props) => {
  const reportRef = useRef<HTMLDivElement>(null);

  const handlePrint = () => {
    const reportEl = reportRef.current;
    if (!reportEl) return;

    const printWindow = window.open("", "_blank", "width=900,height=700");
    if (!printWindow) return;

    // Grab all stylesheet links from the current page so Tailwind classes render correctly
    const styleLinks = Array.from(document.querySelectorAll('link[rel="stylesheet"]'))
      .map((el) => el.outerHTML)
      .join("\n");

    const styleBlocks = Array.from(document.querySelectorAll("style"))
      .map((el) => el.outerHTML)
      .join("\n");

    const timestamp = new Date().toLocaleString();
    const title = modeLabel[mode] ?? "VeriFlow Report";

    // Build source URL block for the printed PDF header
    const sourceUrlHtml = sourceUrl
      ? `<div class="source-url">
           <span class="source-label">Source:</span>
           <a href="${sourceUrl}" target="_blank" class="source-link">${sourceUrl}</a>
         </div>`
      : "";

    printWindow.document.write(`
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${title} - VeriFlow</title>
  ${styleLinks}
  ${styleBlocks}
  <style>
    /* ---- Print overrides: force white background + dark readable text ---- */
    @media print {
      @page {
        margin: 15mm 15mm 15mm 15mm;
        size: A4;
      }
      .no-print { display: none !important; }
      .rounded-xl, section, article { page-break-inside: avoid; }

      /* Reset all dark backgrounds to white */
      body, #report-content, #report-content * {
        background: white !important;
        background-color: white !important;
        color: #111 !important;
        border-color: #d1d5db !important;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
      }

      /* Headings */
      h1, h2, h3, h4, h5, h6 {
        color: #111 !important;
      }

      /* Muted / secondary text: dark gray instead of light gray */
      .text-muted-foreground,
      [class*="text-muted"] {
        color: #374151 !important;
      }

      /* Cards and badges */
      .bg-card, .bg-secondary, .bg-muted,
      [class*="bg-card"], [class*="bg-secondary"], [class*="bg-muted"] {
        background: #f3f4f6 !important;
        background-color: #f3f4f6 !important;
      }

      /* Score / status badges that use colored backgrounds */
      [class*="bg-score-"], [class*="bg-green"], [class*="bg-red"],
      [class*="bg-yellow"], [class*="bg-blue"] {
        filter: brightness(0.85) saturate(1.2);
      }

      /* Links: show URL in parentheses when printed */
      a[href]::after {
        content: " (" attr(href) ")";
        font-size: 0.75em;
        color: #374151 !important;
      }

      /* Source URL in the PDF header */
      .pdf-source-block {
        margin-bottom: 12mm;
        padding-bottom: 4mm;
        border-bottom: 1px solid #d1d5db;
      }
      .pdf-source-block .source-label {
        font-weight: 600;
        color: #111 !important;
        margin-right: 0.4em;
      }
      .pdf-source-block .source-link {
        color: #1d4ed8 !important;
        text-decoration: underline;
      }
      /* Suppress the "after" pseudo-element for the source link to avoid duplication */
      .pdf-source-block .source-link::after {
        content: "" !important;
      }
    }

    /* ---- Screen styles for the preview window ---- */
    body {
      font-family: system-ui, sans-serif;
      background: hsl(222 47% 11%);
      color: hsl(213 31% 91%);
      padding: 2rem;
      max-width: 900px;
      margin: 0 auto;
    }
    .print-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 1rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid hsl(216 34% 17%);
    }
    .print-header h1 {
      font-size: 1.25rem;
      font-weight: 700;
      margin: 0;
    }
    .print-header p {
      font-size: 0.75rem;
      color: hsl(215 20% 55%);
      margin: 0.25rem 0 0;
    }
    .print-actions {
      display: flex;
      gap: 0.5rem;
    }
    .btn {
      padding: 0.4rem 1rem;
      border-radius: 6px;
      font-size: 0.85rem;
      cursor: pointer;
      border: 1px solid hsl(216 34% 30%);
      background: hsl(216 34% 17%);
      color: hsl(213 31% 91%);
    }
    .btn-primary {
      background: hsl(221 83% 53%);
      border-color: hsl(221 83% 53%);
      color: white;
    }
    .btn:hover { opacity: 0.85; }

    /* Source URL block - screen view */
    .source-url {
      margin-top: 0.75rem;
      margin-bottom: 1.25rem;
      font-size: 0.8rem;
    }
    .source-label {
      color: hsl(215 20% 55%);
      margin-right: 0.3em;
    }
    .source-link {
      color: hsl(221 83% 70%);
      text-decoration: underline;
      word-break: break-all;
    }
  </style>
</head>
<body>
  <!-- Screen-only toolbar (hidden when printing) -->
  <div class="print-header no-print">
    <div>
      <h1>VeriFlow - ${title}</h1>
      <p>Generated: ${timestamp}</p>
    </div>
    <div class="print-actions">
      <button class="btn" onclick="window.close()">Close</button>
      <button class="btn btn-primary" onclick="window.print()">Save as PDF</button>
    </div>
  </div>

  <!-- Source URL: visible both on screen and in the printed PDF -->
  ${sourceUrl ? `
  <div class="pdf-source-block source-url">
    <span class="source-label">Source:</span>
    <a href="${sourceUrl}" target="_blank" class="source-link">${sourceUrl}</a>
  </div>` : ""}

  <div id="report-content">
    ${reportEl.innerHTML}
  </div>
</body>
</html>`);

    printWindow.document.close();
  };

  const renderReport = () => {
    switch (mode) {
      case "comprehensive": return <ComprehensiveReport data={data} />;
      case "key-claims": return <KeyClaimsReport data={data} />;
      case "bias-analysis": return <BiasReport data={data} />;
      case "lie-detection": return <DeceptionReport data={data} />;
      case "manipulation": return <ManipulationReport data={data} />;
      case "llm-output": return <LLMOutputReport data={data} />;
      default: return <pre className="text-sm overflow-auto">{JSON.stringify(data, null, 2)}</pre>;
    }
  };

  return (
    <div className="space-y-4">
      <div ref={reportRef}>
        {renderReport()}
      </div>

      <div className="flex justify-center gap-3 pt-2">
        <button
          onClick={onReset}
          className="rounded-lg border border-border px-5 py-2 text-base font-medium hover:bg-secondary transition-colors"
        >
          Analyze Another
        </button>
        <button
          onClick={handlePrint}
          className="rounded-lg border border-border px-5 py-2 text-base font-medium hover:bg-secondary transition-colors"
        >
          Save as PDF
        </button>
      </div>
    </div>
  );
};

export default ReportRenderer;