// src/components/reports/ComprehensiveReport.tsx
import { cn } from "@/lib/utils";
import { ScoreBadge, SessionInfo, getScoreBg, getScoreLabel } from "./shared";
import KeyClaimsReport from "./KeyClaimsReport";
import BiasReport from "./BiasReport";
import DeceptionReport from "./DeceptionReport";
import ManipulationReport from "./ManipulationReport";
import LLMOutputReport from "./LLMOutputReport";
import { useState, useMemo } from "react";
import { ChevronDown, ChevronUp, Copy, Check } from "lucide-react";
import Markdown from "react-markdown";

type Props = {
  data: {
    content_classification?: Record<string, any>;
    source_verification?: Record<string, any>;
    mode_routing?: { selected_modes?: string[]; reasoning?: string };
    mode_reports?: Record<string, any>;
    mode_errors?: Record<string, string>;
    synthesis_report?: {
      overall_score?: number;
      overall_rating?: string;
      overall_credibility_score?: number;  // alternate field name
      overall_credibility_rating?: string; // alternate field name
      confidence?: number;
      summary?: string;
      key_concerns?: string[];
      positive_indicators?: string[];
      recommendations?: string[];
      modes_analyzed?: string[];
    };
    session_id?: string;
    processing_time?: number;
  };
};

const tierColors: Record<number, string> = {
  1: "bg-score-high text-accent-foreground",
  2: "bg-tier-2 text-accent-foreground",
  3: "bg-score-moderate text-foreground",
  4: "bg-score-elevated text-accent-foreground",
  5: "bg-score-low text-accent-foreground",
};

// Backend mode IDs -> display names
const modeLabels: Record<string, string> = {
  key_claims_analysis: "Key Claims Verification",
  bias_analysis: "Bias Analysis",
  manipulation_detection: "Manipulation Detection",
  lie_detection: "Deception Detection",
  llm_output_verification: "LLM Output Verification",
};

const ComprehensiveReport = ({ data }: Props) => {
  // Auto-expand all mode sections by default
  const modeKeys = useMemo(
    () => Object.keys(data.mode_reports || {}),
    [data.mode_reports]
  );
  const [expandedModes, setExpandedModes] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    modeKeys.forEach(k => { initial[k] = true; });
    return initial;
  });

  const synth = data.synthesis_report;
  // Handle both field name variants
  const score = synth?.overall_score ?? synth?.overall_credibility_score ?? 0;
  const rating = synth?.overall_rating ?? synth?.overall_credibility_rating;

  const [copied, setCopied] = useState(false);

  const copyAssessment = () => {
    if (!synth) return;
    const lines: string[] = [];

    lines.push(`OVERALL ASSESSMENT`);
    lines.push(`Score: ${score}/100 -- ${rating || getScoreLabel(score)}`);
    if (synth.confidence != null) lines.push(`Confidence: ${Math.round(synth.confidence)}%`);
    lines.push("");

    if (synth.summary) lines.push(synth.summary, "");

    if (synth.key_concerns?.length) {
      lines.push("KEY CONCERNS");
      synth.key_concerns.forEach(c => lines.push(`- ${c}`));
      lines.push("");
    }

    if (synth.positive_indicators?.length) {
      lines.push("POSITIVE INDICATORS");
      synth.positive_indicators.forEach(p => lines.push(`- ${p}`));
      lines.push("");
    }

    if (synth.recommendations?.length) {
      lines.push("RECOMMENDATIONS");
      synth.recommendations.forEach((r, i) => lines.push(`${i + 1}. ${r}`));
    }

    navigator.clipboard.writeText(lines.join("\n")).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const toggleMode = (mode: string) => {
    setExpandedModes(prev => ({ ...prev, [mode]: !prev[mode] }));
  };

  // Map backend mode IDs to the correct sub-report component
  const renderSubReport = (modeKey: string, modeData: any) => {
    switch (modeKey) {
      case "key_claims_analysis":
        return <KeyClaimsReport data={modeData} />;
      case "bias_analysis":
        return <BiasReport data={modeData} />;
      case "lie_detection":
        return <DeceptionReport data={modeData} />;
      case "manipulation_detection":
        return <ManipulationReport data={modeData} />;
      case "llm_output_verification":
        return <LLMOutputReport data={modeData} />;
      default:
        return <pre className="text-sm overflow-auto">{JSON.stringify(modeData, null, 2)}</pre>;
    }
  };

  return (
    <div className="space-y-4">
      {/* Metadata Panel */}
      <div className="rounded-xl border border-border bg-card p-5">
        <h3 className="text-xl font-display font-semibold mb-3">Comprehensive Analysis</h3>

        <div className="flex flex-wrap gap-2 mb-3">
          {data.content_classification?.content_type && (
            <div className="rounded-lg bg-secondary px-3 py-1.5 text-sm">
              <span className="text-muted-foreground">Type:</span>{" "}
              <span className="font-medium capitalize">{data.content_classification.content_type.replace(/_/g, " ")}</span>
            </div>
          )}
          {data.content_classification?.realm && (
            <div className="rounded-lg bg-secondary px-3 py-1.5 text-sm">
              <span className="text-muted-foreground">Realm:</span>{" "}
              <span className="font-medium capitalize">{data.content_classification.realm}</span>
            </div>
          )}
          {data.source_verification?.credibility_tier && (
            <div className={cn("rounded-lg px-3 py-1.5 text-sm font-medium", tierColors[data.source_verification.credibility_tier] || "bg-muted")}>
              Tier {data.source_verification.credibility_tier}
              {data.source_verification.publication_name && ` -- ${data.source_verification.publication_name}`}
            </div>
          )}
          {data.source_verification?.bias_rating && (
            <div className="rounded-lg bg-secondary px-3 py-1.5 text-sm">
              <span className="text-muted-foreground">Bias:</span>{" "}
              <span className="font-medium">{data.source_verification.bias_rating}</span>
            </div>
          )}
        </div>

        {/* Mode routing info */}
        {data.mode_routing?.selected_modes && (
          <p className="text-sm text-muted-foreground">
            Modes selected: {data.mode_routing.selected_modes.map(m => modeLabels[m] || m).join(", ")}
          </p>
        )}
      </div>

      {/* ===== Overall Assessment — moved to top ===== */}
      {synth && (
        <div className="rounded-xl border border-border bg-card overflow-hidden">
          {/* Header bar with score, rating, confidence, and copy button */}
          <div className="flex items-center gap-5 p-5 border-b border-border bg-secondary/30">
            <ScoreBadge score={score} label={rating || getScoreLabel(score)} />

            <div className="flex-1 min-w-0">
              <h3 className="text-xl font-display font-semibold">Overall Assessment</h3>
              {synth.confidence != null && (
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-sm text-muted-foreground">Confidence:</span>
                  <div className="flex-1 max-w-[120px] h-1.5 rounded-full bg-secondary overflow-hidden">
                    <div
                      className={cn("h-full rounded-full", getScoreBg(synth.confidence))}
                      style={{ width: `${synth.confidence}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium">{Math.round(synth.confidence)}%</span>
                </div>
              )}
            </div>

            <button
              onClick={copyAssessment}
              className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors shrink-0"
              title="Copy overall assessment"
            >
              {copied ? <Check size={13} /> : <Copy size={13} />}
              {copied ? "Copied" : "Copy"}
            </button>
          </div>

          {/* Markdown-rendered summary */}
          {synth.summary && (
            <div className="p-5 prose prose-sm dark:prose-invert max-w-none prose-headings:font-display prose-headings:text-foreground prose-h2:text-lg prose-h2:mt-5 prose-h2:mb-2 prose-h3:text-base prose-h3:mt-4 prose-h3:mb-1 prose-p:text-foreground/90 prose-p:leading-relaxed prose-li:text-foreground/90 prose-strong:text-foreground prose-blockquote:border-primary/40 prose-blockquote:text-foreground/80">
              <Markdown>{synth.summary}</Markdown>
            </div>
          )}

          {/* Key Concerns / Positive Indicators / Recommendations */}
          {(synth.key_concerns?.length || synth.positive_indicators?.length || synth.recommendations?.length) ? (
            <div className="px-5 pb-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {synth.key_concerns?.length ? (
                <div className="rounded-lg border border-score-low/20 bg-score-low/5 p-4">
                  <h4 className="text-sm font-semibold uppercase tracking-wide text-score-low mb-2">Key Concerns</h4>
                  <ul className="space-y-1.5">
                    {synth.key_concerns.map((c, i) => (
                      <li key={i} className="text-sm text-muted-foreground pl-3 border-l-2 border-score-low/40">{c}</li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {synth.positive_indicators?.length ? (
                <div className="rounded-lg border border-score-high/20 bg-score-high/5 p-4">
                  <h4 className="text-sm font-semibold uppercase tracking-wide text-score-high mb-2">Positive Indicators</h4>
                  <ul className="space-y-1.5">
                    {synth.positive_indicators.map((p, i) => (
                      <li key={i} className="text-sm text-muted-foreground pl-3 border-l-2 border-score-high/40">{p}</li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {synth.recommendations?.length ? (
                <div className="rounded-lg border border-primary/20 bg-primary/5 p-4">
                  <h4 className="text-sm font-semibold uppercase tracking-wide text-primary mb-2">Recommendations</h4>
                  <ol className="space-y-1.5">
                    {synth.recommendations.map((r, i) => (
                      <li key={i} className="text-sm text-muted-foreground pl-3 border-l-2 border-primary/40">{r}</li>
                    ))}
                  </ol>
                </div>
              ) : null}
            </div>
          ) : null}
        </div>
      )}

      {/* Mode Reports -- auto-expanded */}
      {data.mode_reports && Object.entries(data.mode_reports).map(([key, value]) => (
        <div key={key} className="rounded-xl border border-border bg-card overflow-hidden">
          <button
            onClick={() => toggleMode(key)}
            className="w-full flex items-center justify-between p-4 text-base font-medium hover:bg-secondary/50 transition-colors"
          >
            <span>{modeLabels[key] || key.replace(/_/g, " ")}</span>
            {expandedModes[key] ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
          {expandedModes[key] && (
            <div className="px-4 pb-4">
              {renderSubReport(key, value)}
            </div>
          )}
        </div>
      ))}

      {/* Mode Errors */}
      {data.mode_errors && Object.keys(data.mode_errors).length > 0 && (
        <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-4">
          <h4 className="text-sm font-semibold uppercase tracking-wide text-destructive mb-2">
            Failed Modes
          </h4>
          {Object.entries(data.mode_errors).map(([key, err]) => (
            <p key={key} className="text-sm text-muted-foreground">
              {modeLabels[key] || key}: {err}
            </p>
          ))}
        </div>
      )}

      <SessionInfo sessionId={data.session_id} processingTime={data.processing_time} />
    </div>
  );
};

export default ComprehensiveReport;