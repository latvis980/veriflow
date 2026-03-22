// src/components/reports/ManipulationReport.tsx
import { cn } from "@/lib/utils";
import { SessionInfo, getScoreColor } from "./shared";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

// Backend manipulation_findings item fields:
//   fact_id, fact_statement, truthfulness, truth_score,
//   manipulation_detected, manipulation_types, manipulation_severity,
//   what_was_omitted, how_it_serves_agenda, corrected_context, sources_used
type ManipulationFinding = {
  fact_id: string;
  fact_statement: string;
  truthfulness: string;
  truth_score: number;
  manipulation_detected: boolean;
  manipulation_types: string[];
  manipulation_severity: string;
  what_was_omitted: string;
  how_it_serves_agenda: string;
  corrected_context: string;
  sources_used: string[];
};

type Props = {
  data: {
    manipulation_score?: number;
    report?: {
      justification?: string;
      narrative_summary?: string;
      techniques_used?: string[];           // plain string array, NOT objects
      what_got_right?: string[];
      misleading_elements?: string[];
      recommendation?: string;
    };
    manipulation_findings?: ManipulationFinding[];
    article_summary?: {
      main_thesis?: string;
      political_lean?: string;
      detected_agenda?: string;
      opinion_fact_ratio?: string;
      emotional_tone?: string;
      target_audience?: string;
      summary?: string;
    };
    session_id?: string;
    processing_time?: number;
  };
};

const manipLevel = (score: number) => {
  if (score <= 3) return { label: "LOW MANIPULATION", color: "text-score-high" };
  if (score <= 6) return { label: "MODERATE MANIPULATION", color: "text-score-moderate" };
  return { label: "HIGH MANIPULATION", color: "text-score-low" };
};

const sevColor = (s: string) => {
  switch (s?.toLowerCase()) {
    case "low": return "bg-score-high/15 text-score-high";
    case "moderate":
    case "medium": return "bg-score-moderate/15 text-score-moderate";
    case "high": return "bg-score-low/15 text-score-low";
    default: return "bg-muted text-muted-foreground";
  }
};

const ManipulationReport = ({ data }: Props) => {
  const [showDetails, setShowDetails] = useState(false);
  const [showFindings, setShowFindings] = useState(false);
  const score = data.manipulation_score ?? 0;
  const level = manipLevel(score);
  const r = data.report;
  const findings = data.manipulation_findings?.filter(f => f.manipulation_detected) || [];

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-border bg-card p-5">
        <h3 className="text-xl font-display font-semibold mb-1">Manipulation Check</h3>
        <p className="text-sm text-muted-foreground mb-4">
          We look for facts taken out of context, cherry-picked data, and misleading framing.
        </p>

        <div className="flex items-center gap-3 mb-3">
          <span className={cn("text-3xl font-bold font-display", level.color)}>
            {score.toFixed(1)}
          </span>
          <span className="text-base text-muted-foreground">/10</span>
          <span className={cn("text-sm font-semibold uppercase", level.color)}>
            {level.label}
          </span>
        </div>

        {r?.narrative_summary && (
          <p className="text-base leading-relaxed mb-4">{r.narrative_summary}</p>
        )}

        {/* Techniques -- plain string array */}
        {r?.techniques_used && r.techniques_used.length > 0 && (
          <div className="space-y-2 mb-3">
            <h4 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
              Techniques Detected
            </h4>
            <div className="flex flex-wrap gap-2">
              {r.techniques_used.map((t, i) => (
                <span
                  key={i}
                  className="rounded-lg border border-border px-3 py-1.5 text-sm font-medium"
                >
                  {t}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Misleading elements */}
        {r?.misleading_elements && r.misleading_elements.length > 0 && (
          <div className="space-y-1 mb-3">
            <h4 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
              Key Misleading Elements
            </h4>
            {r.misleading_elements.map((el, i) => (
              <p key={i} className="text-sm text-muted-foreground pl-3 border-l-2 border-score-low">
                {el}
              </p>
            ))}
          </div>
        )}

        {/* What the article got right */}
        {r?.what_got_right && r.what_got_right.length > 0 && (
          <div className="space-y-1 mb-3">
            <h4 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
              What It Got Right
            </h4>
            {r.what_got_right.map((item, i) => (
              <p key={i} className="text-sm text-muted-foreground pl-3 border-l-2 border-score-high">
                {item}
              </p>
            ))}
          </div>
        )}

        {/* Recommendation */}
        {r?.recommendation && (
          <div className="mb-3 p-3 rounded-lg bg-secondary">
            <h4 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-1">
              Reader Recommendation
            </h4>
            <p className="text-sm text-muted-foreground">{r.recommendation}</p>
          </div>
        )}

        {/* Detailed manipulation findings (expandable) */}
        {findings.length > 0 && (
          <>
            <button
              onClick={() => setShowFindings(!showFindings)}
              className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              Detailed Findings ({findings.length}) {showFindings ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            </button>
            {showFindings && (
              <div className="mt-3 space-y-3">
                {findings.map((f, i) => (
                  <div key={f.fact_id || i} className="rounded-lg border border-border p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm font-semibold text-muted-foreground">#{f.fact_id}</span>
                      <span className={cn(
                        "rounded px-1.5 py-0.5 text-xs font-semibold uppercase",
                        sevColor(f.manipulation_severity)
                      )}>
                        {f.manipulation_severity}
                      </span>
                    </div>
                    <p className="text-base font-medium mb-1">{f.fact_statement}</p>
                    {f.manipulation_types?.length > 0 && (
                      <p className="text-sm text-muted-foreground mb-1">
                        Types: {f.manipulation_types.join(", ")}
                      </p>
                    )}
                    {f.what_was_omitted && (
                      <p className="text-sm text-muted-foreground">
                        <strong>Omitted:</strong> {f.what_was_omitted}
                      </p>
                    )}
                    {f.corrected_context && (
                      <p className="text-sm text-muted-foreground mt-1">
                        <strong>Context:</strong> {f.corrected_context}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {/* Article summary (expandable) */}
        {data.article_summary && (
          <>
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="flex items-center gap-1 mt-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              Article Summary {showDetails ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            </button>
            {showDetails && (
              <div className="mt-2 p-3 rounded-lg bg-secondary text-sm text-muted-foreground space-y-1">
                {data.article_summary.main_thesis && (
                  <p><strong>Thesis:</strong> {data.article_summary.main_thesis}</p>
                )}
                {data.article_summary.political_lean && (
                  <p><strong>Political Lean:</strong> {data.article_summary.political_lean}</p>
                )}
                {data.article_summary.detected_agenda && (
                  <p><strong>Detected Agenda:</strong> {data.article_summary.detected_agenda}</p>
                )}
                {data.article_summary.emotional_tone && (
                  <p><strong>Tone:</strong> {data.article_summary.emotional_tone}</p>
                )}
              </div>
            )}
          </>
        )}
      </div>
      <SessionInfo sessionId={data.session_id} processingTime={data.processing_time} />
    </div>
  );
};

export default ManipulationReport;