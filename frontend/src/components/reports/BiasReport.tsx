// src/components/reports/BiasReport.tsx
import { useState } from "react";
import { cn } from "@/lib/utils";
import { SessionInfo } from "./shared";
import { ChevronDown, ChevronUp } from "lucide-react";

// Backend BiasInstance: { type, direction, severity (1-10), evidence, techniques[] }
type BiasIndicator = {
  type?: string;
  direction?: string;
  severity?: number;
  evidence?: string;
  techniques?: string[];
  // Legacy field names (fallback)
  description?: string;
  examples?: string[];
};

// Backend BiasAnalysisResult (.model_dump()) fields
type BiasModel = {
  model_name?: string;
  // Actual backend field names
  overall_bias_score?: number;       // 0-10 scale
  primary_bias_direction?: string;
  biases_detected?: BiasIndicator[];
  balanced_aspects?: string[];
  missing_perspectives?: string[];
  recommendations?: string[];
  reasoning?: string;
  // Legacy field names (fallback)
  bias_rating?: string;
  bias_score?: number;
  summary?: string;
  bias_indicators?: BiasIndicator[];
  confidence?: string;
  language_analysis?: string;
  source_selection_analysis?: string;
  framing_analysis?: string;
};

// Backend returns flat fields inside analysis, NOT nested under consensus.*
type Props = {
  data: {
    analysis?: {
      gpt_analysis?: BiasModel;
      claude_analysis?: BiasModel;
      combined_report?: Record<string, any>;
      // Flat consensus fields
      consensus_direction?: string;
      consensus_bias_score?: number;   // 0-10 scale
      final_assessment?: string;
      confidence?: number;             // 0-1 scale
      areas_of_agreement?: string[];
      areas_of_disagreement?: string[];
      gpt_unique_findings?: string[];
      claude_unique_findings?: string[];
      recommendations?: string[];
      publication_bias_context?: string;
    };
    session_id?: string;
    processing_time?: number;
  };
};

const tabs = ["Consensus", "GPT-4", "Claude"] as const;

const biasColor = (rating?: string) => {
  if (!rating) return "text-muted-foreground";
  const r = rating.toUpperCase();
  if (r.includes("LEFT")) return "text-blue-600";
  if (r.includes("RIGHT")) return "text-red-600";
  if (r.includes("CENTER") || r.includes("LEAST")) return "text-score-high";
  return "text-muted-foreground";
};

const severityColor = (severity: number) => {
  if (severity >= 7) return "border-score-low";
  if (severity >= 4) return "border-score-moderate";
  return "border-score-high";
};

const ModelView = ({ model }: { model?: BiasModel }) => {
  const [showDetails, setShowDetails] = useState(false);

  if (!model) return <p className="text-base text-muted-foreground">No analysis available.</p>;

  // Resolve field names (backend actual -> legacy fallback)
  const direction = model.primary_bias_direction || model.bias_rating || "Unknown";
  const score = model.overall_bias_score ?? model.bias_score;
  const assessment = model.reasoning || model.summary || "";
  const indicators = model.biases_detected || model.bias_indicators || [];

  return (
    <div className="space-y-3">
      {/* Direction + Score */}
      <div className="flex items-center gap-3">
        <span className={cn("text-xl font-bold font-display capitalize", biasColor(direction))}>
          {direction}
        </span>
        {score != null && (
          <span className="text-base text-muted-foreground">{score.toFixed?.(1) ?? score}/10</span>
        )}
      </div>

      {/* Assessment / Reasoning */}
      {assessment && <p className="text-base leading-relaxed">{assessment}</p>}

      {/* Bias Indicators */}
      {indicators.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
            Bias Indicators ({indicators.length})
          </h4>
          {indicators.map((ind, i) => (
            <div key={i} className={cn("rounded-lg bg-secondary p-3 text-base border-l-3", severityColor(ind.severity || 0))}>
              <div className="flex items-center gap-2 mb-1">
                <span className="font-medium capitalize">{ind.type || "Bias"}</span>
                {ind.severity != null && (
                  <span className="text-sm text-muted-foreground">{ind.severity}/10</span>
                )}
                {ind.direction && (
                  <span className={cn("text-sm font-medium capitalize", biasColor(ind.direction))}>
                    {ind.direction}
                  </span>
                )}
              </div>
              <p className="text-sm text-muted-foreground">
                {ind.evidence || ind.description || ""}
              </p>
              {ind.techniques && ind.techniques.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-1.5">
                  {ind.techniques.map((t, j) => (
                    <span key={j} className="rounded bg-background px-1.5 py-0.5 text-xs text-muted-foreground">
                      {t}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Balanced Aspects + Missing Perspectives (collapsible) */}
      {(model.balanced_aspects?.length || model.missing_perspectives?.length) ? (
        <>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            Full Analysis {showDetails ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </button>
          {showDetails && (
            <div className="space-y-3 mt-1">
              {model.balanced_aspects && model.balanced_aspects.length > 0 && (
                <div>
                  <h4 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                    What It Does Well
                  </h4>
                  {model.balanced_aspects.map((item, i) => (
                    <p key={i} className="text-sm text-muted-foreground pl-3 border-l-2 border-score-high mb-1">
                      {item}
                    </p>
                  ))}
                </div>
              )}
              {model.missing_perspectives && model.missing_perspectives.length > 0 && (
                <div>
                  <h4 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                    Missing Perspectives
                  </h4>
                  {model.missing_perspectives.map((item, i) => (
                    <p key={i} className="text-sm text-muted-foreground pl-3 border-l-2 border-score-moderate mb-1">
                      {item}
                    </p>
                  ))}
                </div>
              )}
              {model.recommendations && model.recommendations.length > 0 && (
                <div>
                  <h4 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                    Recommendations
                  </h4>
                  {model.recommendations.map((item, i) => (
                    <p key={i} className="text-sm text-muted-foreground pl-3 border-l-2 border-border mb-1">
                      {item}
                    </p>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      ) : null}
    </div>
  );
};

const BiasReport = ({ data }: Props) => {
  const [tab, setTab] = useState<typeof tabs[number]>("Consensus");
  const [showComparison, setShowComparison] = useState(false);
  const a = data.analysis;

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-border bg-card p-5">
        <h3 className="text-xl font-display font-semibold mb-1">Bias Analysis</h3>
        <p className="text-sm text-muted-foreground mb-4">
          We detect political slant, loaded language, and one-sided framing.
        </p>

        <div className="flex gap-1 mb-4 rounded-lg bg-secondary p-1">
          {tabs.map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={cn(
                "flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                tab === t ? "bg-card shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"
              )}
            >
              {t}
            </button>
          ))}
        </div>

        {tab === "GPT-4" && <ModelView model={a?.gpt_analysis} />}
        {tab === "Claude" && <ModelView model={a?.claude_analysis} />}
        {tab === "Consensus" && (
          <div className="space-y-3">
            {/* Direction + Score */}
            <div className="flex items-center gap-3">
              <span className={cn("text-xl font-bold font-display capitalize", biasColor(a?.consensus_direction))}>
                {a?.consensus_direction || "N/A"}
              </span>
              {a?.consensus_bias_score != null && (
                <span className="text-base text-muted-foreground">
                  {typeof a.consensus_bias_score === 'number' ? a.consensus_bias_score.toFixed(1) : a.consensus_bias_score}/10
                </span>
              )}
              {a?.confidence != null && (
                <span className="text-sm text-muted-foreground">
                  ({Math.round((a.confidence > 1 ? a.confidence : a.confidence * 100))}% confidence)
                </span>
              )}
            </div>

            {/* Final assessment */}
            {a?.final_assessment && <p className="text-base leading-relaxed">{a.final_assessment}</p>}

            {/* Publication bias context */}
            {a?.publication_bias_context && (() => {
              // Strip legacy header lines so only the clean sentence is shown.
              const cleaned = a.publication_bias_context
                .split("\n")
                .map((line: string) => line.trim())
                .filter((line: string) =>
                  line.length > 0 &&
                  !/^(PUBLICATION CONTEXT|PUBLICATION:|MEDIA BIAS|MBFC|NOTE:|={3,})/i.test(line)
                )
                .join(" ");
              if (!cleaned) return null;
              return (
                <div className="rounded-lg bg-secondary/60 p-3">
                  <h4 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                    Publication Context
                  </h4>
                  <p className="text-sm text-muted-foreground leading-relaxed">{cleaned}</p>
                </div>
              );
            })()}

            {/* Model comparison toggle */}
            {(a?.areas_of_agreement?.length || a?.areas_of_disagreement?.length ||
              a?.gpt_unique_findings?.length || a?.claude_unique_findings?.length) ? (
              <>
                <button
                  onClick={() => setShowComparison(!showComparison)}
                  className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                  Model Comparison {showComparison ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                </button>
                {showComparison && (
                  <div className="space-y-3 mt-2">
                    {a?.areas_of_agreement && a.areas_of_agreement.length > 0 && (
                      <div>
                        <h4 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                          Areas of Agreement
                        </h4>
                        {a.areas_of_agreement.map((item, i) => (
                          <p key={i} className="text-sm text-muted-foreground pl-3 border-l-2 border-score-high mb-1">
                            {item}
                          </p>
                        ))}
                      </div>
                    )}
                    {a?.areas_of_disagreement && a.areas_of_disagreement.length > 0 && (
                      <div>
                        <h4 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                          Areas of Disagreement
                        </h4>
                        {a.areas_of_disagreement.map((item, i) => (
                          <p key={i} className="text-sm text-muted-foreground pl-3 border-l-2 border-score-moderate mb-1">
                            {item}
                          </p>
                        ))}
                      </div>
                    )}
                    {a?.gpt_unique_findings && a.gpt_unique_findings.length > 0 && (
                      <div>
                        <h4 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                          GPT-4 Unique Findings
                        </h4>
                        {a.gpt_unique_findings.map((item, i) => (
                          <p key={i} className="text-sm text-muted-foreground pl-3 border-l-2 border-border mb-1">
                            {item}
                          </p>
                        ))}
                      </div>
                    )}
                    {a?.claude_unique_findings && a.claude_unique_findings.length > 0 && (
                      <div>
                        <h4 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                          Claude Unique Findings
                        </h4>
                        {a.claude_unique_findings.map((item, i) => (
                          <p key={i} className="text-sm text-muted-foreground pl-3 border-l-2 border-border mb-1">
                            {item}
                          </p>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </>
            ) : null}

            {/* Recommendations */}
            {a?.recommendations && a.recommendations.length > 0 && (
              <div className="mt-3 pt-3 border-t border-border">
                <h4 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                  Recommendations
                </h4>
                {a.recommendations.map((rec, i) => (
                  <p key={i} className="text-sm text-muted-foreground pl-3 border-l-2 border-border mb-1">
                    {rec}
                  </p>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      <SessionInfo sessionId={data.session_id} processingTime={data.processing_time} />
    </div>
  );
};

export default BiasReport;