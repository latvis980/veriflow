// src/components/reports/KeyClaimsReport.tsx
import { cn } from "@/lib/utils";
import { ScoreBadge, SessionInfo, getScoreColor } from "./shared";
import { ChevronDown, ChevronUp, ExternalLink } from "lucide-react";
import { useState } from "react";

// Backend returns: id, statement, match_score (0-1), confidence (0-1), report, tier_breakdown
// Key claims mode also sends: claim_text as alias, assessment as alias
// TTS-verified claims also include: tts_story_url, tts_source_count, tts_cluster_title, tts_source_list
type Claim = {
  id: string;
  statement?: string;
  claim_text?: string;      // legacy alias
  match_score?: number;     // 0-1 range
  confidence?: number;
  report?: string;
  assessment?: string;      // legacy alias for report
  tier_breakdown?: { tier1?: number; tier2?: number; tier3?: number; filtered?: number } | null;
  // TTS verification metadata
  tts_story_url?: string | null;
  tts_source_count?: number | null;
  tts_cluster_title?: string | null;
  tts_source_list?: string | null;
};

type Props = {
  data: {
    // Backend uses "facts" (web search mode) or "key_claims" (key claims mode)
    facts?: Claim[];
    key_claims?: Claim[];
    no_claims_found?: boolean;
    summary?: {
      // Web search mode fields
      total_facts?: number;
      average_score?: number;
      // Key claims mode fields
      total_key_claims?: number;
      average_confidence?: number;
      overall_credibility?: string;
      // Shared fields
      verified_count?: number;
      partial_count?: number;
      unverified_count?: number;
    };
    session_id?: string;
    processing_time?: number;
  };
};

// Derive verification status from match_score
const deriveStatus = (score: number): string => {
  if (score >= 0.9) return "verified";
  if (score >= 0.7) return "partially_verified";
  return "unverified";
};

const statusLabel = (s: string) => {
  switch (s) {
    case "verified": return "VERIFIED";
    case "partially_verified": return "PARTIALLY VERIFIED";
    case "unverified": return "UNVERIFIED";
    default: return s?.toUpperCase();
  }
};

const statusColor = (s: string) => {
  switch (s) {
    case "verified": return "text-score-high";
    case "partially_verified": return "text-score-moderate";
    case "unverified": return "text-score-low";
    default: return "text-muted-foreground";
  }
};

const KeyClaimsReport = ({ data }: Props) => {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  // Handle both field name variants: "facts" (web search) or "key_claims" (key claims mode)
  const claims = data.facts || data.key_claims;

  if (data.no_claims_found || !claims?.length) {
    return (
      <div className="rounded-xl border border-border bg-card p-6 text-center">
        <p className="text-base text-muted-foreground">No verifiable claims found in this content.</p>
      </div>
    );
  }

  // Handle both summary field name variants
  const totalClaims = data.summary?.total_facts || data.summary?.total_key_claims || claims.length;

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-border bg-card p-5">
        <h3 className="text-xl font-display font-semibold mb-1">Key Claims Analysis</h3>
        <p className="text-sm text-muted-foreground mb-4">
          We identify up to 5 key claims that define the article's central meaning and verify each one thoroughly.
        </p>

        {data.summary && (
          <div className="flex flex-wrap gap-4 mb-4 p-3 rounded-lg bg-secondary">
            <div className="text-center">
              <span className="block text-xl font-bold">{totalClaims}</span>
              <span className="text-sm text-muted-foreground">Claims</span>
            </div>
            <div className="text-center">
              <span className="block text-xl font-bold text-score-high">{data.summary.verified_count ?? 0}</span>
              <span className="text-sm text-muted-foreground">Verified</span>
            </div>
            <div className="text-center">
              <span className="block text-xl font-bold text-score-moderate">{data.summary.partial_count ?? 0}</span>
              <span className="text-sm text-muted-foreground">Partial</span>
            </div>
            <div className="text-center">
              <span className="block text-xl font-bold text-score-low">{data.summary.unverified_count ?? 0}</span>
              <span className="text-sm text-muted-foreground">Unverified</span>
            </div>
            {data.summary.overall_credibility && (
              <div className="text-center">
                <span className="block text-base font-semibold capitalize">{data.summary.overall_credibility}</span>
                <span className="text-sm text-muted-foreground">Credibility</span>
              </div>
            )}
          </div>
        )}

        <div className="space-y-3">
          {claims.map((claim) => {
            const matchScore = claim.match_score ?? 0;
            const scorePercent = Math.round(matchScore * 100);
            const status = deriveStatus(matchScore);
            const isOpen = expanded[claim.id];
            const isTtsVerified = !!claim.tts_story_url;

            // Handle field name variants
            const statementText = claim.statement || claim.claim_text || "No statement available";
            const reportText = claim.report || claim.assessment || "";

            return (
              <div key={claim.id} className="rounded-lg border border-border p-4">
                <div className="flex items-start gap-3">
                  <ScoreBadge score={scorePercent} className="shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm font-semibold text-muted-foreground">
                        #{claim.id} KEY CLAIM
                      </span>
                      <span className={cn("text-sm font-semibold", statusColor(status))}>
                        {statusLabel(status)}
                      </span>
                    </div>
                    <p className="text-base font-medium mb-2">{statementText}</p>

                    {isTtsVerified ? (
                      /* TTS-verified claim: compact layout with link */
                      <div className="text-sm text-muted-foreground leading-relaxed space-y-1.5">
                        <p>{reportText}</p>
                        <a
                          href={claim.tts_story_url!}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors underline underline-offset-2"
                        >
                          View full context on The True Story
                          <ExternalLink size={11} />
                        </a>
                      </div>
                    ) : (
                      /* Standard claim: original layout */
                      <>
                        {reportText && (
                          <p className="text-sm text-muted-foreground leading-relaxed">{reportText}</p>
                        )}
                      </>
                    )}

                    {claim.tier_breakdown && (
                      <button
                        onClick={() => setExpanded(prev => ({ ...prev, [claim.id]: !isOpen }))}
                        className="flex items-center gap-1 mt-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
                      >
                        Source Tiers {isOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                      </button>
                    )}

                    {isOpen && claim.tier_breakdown && (
                      <div className="mt-2 flex gap-3 text-sm text-muted-foreground">
                        {claim.tier_breakdown.tier1 != null && claim.tier_breakdown.tier1 > 0 && (
                          <span>Tier 1: {claim.tier_breakdown.tier1}</span>
                        )}
                        {claim.tier_breakdown.tier2 != null && claim.tier_breakdown.tier2 > 0 && (
                          <span>Tier 2: {claim.tier_breakdown.tier2}</span>
                        )}
                        {claim.tier_breakdown.tier3 != null && claim.tier_breakdown.tier3 > 0 && (
                          <span>Tier 3: {claim.tier_breakdown.tier3}</span>
                        )}
                        {claim.tier_breakdown.filtered != null && claim.tier_breakdown.filtered > 0 && (
                          <span className="text-score-low">Filtered: {claim.tier_breakdown.filtered}</span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <SessionInfo sessionId={data.session_id} processingTime={data.processing_time} />
    </div>
  );
};

export default KeyClaimsReport;