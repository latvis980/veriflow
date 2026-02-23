// src/components/ContentInput.tsx
import { useState, useEffect, useMemo } from "react";
import { Link2, CheckCircle2, AlertTriangle } from "lucide-react";

type Props = {
  content: string;
  onContentChange: (v: string) => void;
  url: string;
  onUrlChange: (v: string) => void;
  onFetchUrl: () => void;
  isFetching: boolean;
  mode: string;
  onSwitchMode?: (mode: string) => void;
};

// ---------------------------------------------------------------------------
// Link detection helpers
// ---------------------------------------------------------------------------

function countLinks(text: string): number {
  if (!text) return 0;

  let count = 0;

  // HTML anchor tags
  const htmlMatches = text.match(/<\s*a\s+[^>]*href\s*=\s*["'][^"']+["'][^>]*>/gi);
  if (htmlMatches) count += htmlMatches.length;

  // Markdown reference links  [1]: https://...
  const refMatches = text.match(/^\s*\[\d+\]\s*:\s*https?:\/\//gm);
  if (refMatches) count += refMatches.length;

  // Markdown inline links  [text](https://...)
  const inlineMatches = text.match(/\[[^\]]+\]\(https?:\/\/[^)]+\)/g);
  if (inlineMatches) count += inlineMatches.length;

  // Fall back to plain URLs only if no structured links found
  if (count === 0) {
    const urlMatches = text.match(/https?:\/\/[^\s<>"']+/g);
    if (urlMatches) count = urlMatches.length;
  }

  return count;
}

// ---------------------------------------------------------------------------
// Mode descriptions -- shown above the textarea
// ---------------------------------------------------------------------------

const modeDescriptions: Record<string, { label: string; hint: string }> = {
  "comprehensive": {
    label: "Full-spectrum analysis",
    hint: "Runs all applicable analyses in one pass: source credibility, claims verification, bias, deception, and manipulation. Paste any article, text, or AI-generated content.",
  },
  "key-claims": {
    label: "Key claims verification",
    hint: "Identifies the 2-3 strongest factual claims and verifies each against independent sources. Paste the article or text you want to check.",
  },
  "bias-analysis": {
    label: "Bias and framing detection",
    hint: "Detects political or ideological lean, framing techniques, and one-sided sourcing patterns. Paste a news article, op-ed, or any content.",
  },
  "lie-detection": {
    label: "Deception signal analysis",
    hint: "Scans for linguistic markers of deception, hedging, and unsupported assertions. Paste the article or text to analyze.",
  },
  "manipulation": {
    label: "Manipulation detection",
    hint: "Flags agenda-driven framing, cherry-picked data, false equivalences, and emotional manipulation techniques. Paste an article or text to check.",
  },
  "llm-output": {
    label: "LLM output verification",
    hint: "Checks if AI-generated text truly matches the sources it cites. Verifies claims line by line against the linked materials and flags hallucinations, unsupported statements, citation mismatches, and misleading framing. Copy and paste the full AI response together with its source links (ChatGPT, Perplexity, etc.)",
  },
};

const simplePlaceholders: Record<string, string> = {
  "comprehensive": "Paste article, text, or AI-generated content here...",
  "key-claims": "Paste article or text here...",
  "bias-analysis": "Paste article or text here...",
  "lie-detection": "Paste article or text here...",
  "manipulation": "Paste article or text here...",
  "llm-output": "Paste the AI response with its source links here...",
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const ContentInput = ({
  content, onContentChange, url, onUrlChange,
  onFetchUrl, isFetching, mode, onSwitchMode,
}: Props) => {
  const [inputMode, setInputMode] = useState<"text" | "url">("text");

  // After a URL fetch completes and content is populated, switch to text view
  useEffect(() => {
    if (!isFetching && inputMode === "url" && content) {
      setInputMode("text");
    }
  }, [isFetching, content]);

  // LLM Output mode is copy-paste only -- force text input and hide URL toggle
  const isLlmMode = mode === "llm-output";

  useEffect(() => {
    if (isLlmMode && inputMode === "url") {
      setInputMode("text");
    }
  }, [isLlmMode]);

  // --- Link detection (only evaluated in LLM Output mode) ---
  const linkCount = useMemo(
    () => (isLlmMode ? countLinks(content) : 0),
    [content, isLlmMode],
  );

  const showIndicator = isLlmMode && content.length > 50;
  const hasLinks = linkCount > 0;

  const modeInfo = modeDescriptions[mode] || modeDescriptions["comprehensive"];
  const placeholder = simplePlaceholders[mode] || simplePlaceholders["comprehensive"];

  return (
    <div className="rounded-xl border border-border bg-card p-5">
      {/* Header row */}
      <div className="flex items-center justify-between mb-5">
        <h3 className="text-base font-semibold">
          {inputMode === "text" ? "Content to Analyze" : "Article URL"}
        </h3>
        {!isLlmMode && (
          inputMode === "text" ? (
            <button
              onClick={() => setInputMode("url")}
              className="relative flex items-center gap-1.5 rounded-md border border-primary/50 bg-primary/10 px-3 py-1.5 text-sm font-medium text-primary hover:bg-primary/20 hover:border-primary transition-colors"
            >
              {/* pulse ring */}
              <span className="absolute inset-0 rounded-md animate-ping border border-primary/40 pointer-events-none" />
              <Link2 size={15} />
              Paste URL instead
            </button>
          ) : (
            <button
              onClick={() => setInputMode("text")}
              className="flex items-center gap-1.5 rounded-md border border-border bg-secondary/60 px-3 py-1.5 text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
            >
              <Link2 size={15} />
              Paste text instead
            </button>
          )
        )}
      </div>

      {/* Mode description -- always visible above the input */}
      {inputMode === "text" && (
        <p className="text-base text-muted-foreground mb-3 leading-relaxed max-w-[66%]">
          <span className="font-medium text-foreground">{modeInfo.label}: </span>
          {modeInfo.hint}
        </p>
      )}

      {/* Input area */}
      {inputMode === "text" ? (
        <textarea
          value={content}
          onChange={(e) => onContentChange(e.target.value)}
          placeholder={placeholder}
          className="w-full min-h-[200px] resize-y rounded-lg border border-border bg-background p-3 text-base placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
        />
      ) : (
        <div className="flex gap-2">
          <input
            type="url"
            value={url}
            onChange={(e) => onUrlChange(e.target.value)}
            placeholder="https://example.com/article"
            className="flex-1 rounded-lg border border-border bg-background px-3 py-2 text-base placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
          />
          <button
            onClick={onFetchUrl}
            disabled={isFetching || !url}
            className="rounded-lg bg-primary px-4 py-2 text-base font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors"
          >
            {isFetching ? "Fetching..." : "Fetch"}
          </button>
        </div>
      )}

      {/* --- Link format indicator (LLM Output mode only) --- */}
      {showIndicator && (
        <div
          className={`flex items-start gap-2.5 mt-3 px-3.5 py-2.5 rounded-lg text-sm ${
            hasLinks
              ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
              : "bg-amber-500/10 text-amber-700 dark:text-amber-400"
          }`}
        >
          {hasLinks ? (
            <>
              <CheckCircle2 size={16} className="mt-0.5 shrink-0" />
              <span>
                Detected {linkCount} source link{linkCount !== 1 ? "s" : ""}
              </span>
            </>
          ) : (
            <>
              <AlertTriangle size={16} className="mt-0.5 shrink-0" />
              <span>
                No source links detected. LLM Output mode verifies AI responses against their
                cited sources.{" "}
                {onSwitchMode && (
                  <>
                    Try{" "}
                    <button
                      type="button"
                      onClick={() => onSwitchMode("comprehensive")}
                      className="underline font-medium hover:opacity-80"
                    >
                      Comprehensive Analysis
                    </button>{" "}
                    or{" "}
                    <button
                      type="button"
                      onClick={() => onSwitchMode("key-claims")}
                      className="underline font-medium hover:opacity-80"
                    >
                      Key Claims
                    </button>{" "}
                    for text without links.
                  </>
                )}
              </span>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default ContentInput;
