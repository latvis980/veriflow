// src/pages/Index.tsx
import { useState, useCallback, useRef } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import ModeSelector from "@/components/ModeSelector";
import ContentInput from "@/components/ContentInput";
import SourceCard from "@/components/SourceCard";
import ProgressLog from "@/components/ProgressLog";
import ReportRenderer from "@/components/ReportRenderer";
import {
  postJob, fetchJobResult, cancelJob, streamJob,
  type AnalysisMode, MODE_ENDPOINTS, MODE_INFO, buildRequestBody,
} from "@/lib/api";

type AppState = "idle" | "fetching" | "analyzing" | "done" | "error";

const Index = () => {
  const [mode, setMode] = useState<AnalysisMode>("llm-output");
  const [content, setContent] = useState("");
  const [url, setUrl] = useState("");
  const [article, setArticle] = useState<Record<string, any> | null>(null);
  const [state, setState] = useState<AppState>("idle");
  const [messages, setMessages] = useState<string[]>([]);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [analyzedMode, setAnalyzedMode] = useState<AnalysisMode | null>(null);
  const [showContent, setShowContent] = useState(false);
  const closeStreamRef = useRef<(() => void) | null>(null);

  const reset = useCallback(() => {
    setState("idle");
    setMessages([]);
    setResult(null);
    setError(null);
    setAnalyzedMode(null);
    setShowContent(false);
  }, []);

  const fullReset = useCallback(() => {
    reset();
    setContent("");
    setUrl("");
    setArticle(null);
  }, [reset]);

  // When user selects a mode while results are showing,
  // keep the content but clear the results so they can re-analyze.
  const handleModeSelect = useCallback((newMode: AnalysisMode) => {
    if (state === "done" && newMode !== analyzedMode) {
      setState("idle");
      setMessages([]);
      setResult(null);
      setError(null);
      setAnalyzedMode(null);
      setShowContent(false);
    }
    setMode(newMode);
  }, [state, analyzedMode]);

  const handleFetchUrl = useCallback(async () => {
    if (!url) return;
    setState("fetching");
    setMessages(["Fetching article from URL..."]);

    try {
      const { job_id } = await postJob("/api/scrape-url", {
        url,
        extract_metadata: true,
        check_credibility: true,
        run_mbfc_if_missing: true,
      });

      closeStreamRef.current = streamJob(job_id, {
        onMessage: (msg) => setMessages((prev) => [...prev, msg]),
        onComplete: async () => {
          const job = await fetchJobResult(job_id);
          const r = job.result || job;
          setArticle(r);
          if (r.content) setContent(r.content);
          setState("idle");
          setMessages([]);
        },
        onError: (err) => {
          setError(err);
          setState("error");
        },
      });
    } catch (e: any) {
      setError(e.message);
      setState("error");
    }
  }, [url]);

  const handleAnalyze = useCallback(async () => {
    if (!content.trim()) return;
    setState("analyzing");
    setMessages(["Starting analysis..."]);
    setResult(null);
    setError(null);
    setAnalyzedMode(null);
    setShowContent(false);

    const sourceContext = article?.credibility
      ? {
          publication: article.publication_name,
          credibility_tier: article.credibility.tier,
          bias_rating: article.credibility.bias_rating,
          factual_reporting: article.credibility.factual_reporting,
        }
      : undefined;

    const body = buildRequestBody(mode, content, url || article?.url, sourceContext);

    try {
      const { job_id } = await postJob(MODE_ENDPOINTS[mode], body);

      closeStreamRef.current = streamJob(job_id, {
        onMessage: (msg) => setMessages((prev) => [...prev, msg]),
        onComplete: async () => {
          const job = await fetchJobResult(job_id);
          setResult(job.result || job);
          setAnalyzedMode(mode);
          setState("done");
        },
        onError: (err) => {
          setError(err);
          setState("error");
        },
      });
    } catch (e: any) {
      setError(e.message);
      setState("error");
    }
  }, [content, mode, url, article]);

  const handleCancel = useCallback(async () => {
    closeStreamRef.current?.();
    setState("idle");
    setMessages([]);
  }, []);

  const isProcessing = state === "analyzing" || state === "fetching";

  // Truncate content for preview (first ~300 chars)
  const contentPreview = content.length > 300
    ? content.slice(0, 300) + "..."
    : content;

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1">
        <div className="container max-w-5xl py-10">
          {/* Hero */}
          <h1 className="text-4xl md:text-5xl font-display text-center mb-3">
            AI Content Analyzer
          </h1>
          <p className="text-center text-muted-foreground text-base md:text-lg max-w-2xl mx-auto mb-10">
            Verify LLM outputs against original sources and analyze any text for bias,
            deception signals, manipulation patterns, and agenda-driven framing.
          </p>

          {/* Mode Selector */}
          <div className="mb-6">
            <ModeSelector
              selected={mode}
              onSelect={handleModeSelect}
              analyzedMode={analyzedMode}
            />
          </div>

          {/* Source Card */}
          {article && !isProcessing && state !== "done" && (
            <div className="mb-4">
              <SourceCard article={article} />
            </div>
          )}

          {/* Input -- shown when NOT in done state */}
          {state !== "done" && (
            <div className="mb-4">
              <ContentInput
                content={content}
                onContentChange={setContent}
                url={url}
                onUrlChange={setUrl}
                onFetchUrl={handleFetchUrl}
                isFetching={state === "fetching"}
                mode={mode}
                onSwitchMode={(m) => handleModeSelect(m as AnalysisMode)}
              />
            </div>
          )}

          {/* Actions */}
          {state === "idle" && (
            <div className="flex justify-center gap-3 mb-6">
              <button
                onClick={fullReset}
                className="rounded-lg border border-border px-5 py-2.5 text-base font-medium hover:bg-secondary transition-colors"
              >
                Clear
              </button>
              <button
                onClick={handleAnalyze}
                disabled={!content.trim()}
                className="rounded-lg bg-primary px-6 py-2.5 text-base font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors"
              >
                Analyze
              </button>
            </div>
          )}

          {/* Progress */}
          {isProcessing && (
            <div className="mb-6">
              <ProgressLog
                messages={messages}
                isActive={true}
                onCancel={handleCancel}
              />
            </div>
          )}

          {/* Error */}
          {state === "error" && error && (
            <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-4 mb-6 text-center">
              <p className="text-base text-destructive">{error}</p>
              <button
                onClick={reset}
                className="mt-2 text-sm text-muted-foreground hover:text-foreground underline"
              >
                Try again
              </button>
            </div>
          )}

          {/* Collapsible analyzed content + Report */}
          {state === "done" && result && analyzedMode && (
            <>
              {/* Collapsible content preview */}
              <div className="rounded-xl border border-border bg-card mb-4">
                <button
                  onClick={() => setShowContent((v) => !v)}
                  className="w-full flex items-center justify-between px-5 py-3 text-left"
                >
                  <span className="text-sm font-medium text-muted-foreground">
                    Analyzed Content
                    {url && (
                      <span className="ml-2 text-muted-foreground/60">{url}</span>
                    )}
                  </span>
                  {showContent
                    ? <ChevronUp size={14} className="text-muted-foreground" />
                    : <ChevronDown size={14} className="text-muted-foreground" />
                  }
                </button>
                {showContent && (
                  <div className="px-5 pb-4 border-t border-border">
                    <pre className="mt-3 text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap max-h-60 overflow-y-auto">
                      {content}
                    </pre>
                  </div>
                )}
              </div>

              {/* Report -- always rendered with the mode that actually produced results */}
              <ReportRenderer mode={analyzedMode} data={result} onReset={fullReset} sourceUrl={url || undefined} />
            </>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Index;