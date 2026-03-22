# app.py
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import os
import re
import threading
from typing import Optional, Dict
from dotenv import load_dotenv

# Import your components
from orchestrator.llm_output_orchestrator import LLMInterpretationOrchestrator
from orchestrator.web_search_orchestrator import WebSearchOrchestrator
from orchestrator.bias_check_orchestrator import BiasCheckOrchestrator
from orchestrator.lie_detector_orchestrator import LieDetectorOrchestrator
from orchestrator.key_claims_orchestrator import KeyClaimsOrchestrator
from orchestrator.manipulation_orchestrator import ManipulationOrchestrator
from orchestrator.comprehensive_orchestrator import ComprehensiveOrchestrator

from utils.logger import fact_logger
from utils.langsmith_config import langsmith_config
from utils.job_manager import job_manager
from utils.async_utils import run_async_in_thread, cleanup_thread_loop


# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:5173",
            "http://localhost:3000",
            "https://*.lovable.app",
            "https://*.lovableproject.com",
            "https://*.replit.dev",
            "https://*.repl.co",
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
    }
})

# Load configuration
class Config:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.browserless_endpoint = os.getenv('BROWSER_PLAYWRIGHT_ENDPOINT_PRIVATE')
        self.brave_api_key = os.getenv('BRAVE_API_KEY')
        self.langchain_project = os.getenv('LANGCHAIN_PROJECT', 'fact-checker')

        # Validate required env vars
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")

        if not self.brave_api_key:
            fact_logger.logger.warning("  BRAVE_API_KEY not set - web search pipeline will not work")

        fact_logger.logger.info(" Configuration loaded successfully")

config = Config()

# 1. LLM Interpretation Orchestrator (for LLM output with sources)
llm_interpretation_orchestrator: Optional[LLMInterpretationOrchestrator] = None
try:
    llm_interpretation_orchestrator = LLMInterpretationOrchestrator(config)
    fact_logger.logger.info(" LLM Interpretation Orchestrator initialized successfully")
except Exception as e:
    fact_logger.logger.error(f" Failed to initialize LLM Interpretation Orchestrator: {e}")
    llm_interpretation_orchestrator = None

# 2. Web Search Orchestrator (for fact-checking any text via web search)
web_search_orchestrator: Optional[WebSearchOrchestrator] = None
if config.brave_api_key:
    try:
        web_search_orchestrator = WebSearchOrchestrator(config)
        fact_logger.logger.info(" Web Search Orchestrator initialized successfully")
    except Exception as e:
        fact_logger.logger.error(f" Failed to initialize Web Search Orchestrator: {e}")
        fact_logger.logger.warning("  Web search pipeline will not be available")
        web_search_orchestrator = None
else:
    fact_logger.logger.warning("  BRAVE_API_KEY not set - web search will not work")

# 3. Bias Check Orchestrator (analyzes text for political/ideological bias)
bias_orchestrator: Optional[BiasCheckOrchestrator] = None
try:
    bias_orchestrator = BiasCheckOrchestrator(config)
    fact_logger.logger.info(" Bias Check Orchestrator initialized successfully")
except Exception as e:
    fact_logger.logger.error(f" Failed to initialize Bias Check Orchestrator: {e}")
    bias_orchestrator = None

# 4. Lie Detector Orchestrator (detects linguistic markers of deception)
lie_detector_orchestrator: Optional[LieDetectorOrchestrator] = None
try:
    lie_detector_orchestrator = LieDetectorOrchestrator(config)
    fact_logger.logger.info(" Lie Detector Orchestrator initialized successfully")
except Exception as e:
    fact_logger.logger.error(f" Failed to initialize Lie Detector Orchestrator: {e}")
    lie_detector_orchestrator = None

# 5. Key Claims Orchestrator (extracts and verifies 2-3 central thesis claims)
key_claims_orchestrator: Optional[KeyClaimsOrchestrator] = None
if config.brave_api_key:
    try:
        key_claims_orchestrator = KeyClaimsOrchestrator(config)
        fact_logger.logger.info(" Key Claims Orchestrator initialized successfully")
    except Exception as e:
        fact_logger.logger.error(f" Failed to initialize Key Claims Orchestrator: {e}")
        key_claims_orchestrator = None

# 6. Manipulation Detection Orchestrator (detects agenda-driven fact manipulation)
manipulation_orchestrator: Optional[ManipulationOrchestrator] = None
if config.brave_api_key:
    try:
        manipulation_orchestrator = ManipulationOrchestrator(config)
        fact_logger.logger.info(" Manipulation Detection Orchestrator initialized successfully")
    except Exception as e:
        fact_logger.logger.error(f" Failed to initialize Manipulation Orchestrator: {e}")
        manipulation_orchestrator = None
else:
    fact_logger.logger.warning("  Manipulation Detection requires BRAVE_API_KEY for fact verification")

# Log summary
fact_logger.logger.info("Orchestrator initialization complete:")
fact_logger.logger.info(f"  - LLM Interpretation: {'OK' if llm_interpretation_orchestrator else 'SKIP'}")
fact_logger.logger.info(f"  - Web Search: {'OK' if web_search_orchestrator else 'SKIP'}")
fact_logger.logger.info(f"  - Bias Check: {'OK' if bias_orchestrator else 'SKIP'}")
fact_logger.logger.info(f"  - Lie Detection: {'OK' if lie_detector_orchestrator else 'SKIP'}")

def detect_input_format(content: str) -> str:
    """
    Detect if input is HTML/Markdown (LLM output with links) or plain text
    """
    # Check for HTML tags
    html_pattern = r'<\s*[a-z][^>]*>'
    has_html_tags = bool(re.search(html_pattern, content, re.IGNORECASE))
    has_html_links = bool(re.search(r'<\s*a\s+[^>]*href\s*=', content, re.IGNORECASE))

    # Check for markdown reference links: [1]: https://...
    markdown_ref_pattern = r'^\s*\[\d+\]\s*:\s*https?://'
    has_markdown_refs = bool(re.search(markdown_ref_pattern, content, re.MULTILINE))

    # Check for markdown inline links: [text](https://...)
    markdown_inline_pattern = r'\[([^\]]+)\]\(https?://[^\)]+\)'
    has_markdown_inline = bool(re.search(markdown_inline_pattern, content))

    # Check for plain URLs (need at least 2)
    url_pattern = r'https?://[^\s]+'
    url_matches = re.findall(url_pattern, content)
    has_multiple_urls = len(url_matches) >= 2

    if has_html_tags or has_html_links or has_markdown_refs or has_markdown_inline or has_multiple_urls:
        fact_logger.logger.info(" Detected HTML/Markdown input format (LLM output with links)")
        return 'html'
    else:
        fact_logger.logger.info(" Detected plain text input format (no links)")
        return 'text'


# ---------- Serve React frontend ----------
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), 'frontend', 'dist')

@app.route('/')
def index():
    """Serve the React frontend"""
    return send_from_directory(FRONTEND_DIST, 'index.html')

@app.route('/<path:path>')
def serve_frontend(path):
    """Serve React static assets, fall back to index.html for client-side routing"""
    # If the file exists in dist/, serve it (JS, CSS, images, etc.)
    file_path = os.path.join(FRONTEND_DIST, path)
    if os.path.isfile(file_path):
        return send_from_directory(FRONTEND_DIST, path)
    # Otherwise, let React Router handle it
    return send_from_directory(FRONTEND_DIST, 'index.html')


@app.route('/api/check', methods=['POST'])
def check_facts():
    """
    Start async fact-check job and return job_id immediately

    Accepts optional 'input_type' parameter to explicitly specify pipeline:
    - 'html': Use LLM output pipeline (scrapes provided source links)
    - 'text': Use web search pipeline (searches web for verification)

    If input_type is not provided, auto-detects based on content.
    """
    try:
        # Get content from request
        request_json = request.get_json()
        if not request_json:
            return jsonify({"error": "Invalid request format"}), 400

        content = request_json.get('html_content') or request_json.get('content')
        if not content:
            return jsonify({"error": "No content provided"}), 400

        # Check for explicit input_type from frontend
        explicit_type = request_json.get('input_type')

        fact_logger.logger.info(
            " Received fact-check request",
            extra={
                "content_length": len(content),
                "explicit_type": explicit_type
            }
        )

        # Determine input format: use explicit type if provided, otherwise auto-detect
        if explicit_type in ['html', 'text']:
            input_format = explicit_type
            fact_logger.logger.info(f" Using explicit input type: {input_format}")
        else:
            input_format = detect_input_format(content)
            fact_logger.logger.info(f" Auto-detected input type: {input_format}")

        # Type-safe check for web search orchestrator
        if input_format == 'text' and web_search_orchestrator is None:
            return jsonify({
                "error": "Web search pipeline not available",
                "message": "BRAVE_API_KEY not configured or initialization failed. Please use LLM Output mode with content that has source links."
            }), 503

        # Create job
        job_id = job_manager.create_job(content)
        fact_logger.logger.info(f" Created job: {job_id} (format: {input_format})")

        # Start background processing
        threading.Thread(
            target=run_async_task,
            args=(job_id, content, input_format),
            daemon=True
        ).start()

        return jsonify({
            "job_id": job_id,
            "status": "processing",
            "message": f"Fact-checking started ({input_format} pipeline)",
            "pipeline": input_format
        })

    except Exception as e:
        fact_logger.log_component_error("Flask API", e)
        return jsonify({
            "error": str(e),
            "message": "An error occurred during fact checking"
        }), 500

@app.route('/api/comprehensive-analysis', methods=['POST'])
def start_comprehensive_analysis():
    """
    Start comprehensive analysis pipeline

    Request body:
    {
        "content": "text to analyze",
        "source_url": "optional source URL",
        "user_preferences": {  // optional
            "force_include": ["mode_id"],
            "force_exclude": ["mode_id"]
        }
    }

    Response:
    {
        "job_id": "uuid",
        "status": "started",
        "message": "Comprehensive analysis started"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        content = data.get('content', '').strip()

        if not content:
            return jsonify({"error": "Content is required"}), 400

        # Optional parameters
        source_url = data.get('source_url', '').strip() or None
        user_preferences = data.get('user_preferences', {})

        # Create job
        job_id = job_manager.create_job(content=content)

        fact_logger.logger.info(
            f" Starting comprehensive analysis job: {job_id}",
            extra={
                "job_id": job_id,
                "content_length": len(content),
                "has_source_url": source_url is not None
            }
        )

        # Start background task
        def run_comprehensive_task():
            try:
                async def _run():
                    try:
                        orchestrator = ComprehensiveOrchestrator(config)
                        result = await orchestrator.process_with_progress(
                            content=content,
                            job_id=job_id,
                            source_url=source_url,
                            user_preferences=user_preferences
                        )
                        return result
                    except Exception as e:
                        fact_logger.logger.error(f" Comprehensive analysis error: {e}")
                        import traceback
                        fact_logger.logger.error(f"Traceback: {traceback.format_exc()}")
                        job_manager.fail_job(job_id, str(e))
                        raise

                run_async_in_thread(_run())

            except Exception as e:
                fact_logger.logger.error(f" Comprehensive task error: {e}")
                job_manager.fail_job(job_id, str(e))
            finally:
                cleanup_thread_loop()

        # Start in thread
        thread = threading.Thread(target=run_comprehensive_task)
        thread.start()

        return jsonify({
            "job_id": job_id,
            "status": "started",
            "message": "Comprehensive analysis started"
        })

    except Exception as e:
        fact_logger.logger.error(f" Failed to start comprehensive analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/key-claims', methods=['POST'])
def check_key_claims():
    """
    Extract and verify key claims from text.

    NOW ACCEPTS: source_credibility parameter to provide context about
    the source's reliability when verifying claims.

    Request body:
        {
            "content": "Article text to analyze...",
            "source_context": {                          # NEW - Optional
                "url": "https://example.com/article",
                "publication_name": "Example News",
                "publication_date": "2024-01-15"
            },
            "source_credibility": {                      # NEW - Optional
                "tier": 2,
                "bias_rating": "CENTER",
                "factual_reporting": "HIGH",
                ...
            }
        }
    """
    try:
        request_json = request.get_json()
        if not request_json:
            return jsonify({"error": "Invalid request format"}), 400

        content = request_json.get('content')

        # NEW: Accept source context and credibility
        source_context = request_json.get('source_context')  # url, name, date
        source_credibility = request_json.get('source_credibility')

        if not content:
            return jsonify({"error": "No content provided"}), 400

        if key_claims_orchestrator is None:
            return jsonify({
                "error": "Key claims pipeline not available",
                "message": "Key Claims Orchestrator not initialized"
            }), 503

        fact_logger.logger.info(
            " Received key claims request",
            extra={
                "content_length": len(content),
                "has_source_context": source_context is not None,
                "has_credibility": source_credibility is not None
            }
        )

        job_id = job_manager.create_job(content)
        fact_logger.logger.info(f" Created key claims job: {job_id}")

        # Start background processing with new parameters
        threading.Thread(
            target=run_key_claims_task,
            args=(job_id, content, source_context, source_credibility),  # UPDATED
            daemon=True
        ).start()

        return jsonify({
            "job_id": job_id,
            "status": "processing",
            "message": "Key claims analysis started"
        })

    except Exception as e:
        fact_logger.log_component_error("Flask API - Key Claims", e)
        return jsonify({"error": str(e)}), 500

def run_key_claims_task(
    job_id: str, 
    content: str,
    source_context: Optional[Dict] = None,      # NEW PARAMETER
    source_credibility: Optional[Dict] = None   # NEW PARAMETER
):
    """
    Background task runner for key claims verification.

    Args:
        job_id: Job ID for tracking
        content: Text to analyze
        source_context: Optional dict with url, publication_name, publication_date
        source_credibility: Optional pre-fetched credibility data
    """
    try:
        if key_claims_orchestrator is None:
            raise ValueError("Key claims orchestrator not initialized")

        fact_logger.logger.info(
            f" Job {job_id}: Starting key claims analysis",
            extra={
                "has_source_context": source_context is not None,
                "has_credibility": source_credibility is not None
            }
        )

        _ = run_async_in_thread(
            key_claims_orchestrator.process_with_progress(
                text_content=content,
                job_id=job_id,
                source_context=source_context,
                source_credibility=source_credibility
            )
        )

        # Note: job completion handled inside process_with_progress
        fact_logger.logger.info(f" Key claims job {job_id} completed")

    except Exception as e:
        fact_logger.log_component_error(f"Key Claims Job {job_id}", e)
        job_manager.fail_job(job_id, str(e))
    finally:
        cleanup_thread_loop()

@app.route('/api/bias', methods=['POST'])
def check_bias():
    '''Check text for political and other biases using multiple LLMs'''
    try:
        request_json = request.get_json()
        if not request_json:
            return jsonify({"error": "Invalid request format"}), 400

        text = request_json.get('text') or request_json.get('content')
        publication_url = request_json.get('publication_url')

        # NEW: Accept source_context from new frontend
        source_context = request_json.get('source_context')

        # Convert source_context to source_credibility format if provided
        source_credibility = None
        if source_context:
            source_credibility = {
                'publication_name': source_context.get('publication'),
                'tier': source_context.get('credibility_tier'),
                'bias_rating': source_context.get('bias_rating'),
                'factual_reporting': source_context.get('factual_reporting'),
                'source': 'frontend_prefetched'
            }

        if not text:
            return jsonify({"error": "No text provided"}), 400

        if not bias_orchestrator:
            return jsonify({
                "error": "Bias analysis not available",
                "message": "Bias Check Orchestrator not initialized"
            }), 503

        fact_logger.logger.info(
            " Received bias check request",
            extra={
                "text_length": len(text),
                "publication_url": publication_url  # NEW
            }
        )

        # Create job
        job_id = job_manager.create_job(text)
        fact_logger.logger.info(f" Created bias check job: {job_id}")

        # Start background processing
        threading.Thread(
            target=run_bias_task,
            args=(job_id, text, publication_url, source_credibility), 
            daemon=True
        ).start()

        return jsonify({
            "job_id": job_id,
            "status": "processing",
            "message": "Bias analysis started"
        })

    except Exception as e:
        fact_logger.log_component_error("Flask API - Bias Check", e)
        return jsonify({
            "error": str(e),
            "message": "An error occurred during bias analysis"
        }), 500

@app.route('/api/lie-detection', methods=['POST'])
def check_lie_detection():
    """
    Analyze text for linguistic markers of deception/fake news.

    NEW API endpoint (replaces /api/check-lie-detection).
    Accepts source_context from new frontend for credibility calibration.

    Request body:
        {
            "content": "Text to analyze...",
            "text": "Text to analyze...",           // Alternative field name
            "article_source": "Publication name",   // Optional
            "article_date": "2024-01-15",          // Optional
            "source_context": {                     // NEW - from frontend
                "publication": "Fox News",
                "credibility_tier": 3,
                "bias_rating": "RIGHT",
                "factual_reporting": "MIXED"
            },
            "source_credibility": {...}            // Legacy - direct pass-through
        }

    Returns:
        {"job_id": "...", "status": "processing", "message": "..."}
    """
    try:
        # Get content from request
        request_json = request.get_json()
        if not request_json:
            return jsonify({"error": "Invalid request format"}), 400

        # Accept both 'text' and 'content' field names
        text = request_json.get('text') or request_json.get('content')
        article_source = request_json.get('article_source')
        article_date = request_json.get('article_date')

        # Handle source credibility - two possible formats:
        # 1. Direct source_credibility (legacy/internal)
        # 2. source_context from new frontend (needs conversion)
        source_credibility = request_json.get('source_credibility')
        source_context = request_json.get('source_context')

        # Convert source_context to source_credibility format if provided
        if source_context and not source_credibility:
            source_credibility = {
                'publication_name': source_context.get('publication'),
                'tier': source_context.get('credibility_tier'),
                'credibility_tier': source_context.get('credibility_tier'),  # Some functions expect this key
                'bias_rating': source_context.get('bias_rating'),
                'factual_reporting': source_context.get('factual_reporting'),
                'source': 'frontend_prefetched'
            }
            # Use publication as article_source if not provided
            if not article_source and source_context.get('publication'):
                article_source = source_context.get('publication')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        if not lie_detector_orchestrator:
            return jsonify({
                "error": "Lie detection not available",
                "message": "Lie Detector Orchestrator not initialized"
            }), 503

        fact_logger.logger.info(
            " Received lie detection request",
            extra={
                "text_length": len(text),
                "has_source": bool(article_source),
                "has_date": bool(article_date),
                "has_source_credibility": source_credibility is not None,
                "credibility_source": source_credibility.get('source') if source_credibility else None
            }
        )

        # Create job
        job_id = job_manager.create_job(text)
        fact_logger.logger.info(f" Created lie detection job: {job_id}")

        # Start background processing
        threading.Thread(
            target=run_lie_detection_task,
            args=(job_id, text, article_source, article_date, source_credibility),
            daemon=True
        ).start()

        return jsonify({
            "job_id": job_id,
            "status": "processing",
            "message": "Lie detection analysis started"
        })

    except Exception as e:
        fact_logger.log_component_error("Flask API - Lie Detection", e)
        return jsonify({
            "error": str(e),
            "message": "An error occurred during lie detection"
        }), 500

@app.route('/api/manipulation', methods=['POST'])
def check_manipulation():
    """
    Analyze article for opinion manipulation and agenda-driven fact distortion.

    NOW ACCEPTS: source_credibility parameter to calibrate scrutiny level
    based on source reliability.

    Request body:
        {
            "content": "Article text to analyze...",
            "source_info": "https://example.com/article",  # URL or source name
            "source_credibility": {                         # NEW - Optional
                "tier": 4,
                "bias_rating": "RIGHT",
                "factual_reporting": "MIXED",
                "is_propaganda": false,
                ...
            }
        }
    """
    try:
        # Get content from request
        request_json = request.get_json()
        if not request_json:
            return jsonify({"error": "Invalid request format"}), 400

        content = request_json.get('content') or request_json.get('text')
        source_info = request_json.get('source_info', 'Unknown source')

        # NEW: Accept pre-fetched credibility data
        source_credibility = request_json.get('source_credibility')

        if not content:
            return jsonify({"error": "No content provided"}), 400

        if manipulation_orchestrator is None:
            return jsonify({
                "error": "Manipulation detection not available",
                "message": "Manipulation Orchestrator not initialized. Requires BRAVE_API_KEY."
            }), 503

        fact_logger.logger.info(
            " Received manipulation detection request",
            extra={
                "content_length": len(content),
                "source_info": source_info,
                "has_credibility": source_credibility is not None
            }
        )

        # Create job
        job_id = job_manager.create_job(content)
        fact_logger.logger.info(f" Created manipulation detection job: {job_id}")

        # Start background processing with source_credibility
        threading.Thread(
            target=run_manipulation_task,
            args=(job_id, content, source_info, source_credibility),  # UPDATED
            daemon=True
        ).start()

        return jsonify({
            "job_id": job_id,
            "status": "processing",
            "message": "Manipulation analysis started"
        })

    except Exception as e:
        fact_logger.log_component_error("Flask API - Manipulation Detection", e)
        return jsonify({
            "error": str(e),
            "message": "An error occurred during manipulation analysis"
        }), 500


def run_lie_detection_task(
    job_id: str, 
    text: str, 
    article_source: Optional[str], 
    article_date: Optional[str],
    source_credibility: Optional[Dict] = None  # NEW PARAMETER
):
    """
    Background task runner for lie detection analysis.

    Args:
        job_id: Job ID for tracking
        text: Text to analyze
        article_source: Optional publication name
        article_date: Optional publication date
        source_credibility: Optional pre-fetched credibility data (NEW)
    """
    try:
        if lie_detector_orchestrator is None:
            raise ValueError("Lie detector orchestrator not initialized")

        fact_logger.logger.info(
            f" Job {job_id}: Starting lie detection analysis",
            extra={
                "has_source": bool(article_source),
                "has_credibility": source_credibility is not None
            }
        )

        _ = run_async_in_thread(
            lie_detector_orchestrator.process_with_progress(
                text=text,
                job_id=job_id,
                article_source=article_source,
                publication_date=article_date,
                source_credibility=source_credibility
            )
        )

        # Note: job completion is handled inside process_with_progress
        fact_logger.logger.info(f" Lie detection job {job_id} completed successfully")

    except Exception as e:
        fact_logger.log_component_error(f"Lie Detection Job {job_id}", e)
        job_manager.fail_job(job_id, str(e))

    finally:
        cleanup_thread_loop()

def run_manipulation_task(
    job_id: str, 
    content: str, 
    source_info: str,
    source_credibility: Optional[Dict] = None  # NEW PARAMETER
):
    """
    Background task runner for manipulation detection.

    Args:
        job_id: Job ID for tracking
        content: Article text to analyze
        source_info: URL or source name
        source_credibility: Optional pre-fetched credibility data (NEW)
    """
    try:
        if manipulation_orchestrator is None:
            raise ValueError("Manipulation orchestrator not initialized")

        fact_logger.logger.info(
            f" Job {job_id}: Starting manipulation detection",
            extra={
                "source_info": source_info,
                "has_credibility": source_credibility is not None
            }
        )

        _ = run_async_in_thread(
            manipulation_orchestrator.process_with_progress(
                content=content,
                job_id=job_id,
                source_info=source_info,
                source_credibility=source_credibility
            )
        )

        # Note: job completion handled inside process_with_progress
        fact_logger.logger.info(f" Manipulation detection job {job_id} completed")

    except Exception as e:
        fact_logger.log_component_error(f"Manipulation Job {job_id}", e)
        job_manager.fail_job(job_id, str(e))
    finally:
        cleanup_thread_loop()


def run_async_task(job_id: str, content: str, input_format: str):
    """
    Background task runner for fact checking.
    Routes to appropriate orchestrator based on input format:
    - 'html'  LLM Interpretation Orchestrator (checks if LLM interpreted sources correctly)
    - 'text'  Web Search Orchestrator (fact-checks via web search)
    """
    try:
        if input_format == 'html':
            # LLM output with sources  Interpretation verification
            if llm_interpretation_orchestrator is None:
                raise ValueError("LLM Interpretation orchestrator not initialized")

            fact_logger.logger.info(f" Job {job_id}: LLM Interpretation Verification pipeline")
            result = run_async_in_thread(
                llm_interpretation_orchestrator.process_with_progress(content, job_id)
            )

        else:  # input_format == 'text'
            # Plain text  Fact-checking via web search
            if web_search_orchestrator is None:
                raise ValueError("Web search orchestrator not initialized - BRAVE_API_KEY may be missing")

            fact_logger.logger.info(f" Job {job_id}: Web Search Fact-Checking pipeline")
            result = run_async_in_thread(
                web_search_orchestrator.process_with_progress(content, job_id)
            )

        # Store successful result
        job_manager.complete_job(job_id, result)
        fact_logger.logger.info(f" Job {job_id} completed successfully")

    except Exception as e:
        fact_logger.log_component_error(f"Job {job_id}", e)
        job_manager.fail_job(job_id, str(e))

    finally:
        cleanup_thread_loop()

def run_bias_task(
    job_id: str, 
    text: str, 
    publication_url: Optional[str],
    source_credibility: Optional[Dict] = None
):
    """Background task runner for bias analysis."""
    try:
        if bias_orchestrator is None:
            raise ValueError("Bias orchestrator not initialized")

        fact_logger.logger.info(
            f" Job {job_id}: Starting bias analysis",
            extra={
                "has_publication_url": bool(publication_url),
                "has_credibility": source_credibility is not None
            }
        )

        _ = run_async_in_thread(
            bias_orchestrator.process(
                text=text,
                publication_url=publication_url,
                source_credibility=source_credibility
            )
        )

        job_manager.complete_job(job_id, _)
        fact_logger.logger.info(f" Bias check job {job_id} completed successfully")

    except Exception as e:
        fact_logger.log_component_error(f"Bias Job {job_id}", e)
        job_manager.fail_job(job_id, str(e))
    finally:
        cleanup_thread_loop()

@app.route('/api/job/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    """Get current job status and result"""
    import json
    from datetime import datetime as dt

    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    def _safe_json(obj):
        if isinstance(obj, (dt, datetime)):
            return obj.isoformat()
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    # Build response - omit progress_log to keep payload smaller
    result = job.get('result')

    # DIAGNOSTIC: Log what we're returning
    if result is not None:
        fact_logger.logger.info(f"DIAGNOSTIC GET /api/job/{job_id}: result has {len(result)} keys: {list(result.keys())}")
        for k in ['content_classification', 'source_verification', 'synthesis_report', 'mode_reports', 'mode_routing']:
            v = result.get(k)
            if v is None:
                fact_logger.logger.info(f"  DIAGNOSTIC: result['{k}'] = None")
            elif isinstance(v, dict):
                fact_logger.logger.info(f"  DIAGNOSTIC: result['{k}'] = dict with {len(v)} keys: {list(v.keys())[:5]}")
            elif isinstance(v, list):
                fact_logger.logger.info(f"  DIAGNOSTIC: result['{k}'] = list with {len(v)} items")
            else:
                fact_logger.logger.info(f"  DIAGNOSTIC: result['{k}'] = {type(v).__name__}: {str(v)[:80]}")
    else:
        fact_logger.logger.info(f"DIAGNOSTIC GET /api/job/{job_id}: result is None, status={job.get('status')}")

    response_data = {
        "job_id": job_id,
        "status": job.get('status', 'unknown'),
        "result": result,
        "error": job.get('error')
    }

    # Use json.dumps with _safe_json to handle datetime/Pydantic/set/bytes
    try:
        json_str = json.dumps(response_data, default=_safe_json)
        fact_logger.logger.info(f"DIAGNOSTIC: Serialized JSON size: {len(json_str)} bytes")
        # Log a sample to verify content_classification is in the output
        if 'content_classification' in json_str:
            idx = json_str.index('content_classification')
            fact_logger.logger.info(f"DIAGNOSTIC: content_classification excerpt: ...{json_str[idx:idx+200]}...")
        else:
            fact_logger.logger.info("DIAGNOSTIC: 'content_classification' NOT FOUND in serialized JSON")
        return Response(json_str, mimetype='application/json')
    except Exception as e:
        fact_logger.logger.error(f"Job result serialization failed: {e}")
        import traceback
        fact_logger.logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "job_id": job_id,
            "status": job.get('status', 'unknown'),
            "error": f"Result serialization failed: {str(e)}"
        }), 500


@app.route('/api/job/<job_id>/stream')
def stream_job_progress(job_id: str):
    """Server-Sent Events stream for real-time progress"""
    import json
    from datetime import datetime as dt

    def _safe_json(obj):
        """JSON serializer that handles datetime and other non-standard types"""
        if isinstance(obj, (dt, datetime)):
            return obj.isoformat()
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def generate():
        job = job_manager.get_job(job_id)
        if not job:
            yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
            return

        # If job is already completed (e.g. SSE reconnect), send status signal only.
        # The frontend fetches the full result via GET /api/job/<id>.
        if job.get('status') in ['completed', 'failed', 'cancelled']:
            final_data = {
                'status': job['status'],
                'error': job.get('error')
            }
            yield f"data: {json.dumps(final_data, default=_safe_json)}\n\n"
            return

        yield f"data: {json.dumps({'status': job.get('status', 'unknown')})}\n\n"

        progress_queue = job_manager.get_progress_queue(job_id)
        if not progress_queue:
            return

        while True:
            try:
                # First, drain any pending progress items from the queue
                try:
                    progress = progress_queue.get(timeout=1)
                    # Send only the message text, not partial_result or other details
                    sse_payload = {'message': progress.get('message', '')}
                    yield f"data: {json.dumps(sse_payload)}\n\n"
                except Exception:
                    # No progress item available - check if job finished
                    current_job = job_manager.get_job(job_id)
                    if current_job and current_job.get('status') in ['completed', 'failed', 'cancelled']:
                        # Drain any remaining queued messages
                        while not progress_queue.empty():
                            try:
                                remaining = progress_queue.get_nowait()
                                sse_payload = {'message': remaining.get('message', '')}
                                yield f"data: {json.dumps(sse_payload)}\n\n"
                            except Exception:
                                break

                        # Send completion signal WITHOUT the result.
                        # Frontend fetches the result via GET /api/job/<id>.
                        final_data = {
                            'status': current_job['status'],
                            'error': current_job.get('error')
                        }
                        yield f"data: {json.dumps(final_data, default=_safe_json)}\n\n"
                        return
                    else:
                        # Job still running, send heartbeat
                        yield f"data: {json.dumps({'heartbeat': True})}\n\n"

            except GeneratorExit:
                return
            except Exception as e:
                fact_logger.logger.error(f"SSE Error: {e}")
                import traceback
                fact_logger.logger.error(f"SSE Traceback: {traceback.format_exc()}")
                return

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

# ============================================
# URL SCRAPING ENDPOINT - JOB-BASED (Avoids asyncio conflicts)
# ============================================

@app.route('/api/scrape-url', methods=['POST'])
def scrape_url():
    """
    Scrape, extract metadata, and check credibility for an article URL.

    Returns a job_id for polling (consistent with other endpoints).
    All async ops run in single background thread - MBFC lookups work properly.

    Request body:
        {
            "url": "https://example.com/article",
            "extract_metadata": true,      // optional, default true
            "check_credibility": true,     // optional, default true
            "run_mbfc_if_missing": true    // optional, default true (lazy DB population)
        }

    Returns immediately:
        {"job_id": "...", "status": "processing"}

    Poll /api/job/<job_id> for result with full enriched data.
    """
    try:
        request_json = request.get_json()
        if not request_json:
            return jsonify({"error": "Invalid request format"}), 400

        url = request_json.get('url')
        if not url:
            return jsonify({"error": "No URL provided"}), 400

        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            return jsonify({
                "error": "Invalid URL format",
                "message": "URL must start with http:// or https://"
            }), 400

        # Get optional parameters
        extract_metadata = request_json.get('extract_metadata', True)
        check_credibility = request_json.get('check_credibility', True)
        run_mbfc_if_missing = request_json.get('run_mbfc_if_missing', True)

        fact_logger.logger.info(
            " Received enriched scrape request",
            extra={
                "url": url,
                "extract_metadata": extract_metadata,
                "check_credibility": check_credibility,
                "run_mbfc_if_missing": run_mbfc_if_missing
            }
        )

        # Create job
        job_id = job_manager.create_job(url)
        fact_logger.logger.info(f" Created scrape job: {job_id}")

        # Start background processing
        threading.Thread(
            target=run_scrape_task,
            args=(job_id, url, extract_metadata, check_credibility, run_mbfc_if_missing),
            daemon=True
        ).start()

        return jsonify({
            "job_id": job_id,
            "status": "processing",
            "message": "Scraping and enrichment started"
        })

    except Exception as e:
        fact_logger.log_component_error("Scrape URL API", e)
        return jsonify({
            "error": str(e),
            "message": "An error occurred while starting the scrape job."
        }), 500


def run_scrape_task(
    job_id: str, 
    url: str, 
    extract_metadata: bool, 
    check_credibility: bool,
    run_mbfc_if_missing: bool
):
    """
    Background task for scraping URL with metadata and credibility enrichment.

    All async operations run in a SINGLE event loop, avoiding Playwright conflicts.
    MBFC lookups work here and populate Supabase for future requests.
    """
    try:
        job_manager.add_progress(job_id, f" Starting enriched scrape for {url}")

        # Run everything in one async function to keep single event loop
        async def do_enriched_scrape():
            from utils.browserless_scraper import BrowserlessScraper
            from urllib.parse import urlparse

            errors: list = []

            # Extract domain
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]

            # ============================================
            # STEP 1: Scrape content
            # ============================================
            job_manager.add_progress(job_id, " Scraping article content...")

            scraper = BrowserlessScraper(config)
            content = ""

            try:
                await scraper._initialize_browser_pool()
                results = await scraper.scrape_urls_for_facts([url])
                content = results.get(url, "")
            finally:
                await scraper.close()

            # Determine if scraping failed and why
            scrape_failed = not content or len(content.strip()) < 100
            scrape_error = None
            scrape_error_type = None  # 'paywall', 'blocked', 'empty'

            if scrape_failed:
                # Determine the specific failure reason from scraper stats
                if scraper.stats.get("paywall_detected", 0) > 0:
                    scrape_error_type = "paywall"
                    scrape_error = "This article appears to be behind a paywall. Please copy and paste the article text below."
                elif scraper.stats.get("site_failures", {}).get(domain, 0) > 0:
                    scrape_error_type = "blocked"
                    scrape_error = f"The site {domain} is blocking automated access. Please copy and paste the article text below."
                else:
                    scrape_error_type = "empty"
                    scrape_error = "Could not extract article content from this URL. The page may require JavaScript, login, or have an unusual layout. Please copy and paste the article text below."

                job_manager.add_progress(job_id, f"Content extraction failed: {scrape_error_type}")
                fact_logger.logger.warning(
                    f"Scrape failed for {url}: {scrape_error_type}",
                    extra={"url": url, "domain": domain, "error_type": scrape_error_type}
                )
            else:
                job_manager.add_progress(job_id, f"Scraped {len(content)} characters")

            # Initialize result variables
            title = None
            author = None
            publication_date = None
            publication_date_raw = None
            publication_name = None
            article_type = None
            section = None
            metadata_confidence = 0.0

            credibility_tier = 3
            credibility_rating = None
            bias_rating = None
            factual_reporting = None
            is_propaganda = False
            special_tags: list = []
            credibility_source = "unknown"
            tier_reasoning = None
            mbfc_url = None

            # ============================================
            # STEP 2: Extract metadata (only if we have content)
            # ============================================
            if extract_metadata and not scrape_failed:
                job_manager.add_progress(job_id, "Extracting article metadata...")
                try:
                    from utils.article_metadata_extractor import ArticleMetadataExtractor

                    extractor = ArticleMetadataExtractor(config)
                    metadata = await extractor.extract_metadata(url, content)

                    title = metadata.title
                    author = metadata.author
                    publication_date = metadata.publication_date
                    publication_date_raw = metadata.publication_date_raw
                    publication_name = metadata.publication_name
                    article_type = metadata.article_type
                    section = metadata.section
                    metadata_confidence = metadata.extraction_confidence

                    job_manager.add_progress(
                        job_id, 
                        f"Metadata: {title[:40] if title else 'No title'}... by {author or 'Unknown'}"
                    )

                except Exception as e:
                    fact_logger.logger.warning(f"Metadata extraction failed: {e}")
                    errors.append(f"Metadata extraction failed: {str(e)}")

            # ============================================
            # STEP 3: Check credibility (ALWAYS runs - only needs URL/domain)
            # ============================================
            if check_credibility:
                job_manager.add_progress(job_id, f"Checking source credibility for {domain}...")
                try:
                    from utils.source_credibility_service import SourceCredibilityService
                    from utils.brave_searcher import BraveSearcher

                    # Initialize with Brave searcher for MBFC lookups
                    brave_searcher = None
                    mbfc_scraper = None

                    if run_mbfc_if_missing and config.brave_api_key:
                        brave_searcher = BraveSearcher(config)
                        # Create a NEW scraper instance for MBFC (we already closed the first one)
                        mbfc_scraper = BrowserlessScraper(config)

                    service = SourceCredibilityService(
                        config=config,
                        brave_searcher=brave_searcher,
                        scraper=mbfc_scraper
                    )

                    cred = await service.check_credibility(
                        url=url,
                        run_mbfc_if_missing=run_mbfc_if_missing
                    )

                    # Close MBFC scraper if we created one
                    if mbfc_scraper:
                        await mbfc_scraper.close()

                    credibility_tier = cred.credibility_tier
                    credibility_rating = cred.credibility_rating
                    bias_rating = cred.bias_rating
                    factual_reporting = cred.factual_reporting
                    is_propaganda = cred.is_propaganda
                    special_tags = cred.special_tags
                    credibility_source = cred.source
                    tier_reasoning = cred.tier_reasoning
                    mbfc_url = cred.mbfc_url

                    if not publication_name and cred.publication_name:
                        publication_name = cred.publication_name

                    # Log what happened
                    if cred.source == "supabase":
                        job_manager.add_progress(job_id, f"Found {domain} in database (Tier {credibility_tier})")
                    elif cred.source == "mbfc":
                        job_manager.add_progress(job_id, f"MBFC lookup complete, saved to database (Tier {credibility_tier})")
                    else:
                        job_manager.add_progress(job_id, f"No credibility data found for {domain} (Tier 3 default)")

                except Exception as e:
                    fact_logger.logger.warning(f"Credibility check failed: {e}")
                    errors.append(f"Credibility check failed: {str(e)}")

            # Build tier descriptions (used by both success and failure paths)
            tier_descriptions = {
                1: "Highly Credible - Official sources, major wire services, highly reputable news",
                2: "Credible - Reputable mainstream media with strong factual reporting",
                3: "Mixed - Requires verification, may have bias or mixed factual reporting",
                4: "Low Credibility - Significant bias issues or poor factual reporting",
                5: "Unreliable - Propaganda, conspiracy, or known disinformation source"
            }

            # If scrape failed, return with credibility data but flag the failure
            if scrape_failed:
                return {
                    "success": False,
                    "scrape_failed": True,
                    "scrape_error": scrape_error,
                    "scrape_error_type": scrape_error_type,
                    "url": url,
                    "domain": domain,
                    "publication_name": publication_name,

                    # Credibility data (gathered even though scrape failed)
                    "credibility": {
                        "tier": credibility_tier,
                        "tier_description": tier_descriptions.get(credibility_tier, "Unknown"),
                        "rating": credibility_rating,
                        "bias_rating": bias_rating,
                        "factual_reporting": factual_reporting,
                        "is_propaganda": is_propaganda,
                        "special_tags": special_tags,
                        "source": credibility_source,
                        "reasoning": tier_reasoning,
                        "mbfc_url": mbfc_url
                    },

                    "errors": errors
                }

            # Fallback title extraction
            if not title:
                lines = content.strip().split('\n')
                if lines:
                    first_line = lines[0].strip()
                    if first_line.startswith('#'):
                        title = first_line.lstrip('#').strip()[:200]

            # Build success result
            return {
                "success": True,
                "url": url,
                "domain": domain,

                # Content
                "content": content,
                "content_length": len(content),

                # Metadata
                "title": title,
                "author": author,
                "publication_date": publication_date,
                "publication_date_raw": publication_date_raw,
                "publication_name": publication_name,
                "article_type": article_type,
                "section": section,
                "metadata_confidence": metadata_confidence,

                # Credibility
                "credibility": {
                    "tier": credibility_tier,
                    "tier_description": tier_descriptions.get(credibility_tier, "Unknown"),
                    "rating": credibility_rating,
                    "bias_rating": bias_rating,
                    "factual_reporting": factual_reporting,
                    "is_propaganda": is_propaganda,
                    "special_tags": special_tags,
                    "source": credibility_source,
                    "reasoning": tier_reasoning,
                    "mbfc_url": mbfc_url
                },

                # Processing info
                "errors": errors
            }

        # Run everything in one thread with one event loop
        result = run_async_in_thread(do_enriched_scrape())

        job_manager.add_progress(job_id, " Enriched scrape complete!")
        job_manager.complete_job(job_id, result)

        fact_logger.logger.info(f" Scrape job {job_id} completed successfully")

    except Exception as e:
        fact_logger.log_component_error(f"Scrape Job {job_id}", e)
        job_manager.fail_job(job_id, str(e))

    finally:
        cleanup_thread_loop()


# ============================================
# CREDIBILITY CHECK ENDPOINT
# ============================================

@app.route('/api/check-credibility', methods=['POST'])
def check_credibility():
    """
    Check credibility of a publication without scraping content.

    If not in Supabase cache and run_mbfc_if_missing=true (default),
    will do MBFC lookup and save to database for future requests.
    """
    try:
        request_json = request.get_json()
        if not request_json:
            return jsonify({"error": "Invalid request format"}), 400

        url = request_json.get('url')
        if not url:
            return jsonify({"error": "No URL provided"}), 400

        run_mbfc = request_json.get('run_mbfc_if_missing', True)  # Default true for lazy population

        # Create job (credibility check with MBFC can take time)
        job_id = job_manager.create_job(url)

        threading.Thread(
            target=run_credibility_task,
            args=(job_id, url, run_mbfc),
            daemon=True
        ).start()

        return jsonify({
            "job_id": job_id,
            "status": "processing",
            "message": "Credibility check started"
        })

    except Exception as e:
        fact_logger.log_component_error("Credibility Check API", e)
        return jsonify({"error": str(e)}), 500


def run_credibility_task(job_id: str, url: str, run_mbfc: bool):
    """Background task for credibility check with MBFC lookup"""
    try:
        async def do_check():
            from utils.source_credibility_service import SourceCredibilityService
            from utils.browserless_scraper import BrowserlessScraper
            from utils.brave_searcher import BraveSearcher

            brave_searcher = None
            scraper = None

            if run_mbfc and config.brave_api_key:
                brave_searcher = BraveSearcher(config)
                scraper = BrowserlessScraper(config)

            try:
                service = SourceCredibilityService(
                    config=config,
                    brave_searcher=brave_searcher,
                    scraper=scraper
                )

                result = await service.check_credibility(
                    url=url,
                    run_mbfc_if_missing=run_mbfc
                )

                return result
            finally:
                if scraper:
                    await scraper.close()

        cred_result = run_async_in_thread(do_check())

        tier_descriptions = {
            1: "Highly Credible - Official sources, major wire services, highly reputable news",
            2: "Credible - Reputable mainstream media with strong factual reporting",
            3: "Mixed - Requires verification, may have bias or mixed factual reporting",
            4: "Low Credibility - Significant bias issues or poor factual reporting",
            5: "Unreliable - Propaganda, conspiracy, or known disinformation source"
        }

        result = {
            "success": True,
            "url": url,
            "domain": cred_result.domain,
            "publication_name": cred_result.publication_name,
            "credibility": {
                "tier": cred_result.credibility_tier,
                "tier_description": tier_descriptions.get(cred_result.credibility_tier, "Unknown"),
                "rating": cred_result.credibility_rating,
                "bias_rating": cred_result.bias_rating,
                "factual_reporting": cred_result.factual_reporting,
                "is_propaganda": cred_result.is_propaganda,
                "special_tags": cred_result.special_tags,
                "source": cred_result.source,
                "reasoning": cred_result.tier_reasoning,
                "mbfc_url": cred_result.mbfc_url
            }
        }

        job_manager.complete_job(job_id, result)

    except Exception as e:
        fact_logger.log_component_error(f"Credibility Job {job_id}", e)
        job_manager.fail_job(job_id, str(e))
    finally:
        cleanup_thread_loop()

@app.route('/api/job/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id: str):
    """Cancel a running job"""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    job_manager.cancel_job(job_id)
    return jsonify({
        "job_id": job_id,
        "status": "cancelled",
        "message": "Job cancellation requested"
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "llm_orchestrator": llm_interpretation_orchestrator is not None,
        "web_search_orchestrator": web_search_orchestrator is not None,
        "bias_orchestrator": bias_orchestrator is not None
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'

    fact_logger.logger.info(f" Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)